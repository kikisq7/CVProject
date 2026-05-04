import sys
import torch
import json
import os
import logging
import numpy as np
import torch.distributed as dist
from datasets import load_from_disk
from transformers import (
    Seq2SeqTrainingArguments,
    TrainerState,
    HfArgumentParser, 
    AutoConfig,
    AutoProcessor,
    AutoModel,
    set_seed,
)
from transformers.trainer import TRAINER_STATE_NAME
from accelerate.logging import get_logger
from legato.config import DataArguments, ModelArguments
from legato.models import (
    LegatoConfig,
    LegatoModel,
    apply_peft,
    load_peft_adapters,
    count_trainable_parameters,
)
from legato.trainer import LegatoTrainer
from legato.metrics import compute_error_rates


def main():
    parser = HfArgumentParser((Seq2SeqTrainingArguments, DataArguments, ModelArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        training_args, data_args, model_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        training_args, data_args, model_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

    logging.basicConfig(level=training_args.log_level.upper(), format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s')
    logger = get_logger(__name__)

    torch.set_float32_matmul_precision('high')
    if training_args.torch_compile:
        # Increase the cache size limit to avoid recompilation
        # Large RAM usage may occur if the cache size is too large
        torch._dynamo.config.cache_size_limit = 256

    #### Load dataset
    dataset = load_from_disk(data_args.dataset_path)
    for split, mini_file in [("val", data_args.mini_val_file), ("test", data_args.mini_test_file)]:
        if mini_file:
            logger.info(f"Using mini {split} set: {mini_file}")
            with open(mini_file, "r") as f:
                filenames = json.load(f)
            dataset[split] = dataset[split].select(
                [dataset[split]['filename'].index(filename) for filename in filenames]
            )

    if data_args.dummy_data:
        logger.info("Using dummy data (32 items) for debugging only...")
        dataset['train'] = dataset['train'].select(range(32))
        dataset['val'] = dataset['val'].select(range(32))
        dataset['test'] = dataset['test'].select(range(32))

    #### Load model and tokenizer
    set_seed(training_args.seed)

    # Optional bitsandbytes quantization for the frozen vision encoder.
    # Used to fit legato-small into a 16 GB T4 with PEFT enabled (QLoRA-style).
    quantization_config = None
    if model_args.load_in_4bit or model_args.load_in_8bit:
        from transformers import BitsAndBytesConfig

        if model_args.load_in_4bit:
            compute_dtype = getattr(torch, model_args.bnb_4bit_compute_dtype)
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
        else:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model_load_kwargs = {}
    if quantization_config is not None:
        model_load_kwargs["quantization_config"] = quantization_config

    if model_args.pretrained_model:
        model = AutoModel.from_pretrained(model_args.pretrained_model, **model_load_kwargs)
    else:
        config = AutoConfig.from_pretrained(model_args.model_config)
        model = LegatoModel(config)

    if quantization_config is not None:
        try:
            from peft import prepare_model_for_kbit_training

            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=getattr(training_args, "gradient_checkpointing", False),
            )
            logger.info(
                f"Loaded model with bitsandbytes "
                f"{'4-bit NF4' if model_args.load_in_4bit else '8-bit'} quantization."
            )
        except ImportError:
            logger.warning("peft not installed; skipping prepare_model_for_kbit_training.")

    # Attach PEFT adapters if requested. When an adapter checkpoint is
    # provided, load it in trainable mode for resumed training or in
    # inference mode when only evaluating / predicting.
    if model_args.peft_adapter_path:
        is_trainable = training_args.do_train
        model = load_peft_adapters(
            model, model_args.peft_adapter_path, is_trainable=is_trainable
        )
        logger.info(f"Loaded PEFT adapters from {model_args.peft_adapter_path}")
    elif model_args.peft_strategy and model_args.peft_strategy.lower() != "none":
        model = apply_peft(
            model,
            strategy=model_args.peft_strategy,
            vision_rank=model_args.vision_rank,
            decoder_rank=model_args.decoder_rank,
            alpha=model_args.peft_alpha,
            dropout=model_args.peft_dropout,
        )
        stats = count_trainable_parameters(model)
        logger.info(
            f"Applied PEFT strategy={model_args.peft_strategy} "
            f"(vision_rank={model_args.vision_rank}, "
            f"decoder_rank={model_args.decoder_rank}, alpha={model_args.peft_alpha}). "
            f"Trainable: {stats['trainable_params']:,} / {stats['total_params']:,} "
            f"({100 * stats['trainable_ratio']:.3f}%)."
        )

    processor = AutoProcessor.from_pretrained(model_args.model_config)
    tokenizer = processor.tokenizer

    def get_metric_target(examples):
        return {
            'label_ids': processor(text=examples['transcription'], add_special_tokens=False, verbose=False, truncation=False)['input_ids'],
        }

    if not training_args.do_predict:
        metric_targets = dataset['val'].map(
            get_metric_target, 
            remove_columns=dataset['val'].column_names,
            num_proc=training_args.dataloader_num_workers, 
            batched=True
        ).to_dict()
    else:
        metric_targets = dataset['test'].map(
            get_metric_target, 
            remove_columns=dataset['test'].column_names,
            num_proc=training_args.dataloader_num_workers, 
            batched=True
        ).to_dict() if 'transcription' in dataset['test'].column_names else None

    # We don't predict image tokens or padding tokens
    tokens_to_mask = torch.tensor([*tokenizer.additional_special_tokens_ids, tokenizer.pad_token_id])

    def collate_fn(examples):
        outputs = processor(
            images=[example['image'] for example in examples],
            text=[example['transcription'] for example in examples],
            return_num_tiles=True,
            truncation=True,
            padding="max_length",
            return_tensors='pt',
        ) # pad to max length to reduce torch compilation overhead
        gen_outputs = processor(
            num_tiles=outputs.pop('num_tiles'), # Reuse num_tiles to save computation
            truncation=True,
            padding=True,
            return_tensors='pt',
        )
        outputs.update({
            f'gen_{k}': outputs[k] if k not in gen_outputs else gen_outputs[k]
            for k in outputs
        })
        outputs['labels'] = outputs['input_ids'].clone().masked_fill(
            torch.isin(outputs['input_ids'], tokens_to_mask), -100
        ) # We don't predict image tokens or padding tokens
        return outputs

    special_tokens = [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id, -100]
    def remove_special_tokens(array):
        masks = np.isin(array, special_tokens, invert=True)
        return [a[mask] for a, mask in zip(array, masks)]

    def metric_fn(p):
        preds = remove_special_tokens(p.predictions)
        results = [compute_error_rates(
            tokenizer, training_args.dataloader_num_workers, *metric_targets.values(), preds
        )] if training_args.process_index == 0 else [None]
        dist.broadcast_object_list(results, src=0)
        return results[0]

    # HF Trainer.__init__ refuses a purely quantized base model (no PEFT) even
    # for predict-only runs. For zero-shot baseline eval we bypass Trainer and
    # run a minimal generation loop instead.
    def _is_peft_model(m):
        try:
            from peft import PeftModel
        except ImportError:
            PeftModel = ()
        if PeftModel and isinstance(m, PeftModel):
            return True
        return getattr(m, "_hf_peft_config_loaded", False) or hasattr(m, "peft_config")

    predict_only = (
        training_args.do_predict
        and not training_args.do_train
        and not training_args.do_eval
    )
    quantized_no_peft = (
        getattr(model, "is_quantized", False)
        or quantization_config is not None
    ) and not _is_peft_model(model)

    if predict_only and quantized_no_peft:
        from torch.utils.data import DataLoader
        try:
            from tqdm.auto import tqdm
        except ImportError:  # tqdm is a transformers dep; fallback just in case
            def tqdm(x, **_):
                return x

        logger.info(
            "Running quantized predict-only path (bypassing HF Trainer because "
            "the base model is purely quantized with no PEFT adapters)."
        )

        model.eval()
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        loader = DataLoader(
            dataset['test'],
            batch_size=training_args.per_device_eval_batch_size,
            collate_fn=collate_fn,
            num_workers=training_args.dataloader_num_workers,
        )

        gen_kwargs = {}
        if training_args.generation_max_length is not None:
            gen_kwargs["max_length"] = training_args.generation_max_length
        if training_args.generation_num_beams is not None:
            gen_kwargs["num_beams"] = training_args.generation_num_beams

        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        all_preds = []
        for batch in tqdm(loader, desc="predict"):
            gen_inputs = {
                k.replace("gen_", ""): (v.to(device) if hasattr(v, "to") else v)
                for k, v in batch.items()
                if k.startswith("gen_")
            }
            gen_inputs.pop("labels", None)
            length = gen_inputs["input_ids"].shape[1]
            with torch.inference_mode():
                generated = model.generate(**gen_inputs, **gen_kwargs)
            # Mirror Seq2SeqTrainer.prediction_step: keep the last prompt
            # token + everything generated after it.
            generated = generated[:, length - 1:].detach().cpu().numpy()
            all_preds.append(generated)

        if all_preds:
            max_len = max(p.shape[1] for p in all_preds)
            padded = []
            for p in all_preds:
                if p.shape[1] < max_len:
                    pad = np.full(
                        (p.shape[0], max_len - p.shape[1]), pad_id, dtype=p.dtype
                    )
                    p = np.concatenate([p, pad], axis=1)
                padded.append(p)
            predictions = np.concatenate(padded, axis=0)
        else:
            predictions = np.zeros((0, 0), dtype=np.int64)

        os.makedirs(training_args.output_dir, exist_ok=True)
        abc_outputs = processor.batch_decode(predictions, skip_special_tokens=True)
        preds = remove_special_tokens(predictions)
        with open(os.path.join(training_args.output_dir, "test_predictions.json"), "w") as f:
            json.dump(
                {'abc_transcription': abc_outputs,
                 'tokens': [p.tolist() for p in preds]},
                f,
            )

        if metric_targets:
            results = compute_error_rates(
                tokenizer,
                training_args.dataloader_num_workers,
                *metric_targets.values(),
                preds,
            )
            logger.info(f"Test metrics: {results}")
        return

    trainer = LegatoTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=dataset['train'] if 'train' in dataset else None,
        eval_dataset=dataset['val'] if 'val' in dataset else None,
        compute_metrics=metric_fn,
    )

    def _unwrap_and_save_model(trainer, output_dir):
        if trainer.is_world_process_zero():
            logger.info("Unwrapping the model...")
            unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
            logger.info("Model unwrapped.")
            unwrapped_model.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)
            logger.info(f"Model and Processor saved. to {output_dir}")

    #### Train
    if training_args.do_train:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    #### Evaluate
    if training_args.do_eval:
        if not training_args.do_train and not model_args.pretrained_model: # if no training is done and no pretrained model is provided, evaluate all checkpoints
            ckpts = [ckpt for ckpt in os.listdir(training_args.output_dir) if ckpt.startswith("checkpoint")]
            assert len(ckpts) > 0, f"No checkpoints found in {training_args.output_dir}"
            best_ckpt, best_result = None, None
            for ckpt in sorted(ckpts):
                logger.info(f"Evaluating checkpoint {ckpt}...")
                trainer._load_from_checkpoint(os.path.join(training_args.output_dir, ckpt))
                trainer.state = TrainerState.load_from_json(os.path.join(training_args.output_dir, ckpt, TRAINER_STATE_NAME))
                trainer.state.init_training_references(trainer, trainer.state.max_steps, trainer.state.num_train_epochs, None)
                trainer._load_callback_state()
                result = trainer.evaluate()
                if best_result is None or result['eval_SER'] < best_result['eval_SER']:
                    best_result, best_ckpt = result, ckpt
                trainer.log_metrics("eval", result)

            logger.info(f"Best checkpoint: {best_ckpt}")
            trainer._load_from_checkpoint(os.path.join(training_args.output_dir, best_ckpt))

        else:
            best_result = trainer.evaluate()

        _unwrap_and_save_model(trainer, training_args.output_dir)
        final_val_results = {k.replace("eval_", "eval_best_"): v for k, v in best_result.items() if k.startswith("eval_")}
        trainer.log_metrics("best eval", final_val_results)
        trainer.log(final_val_results)

    #### Predict
    if training_args.do_predict:
        outputs = trainer.predict(dataset['test'])

        if trainer.is_world_process_zero():
            abc_outputs = processor.batch_decode(outputs.predictions, skip_special_tokens=True)
            preds = remove_special_tokens(outputs.predictions)
            with open(os.path.join(training_args.output_dir, "test_predictions.json"), "w") as f:
                json.dump({'abc_transcription': abc_outputs, 'tokens': [p.tolist() for p in preds]}, f)

            if metric_targets:
                results = compute_error_rates(
                    tokenizer, training_args.dataloader_num_workers, *metric_targets.values(), preds
                )
                trainer.log_metrics("test", results)
    

if __name__ == "__main__":
    try:
        main()
    finally:
        # Cleanup distributed process group
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

