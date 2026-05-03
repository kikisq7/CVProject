import torch
import argparse
import os
import json
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk
from PIL import Image
from legato.models import LegatoModel
from transformers import AutoProcessor, GenerationConfig

def remove_special_tokens(arrays, special_tokens):
    outputs = []
    for array in arrays:
        outputs.append([tok for tok in array if tok not in special_tokens])
    return outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for Legato model. Output to standard output.")
    parser.add_argument("--model_path", type=str, default="guangyangmusic/legato", help="Path to the trained model")
    parser.add_argument("--processor_path", type=str, default=None, help="Path to the processor (tokenizer)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image or directory containing images for inference")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save the predictions")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing images")
    parser.add_argument("--beam_size", type=int, default=10, help="Beam size for generation")
    parser.add_argument("--max_length", type=int, default=2048, help="Generation max length")
    parser.add_argument("--fp16", action='store_true', help="Use fp16 precision for inference")
    parser.add_argument("--load_in_4bit", action='store_true', help="Load the frozen vision encoder in 4-bit (QLoRA-style). Required to fit on T4.")
    parser.add_argument("--peft_adapter_path", type=str, default=None, help="Optional PEFT adapter directory to mount on top of model_path.")
    parser.add_argument("--max_images", type=int, default=None, help="Optional cap on number of images processed.")
    parser.add_argument("--print_predictions", action='store_true', help="Print each (filename, prediction) to stdout as we go.")

    args = parser.parse_args()

    if args.processor_path is None:
        args.processor_path = args.model_path

    # Optional bitsandbytes quantization for the frozen encoder.
    model_kwargs = {}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16 if args.fp16 else torch.bfloat16,
        )

    # Load the model and processor
    model = LegatoModel.from_pretrained(args.model_path, **model_kwargs)
    if args.peft_adapter_path:
        from legato.models import load_peft_adapters

        model = load_peft_adapters(model, args.peft_adapter_path, is_trainable=False)
        print(f"[inference] loaded PEFT adapters from {args.peft_adapter_path}")

    processor = AutoProcessor.from_pretrained(args.processor_path)
    generation_config = GenerationConfig(max_length=args.max_length, num_beams=args.beam_size, repetition_penalty=1.1)

    args.image_path = os.path.abspath(args.image_path)

    # Load the image(s) and capture per-sample filenames for diagnostic prints.
    filenames = []
    if os.path.isdir(args.image_path):
        listed = sorted(os.listdir(args.image_path))
        if all(img.endswith(('.png', '.jpg', '.jpeg')) for img in listed):
            imgs = []
            for img_path in listed:
                imgs.append(Image.open(os.path.join(args.image_path, img_path)).convert("RGB"))
                filenames.append(img_path)
        else:
            dataset = load_from_disk(args.image_path)
            imgs = list(dataset['image'])
            filenames = list(dataset.get('filename', [f"sample_{i}" for i in range(len(imgs))]))
    else:
        imgs = [Image.open(args.image_path).convert("RGB")]
        filenames = [os.path.basename(args.image_path)]

    if args.max_images is not None:
        imgs = imgs[: args.max_images]
        filenames = filenames[: args.max_images]

    # 4-bit / 8-bit models are already on the GPU and cannot be moved by .to().
    is_quantized = bool(getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False))
    if not is_quantized:
        model = model.to(device=args.device)
        if args.fp16:
            model = model.half()

    output_tokens = []
    abc_so_far = []
    for i in tqdm(range(0, len(imgs), args.batch_size), desc="Predicting..."):
        batch_imgs = imgs[i:min(i + args.batch_size, len(imgs))]
        batch_names = filenames[i:min(i + args.batch_size, len(imgs))]
        inputs = processor(
            images=batch_imgs,
            truncation=True,
            return_tensors='pt'
        )
        inputs = {k: v.to(args.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, generation_config=generation_config, use_model_defaults=False)

        output_tokens.extend(outputs.tolist())
        if args.print_predictions:
            decoded = processor.batch_decode(outputs.tolist(), skip_special_tokens=True)
            for fname, abc in zip(batch_names, decoded):
                abc_so_far.append(abc)
                print(f"\n=== {fname} ===\n{abc}\n")

    abc_outputs = processor.batch_decode(output_tokens, skip_special_tokens=True)

    special_tokens = processor.tokenizer.all_special_ids 
    preds = remove_special_tokens(output_tokens, special_tokens)

    if not os.path.isdir(args.image_path):
        print(abc_outputs[0])

    if args.output_path is None:
        args.output_path = os.path.dirname(args.image_path) 

    if os.path.isdir(args.output_path):
        output_file = os.path.join(args.output_path, f"{os.path.basename(args.image_path).split('.')[0]}_{args.model_path.replace('/', '_')}_abc.json")
    else:
        output_file = args.output_path
    with open(output_file, "w") as f:
        json.dump({'abc_transcription': abc_outputs, 'tokens': preds}, f)

    print("Inference completed. Output saved to:", output_file)
