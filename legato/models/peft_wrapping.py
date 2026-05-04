"""PEFT (LoRA / DoRA / DLoRA) wrapping for LegatoModel.

The Legato model is a vision-language model whose vision encoder is a
``MllamaVisionModel`` (``model.vision_model``) and whose text decoder is a
``MllamaForCausalLM`` (``model.language_model``). For handwritten fine-tuning
we want an asymmetric "DLoRA" adaptation:

* DoRA on the vision encoder's attention and MLP projections, since the
  distribution shift (engraved -> handwritten glyphs) is primarily visual.
* Plain LoRA on the decoder's cross-attention ``q/k/v/o`` projections, which
  only need to recalibrate how the decoder reads the adapted encoder
  embeddings. Decoder self-attention, token embeddings, and the LM head stay
  frozen to avoid catastrophic forgetting of ABC syntax.

Two separate PEFT adapters are registered so that DoRA (``use_dora=True``)
can be combined with plain LoRA (``use_dora=False``) on disjoint module sets
within one model.
"""

from typing import Optional

from peft import LoraConfig, PeftModel, get_peft_model


VISION_ADAPTER_NAME = "vision_dora"
DECODER_ADAPTER_NAME = "decoder_lora"

VISION_TARGET_REGEX = r".*vision_model\..*\.(q_proj|k_proj|v_proj|o_proj|fc1|fc2)$"
DECODER_CROSS_ATTN_REGEX = r".*language_model\..*\.cross_attn\.(q_proj|k_proj|v_proj|o_proj)$"


def _build_vision_config(rank: int, alpha: int, dropout: float, use_dora: bool) -> LoraConfig:
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=VISION_TARGET_REGEX,
        lora_dropout=dropout,
        bias="none",
        use_dora=use_dora,
        init_lora_weights=True,
    )


def _build_decoder_config(rank: int, alpha: int, dropout: float, use_dora: bool) -> LoraConfig:
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=DECODER_CROSS_ATTN_REGEX,
        lora_dropout=dropout,
        bias="none",
        use_dora=use_dora,
        init_lora_weights=True,
    )


def apply_peft(
    model,
    strategy: str = "dlora",
    vision_rank: int = 16,
    decoder_rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.0,
) -> PeftModel:
    """Attach PEFT adapters to a ``LegatoModel`` in-place and return a ``PeftModel``.

    Parameters
    ----------
    model: LegatoModel
        The loaded Legato model whose vision backbone is already materialized.
    strategy: str
        One of:

        * ``"none"``: return the model unchanged (caller handles full-FT).
        * ``"lora"``: symmetric LoRA on vision + decoder cross-attn.
        * ``"dora"``: symmetric DoRA on vision + decoder cross-attn.
        * ``"dlora"``: DoRA on vision, plain LoRA on decoder cross-attn (ours).
    vision_rank, decoder_rank, alpha, dropout:
        Standard LoRA hyperparameters.
    """
    strategy = strategy.lower()
    if strategy == "none":
        return model

    if strategy not in {"lora", "dora", "dlora"}:
        raise ValueError(
            f"Unknown peft_strategy={strategy!r}. "
            "Expected one of: none, lora, dora, dlora."
        )

    vision_uses_dora = strategy in {"dora", "dlora"}
    decoder_uses_dora = strategy == "dora"

    vision_config = _build_vision_config(vision_rank, alpha, dropout, vision_uses_dora)
    decoder_config = _build_decoder_config(decoder_rank, alpha, dropout, decoder_uses_dora)

    # PEFT freezes all base weights and only unfreezes the newly injected
    # adapter params. We register the vision adapter first and then add the
    # decoder adapter on top of the resulting PeftModel.
    peft_model = get_peft_model(model, vision_config, adapter_name=VISION_ADAPTER_NAME)
    peft_model.add_adapter(DECODER_ADAPTER_NAME, decoder_config)
    _activate_adapters(peft_model, [VISION_ADAPTER_NAME, DECODER_ADAPTER_NAME])

    _ensure_adapters_trainable(peft_model)
    return peft_model


def _activate_adapters(peft_model: PeftModel, adapter_names: list) -> None:
    """Activate multiple adapters simultaneously.

    ``PeftModel.set_adapter`` (the user-facing wrapper) only accepts a single
    adapter name in current ``peft`` releases; passing a list raises
    ``TypeError: unhashable type: 'list'``. The underlying tuner
    (``peft_model.base_model``, e.g. ``LoraModel``) does accept a list, which
    is the documented way to activate multiple disjoint LoRA/DoRA adapters
    (as in our DLoRA setup, where vision and decoder adapters target
    non-overlapping modules).
    """
    base = getattr(peft_model, "base_model", None)
    if base is not None and hasattr(base, "set_adapter"):
        base.set_adapter(adapter_names)
    else:  # fall back to single-adapter activation for older peft layouts
        for name in adapter_names:
            peft_model.set_adapter(name)


def _ensure_adapters_trainable(peft_model: PeftModel) -> None:
    """Guarantee that only LoRA/DoRA adapter params have ``requires_grad=True``.

    ``LegatoModel.__init__`` and ``from_pretrained`` explicitly freeze the
    vision model. After PEFT wraps the model, the newly injected adapter
    parameters sit inside the vision model, so we must re-enable grads on
    them while keeping the base weights frozen.
    """
    for name, param in peft_model.named_parameters():
        if _is_adapter_param(name):
            param.requires_grad = True
        else:
            param.requires_grad = False


def _is_adapter_param(name: str) -> bool:
    adapter_markers = (
        "lora_A",
        "lora_B",
        "lora_embedding_A",
        "lora_embedding_B",
        "lora_magnitude_vector",  # DoRA-specific magnitude vector
    )
    return any(marker in name for marker in adapter_markers)


def count_trainable_parameters(model) -> dict:
    """Return a small summary dict for logging."""
    trainable, total = 0, 0
    for _, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_ratio": trainable / max(total, 1),
    }


def load_peft_adapters(base_model, adapter_path: str, is_trainable: Optional[bool] = False) -> PeftModel:
    """Re-load adapters from ``adapter_path`` onto a base ``LegatoModel``.

    ``adapter_path`` should be a directory produced by
    ``PeftModel.save_pretrained`` (contains ``adapter_config.json`` and
    ``adapter_model.safetensors`` either at the top level or under per-adapter
    subdirectories for multi-adapter checkpoints).
    """
    import os

    # Multi-adapter checkpoint: subdirectories per adapter name.
    subdirs = [
        d
        for d in (VISION_ADAPTER_NAME, DECODER_ADAPTER_NAME)
        if os.path.isdir(os.path.join(adapter_path, d))
    ]
    if subdirs:
        peft_model = PeftModel.from_pretrained(
            base_model,
            os.path.join(adapter_path, subdirs[0]),
            adapter_name=subdirs[0],
            is_trainable=bool(is_trainable),
        )
        for extra in subdirs[1:]:
            peft_model.load_adapter(
                os.path.join(adapter_path, extra),
                adapter_name=extra,
                is_trainable=bool(is_trainable),
            )
        _activate_adapters(peft_model, subdirs)
        return peft_model

    # Single-adapter layout.
    return PeftModel.from_pretrained(
        base_model,
        adapter_path,
        is_trainable=bool(is_trainable),
    )


__all__ = [
    "apply_peft",
    "load_peft_adapters",
    "count_trainable_parameters",
    "VISION_ADAPTER_NAME",
    "DECODER_ADAPTER_NAME",
]
