from dataclasses import dataclass, field
from typing import Union, Optional

@dataclass
class DataArguments:
    dataset_path: str
    mini_val_file: str = field(
        default=None,
        metadata={"help": "Path to the mini validation file. If not provided, the the whole validation set will be used."},
    )
    mini_test_file: str = field(
        default=None,
        metadata={"help": "Path to the mini test file. If not provided, the the whole test set will be used."},
    )
    dummy_data: bool = field(
        default=False,
        metadata={"help": "Use dummy data (32 items) for testing purposes."},
    )

@dataclass
class ModelArguments:
    model_config: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the model configuration file."},
    )
    pretrained_model: Optional[str] = field(
        default=None,
        metadata={"help": "Set the path to load a pretrained model before training or evaluation."},
    )
    peft_strategy: str = field(
        default="none",
        metadata={
            "help": (
                "Parameter-efficient fine-tuning strategy. One of "
                "{none, lora, dora, dlora}. 'dlora' applies DoRA to the "
                "vision encoder and plain LoRA to the decoder "
                "cross-attention layers (ours)."
            )
        },
    )
    vision_rank: int = field(
        default=16,
        metadata={"help": "LoRA/DoRA rank for the vision encoder adapter."},
    )
    decoder_rank: int = field(
        default=16,
        metadata={"help": "LoRA rank for the decoder cross-attention adapter."},
    )
    peft_alpha: int = field(
        default=32,
        metadata={"help": "lora_alpha used for both adapters."},
    )
    peft_dropout: float = field(
        default=0.0,
        metadata={"help": "LoRA dropout probability."},
    )
    peft_adapter_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Optional path to a previously saved adapter checkpoint to "
                "re-load (e.g. for evaluation / prediction). If set, "
                "`pretrained_model` is taken as the frozen base model and "
                "adapters are attached on top of it."
            )
        },
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={
            "help": (
                "Load the frozen vision encoder in NF4 4-bit precision via "
                "bitsandbytes (QLoRA-style). Required to fit legato-small "
                "into a 16 GB T4 with PEFT training enabled."
            )
        },
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Alternative to load_in_4bit; coarser but more compatible."},
    )
    bnb_4bit_compute_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Compute dtype used by bitsandbytes 4-bit kernels."},
    )
