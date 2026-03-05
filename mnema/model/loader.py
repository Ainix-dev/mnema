# mnema/model/loader.py
"""
Convenience loader — loads any HuggingFace CausalLM with LoRA.
Optional — users can load their own model and pass it to MNEMA().
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from ..config import MNEMAConfig


def load_model(
    model_id:     str,
    adapter_path: str = None,
    config:       MNEMAConfig = None,
):
    """
    Load a frozen CausalLM with LoRA adapter.

    Parameters
    ----------
    model_id     : HuggingFace model ID or local path
    adapter_path : path to saved LoRA adapter (optional)
    config       : MNEMAConfig — uses defaults if None

    Returns
    -------
    (model, tokenizer)
    """
    cfg = config or MNEMAConfig()

    print(f"[MNEMA] Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Quantization
    quant_cfg = None
    if cfg.load_in_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    print(f"[MNEMA] Loading base model (frozen)...")
    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_cfg,
        torch_dtype=torch.bfloat16 if not cfg.load_in_4bit else None,
        device_map="auto",
        trust_remote_code=True,
    )

    # Freeze all base weights
    for param in base.parameters():
        param.requires_grad = False

    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()

    # Attach LoRA
    if adapter_path and os.path.exists(adapter_path):
        print(f"[MNEMA] Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(base, adapter_path, is_trainable=True)
    else:
        print(f"[MNEMA] Creating fresh LoRA adapter...")
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=cfg.lora_target_modules,
            layers_to_transform=cfg.lora_layers_to_transform,
            lora_dropout=cfg.lora_dropout,
            bias="none",
        )
        model = get_peft_model(base, lora_cfg)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[MNEMA] Trainable: {trainable:,} / {total:,} "
          f"({100*trainable/total:.2f}%)")

    model.eval()
    return model, tokenizer
