import gc
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(model_path: str, for_training: bool):
    path = Path(model_path)
    index = path / "model.safetensors.index.json"
    if index.exists():
        import json
        weight_files = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
        missing = [name for name in weight_files if not (path / name).exists()]
        if missing:
            raise FileNotFoundError(f"Incomplete local model in {path}; missing: {missing}")
    device = get_device()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.config.use_cache = not for_training
    if for_training:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    return model


def add_lora(model, config):
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.target_modules),
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"LoRA trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)")
    return model


def load_lora_model(config):
    return add_lora(load_base_model(config.model_path, for_training=True), config)


def load_saved_adapter(config, adapter_dir: str):
    base = load_base_model(config.model_path, for_training=False)
    model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
    model.eval()
    return model


def lora_state_dict(model, device=None) -> dict[str, torch.Tensor]:
    state = {}
    for key, value in model.state_dict().items():
        if "lora_" in key:
            tensor = value.detach().clone()
            state[key] = tensor.to(device) if device is not None else tensor
    return state


def release_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
