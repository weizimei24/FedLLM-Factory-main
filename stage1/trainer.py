import math
import random

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from stage1.data import QACollator, QADataset


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_phase(model, rows: list[dict], tokenizer, config, phase_seed: int, label: str) -> dict:
    set_seed(phase_seed)
    dataset = QADataset(rows, tokenizer, config.max_length)
    generator = torch.Generator().manual_seed(phase_seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=QACollator(tokenizer.pad_token_id),
    )
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=config.learning_rate)
    model.train()
    model.config.use_cache = False
    optimizer.zero_grad(set_to_none=True)
    running_nll = 0.0
    running_tokens = 0
    pending = 0

    for step, batch in enumerate(loader, start=1):
        batch = {key: value.to(model.device) for key, value in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            output = model(**batch)
            loss = output.loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss in {label} at batch {step}: {loss.item()}")
        valid_tokens = int((batch["labels"][:, 1:] != -100).sum().item())
        running_nll += loss.detach().float().item() * valid_tokens
        running_tokens += valid_tokens
        (loss / config.grad_accum).backward()
        pending += 1

        if pending == config.grad_accum:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            pending = 0

    if pending:
        # Make the final partial accumulation an average over its actual size.
        scale = config.grad_accum / pending
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(scale)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    mean_nll = running_nll / max(running_tokens, 1)
    result = {
        "label": label,
        "samples": len(rows),
        "answer_tokens": running_tokens,
        "train_nll": mean_nll,
        "train_perplexity": math.exp(mean_nll) if mean_nll < 20 else float("inf"),
    }
    print(f"{label}: samples={len(rows)} train_nll={mean_nll:.4f}")
    return result
