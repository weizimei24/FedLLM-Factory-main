"""Continue the original 10-round zsRE FedIT run through round 30.

The original adapter is read-only. Completed rounds 15, 20, 25, and 30 are
saved separately under exp/zsre8_chat_eos_v2/fedit_resume_30r/. Restarting
this script resumes from the latest complete five-round snapshot.

PowerShell:
  & 'D:\\sofrware\\anaconda3\\envs\\wzm\\python.exe' -X utf8 -u tools/resume_zsre_fedit.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SOURCE = ROOT / 'exp/zsre8_chat_eos_v2/fedit/adapter/fedit_zsre_Qwen3-1.7B_8c_1E_lr0.0001'
OUTPUT = ROOT / 'exp/zsre8_chat_eos_v2/fedit_resume_30r'
ORIGINAL_ROUNDS = 10
TARGET_ROUNDS = 30
SAVE_EVERY = 5
SNAPSHOT_ROUNDS = tuple(range(ORIGINAL_ROUNDS + SAVE_EVERY, TARGET_ROUNDS + 1, SAVE_EVERY))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def snapshot_dir(completed_round: int) -> Path:
    return OUTPUT / f'round_{completed_round:03d}'


def select_resume(source_hash: str) -> tuple[int, Path]:
    """Trust only complete snapshots made from this exact original adapter."""
    for completed in reversed(SNAPSHOT_ROUNDS):
        folder = snapshot_dir(completed)
        checkpoint = folder / 'lora_weights.pt'
        manifest = folder / 'manifest.json'
        if not checkpoint.exists() or not manifest.exists():
            continue
        info = json.loads(manifest.read_text(encoding='utf-8'))
        if (info.get('completed_round') != completed or
                info.get('source_sha256') != source_hash or
                info.get('checkpoint_sha256') != sha256(checkpoint)):
            raise RuntimeError(f'Invalid or mismatched snapshot: {folder}')
        return completed, checkpoint
    return ORIGINAL_ROUNDS, SOURCE / 'lora_weights.pt'


def verify_config(config: dict) -> None:
    expected = {'alg': 'fedit', 'dataset': 'zsre', 'model': 'Qwen3-1.7B',
                'cn': 8, 'sr': 1.0, 'rnd': ORIGINAL_ROUNDS,
                'lora_rank': 8, 'lora_alpha': 32, 'lora_dropout': 0.05,
                'seed': 42, 'deterministic': True, 'test_gap': 1,
                'global_test': False, 'cross_domain_qa': False}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'Original config {key}={config.get(key)!r}, expected {value!r}')


def load_adapter(path: Path, model) -> dict:
    saved = torch.load(path, map_location='cpu', weights_only=True)
    current = {k: v for k, v in model.state_dict().items() if 'lora_' in k}
    if set(saved) != set(current):
        raise ValueError('Saved LoRA tensor keys do not match the current model')
    for key, tensor in saved.items():
        if not isinstance(tensor, torch.Tensor) or tensor.shape != current[key].shape:
            raise ValueError(f'Invalid LoRA tensor: {key}')
        if not torch.isfinite(tensor).all():
            raise ValueError(f'Non-finite LoRA tensor: {key}')
    result = {key: tensor.to(device=current[key].device, dtype=current[key].dtype)
              for key, tensor in saved.items()}
    model.load_state_dict(result, strict=False)
    return result


def save_snapshot(server, completed_round: int, source_hash: str) -> None:
    folder = snapshot_dir(completed_round)
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / 'lora_weights.pt'
    temporary = folder / 'lora_weights.pt.tmp'
    weights = {k: v.detach().cpu().clone() for k, v in server.global_lora.items()}
    torch.save(weights, temporary)
    os.replace(temporary, checkpoint)
    atomic_json(folder / 'manifest.json', {
        'completed_round': completed_round,
        'source_checkpoint': str(SOURCE / 'lora_weights.pt'),
        'source_sha256': source_hash,
        'checkpoint_sha256': sha256(checkpoint),
        'round_indexing': 'round_015 means 15 completed rounds, last zero-based index 14',
    })
    print(f'Saved completed round {completed_round}: {checkpoint}', flush=True)


def run() -> None:
    source_checkpoint = SOURCE / 'lora_weights.pt'
    source_config = SOURCE / 'training_config.json'
    config = json.loads(source_config.read_text(encoding='utf-8'))
    verify_config(config)
    source_hash = sha256(source_checkpoint)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # Exclusive creation prevents two continuation processes writing the same snapshots.
    lock_path = OUTPUT / 'run.lock'
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f'Continuation lock already exists: {lock_path}') from exc
    try:
        with os.fdopen(lock_fd, 'w', encoding='utf-8') as handle:
            handle.write(str(os.getpid()))
        completed, checkpoint = select_resume(source_hash)
        state_path = OUTPUT / 'sequence_state.json'
        metrics_path = OUTPUT / 'round_metrics.json'
        metrics = json.loads(metrics_path.read_text(encoding='utf-8')) if metrics_path.exists() else {}
        metrics = {key: value for key, value in metrics.items() if int(key) <= completed}
        atomic_json(metrics_path, metrics)
        if completed == TARGET_ROUNDS:
            atomic_json(state_path, {'status': 'completed', 'completed_round': completed,
                                     'source_sha256': source_hash})
            print('All 30 rounds are already complete.', flush=True)
            return

        # Keep the training and evaluation configuration unchanged. Only the
        # total round count and output directory differ from the original run.
        args = SimpleNamespace(**config)
        args.rnd = TARGET_ROUNDS
        args.suffix = str(OUTPUT)
        from alg.fedit import Client, Server
        from utils.seed_utils import set_global_seed
        set_global_seed(args.seed, device=args.device, deterministic=args.deterministic)
        clients = [Client(index, args) for index in range(args.cn)]
        set_global_seed(args.seed, device=args.device, deterministic=args.deterministic)
        server = Server(args, clients)
        server.global_lora = load_adapter(checkpoint, server.model)
        atomic_json(OUTPUT / 'resume_config.json', {
            'original_config': config, 'total_rounds': TARGET_ROUNDS,
            'snapshot_every_completed_rounds': SAVE_EVERY,
            'source_sha256': source_hash,
        })
        print(f'Resuming from {completed} completed rounds; next zero-based round {completed}', flush=True)

        for round_index in range(completed, TARGET_ROUNDS):
            start = time.time()
            atomic_json(state_path, {'status': 'running', 'completed_snapshot_round': completed,
                                     'current_round': round_index + 1, 'pid': os.getpid(),
                                     'source_sha256': source_hash})
            print(f'=== FedIT round {round_index + 1}/{TARGET_ROUNDS} ===', flush=True)
            server.round = round_index
            server.run()
            evaluation = server.test_all()  # Same per-round loss/ppl test as original.
            metrics[str(round_index + 1)] = {
                'round': round_index + 1,
                'learning_rate': args.lr * (0.99 ** round_index),
                'test': {key: float(value) for key, value in evaluation.items()},
                'client_train_loss': {str(c.id): float(c.last_train_loss) for c in clients},
                'elapsed_seconds': time.time() - start,
            }
            atomic_json(metrics_path, metrics)
            print(f'Round {round_index + 1} test: {evaluation}', flush=True)
            if (round_index + 1) % SAVE_EVERY == 0:
                save_snapshot(server, round_index + 1, source_hash)
                completed = round_index + 1

        atomic_json(state_path, {'status': 'completed', 'completed_round': TARGET_ROUNDS,
                                 'pid': os.getpid(), 'source_sha256': source_hash})
    except Exception as exc:
        atomic_json(OUTPUT / 'sequence_state.json', {
            'status': 'failed', 'error': repr(exc), 'traceback': traceback.format_exc(),
            'pid': os.getpid(), 'source_sha256': source_hash,
        })
        raise
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == '__main__':
    run()
