"""Evaluate saved post-aggregation FedIT rounds on the same 8x500 zsRE test set.

This reads the original round-10 global adapter and the continued round-15,
20, 25, 30 global adapters. It does no training and writes only to a new
round_test_eval directory under fedit_resume_30r.

PowerShell:
  & 'D:\\sofrware\\anaconda3\\envs\\wzm\\python.exe' -X utf8 -u tools/eval_zsre_fedit_rounds.py
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch
from peft import get_peft_model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval_zsre import generate_and_score, generation_metadata, read_jsonl
from tools.fedfh_module1_diagnostic import (
    answer_nll, atomic_json, file_sha256, verify_checkpoint, write_predictions,
)
from utils.model_utils import load_lora_config, load_model, load_tokenizer
from utils.seed_utils import set_global_seed
from utils.zsre_utils import encode_zsre_training_example

ORIGINAL = ROOT / 'exp/zsre8_chat_eos_v2/fedit/adapter/fedit_zsre_Qwen3-1.7B_8c_1E_lr0.0001'
CONTINUED = ROOT / 'exp/zsre8_chat_eos_v2/fedit_resume_30r'
OUTPUT = CONTINUED / 'round_test_eval'
PRIOR = ROOT / 'exp/zsre8_chat_eos_v2/fedit/evaluation/zsre/metrics.json'
ROUNDS = (10, 15, 20, 25, 30)
SCORE_KEYS = ('em', 'f1', 'answer_nll', 'eos_rate', 'max_tokens_rate', 'average_generated_tokens')


def checkpoints() -> dict[int, Path]:
    paths = {10: ORIGINAL / 'lora_weights.pt'}
    paths.update({round_number: CONTINUED / f'round_{round_number:03d}' / 'lora_weights.pt'
                  for round_number in ROUNDS[1:]})
    return paths


def preflight():
    state = json.loads((CONTINUED / 'sequence_state.json').read_text(encoding='utf-8'))
    if state.get('status') != 'completed' or state.get('completed_round') != 30:
        raise RuntimeError('FedIT continuation has not completed all 30 rounds')
    config = json.loads((ORIGINAL / 'training_config.json').read_text(encoding='utf-8'))
    if (config.get('alg'), config.get('dataset'), config.get('model'),
        config.get('cn'), config.get('lora_rank'), config.get('lora_alpha')) != (
            'fedit', 'zsre', 'Qwen3-1.7B', 8, 8, 32):
        raise ValueError('Unexpected FedIT training configuration')
    paths = checkpoints()
    source_hash = file_sha256(paths[10])
    if source_hash != state.get('source_sha256'):
        raise ValueError('Original round-10 adapter differs from continuation source')
    for round_number in ROUNDS[1:]:
        path = paths[round_number]
        saved = json.loads((path.parent / 'manifest.json').read_text(encoding='utf-8'))
        if (saved.get('completed_round') != round_number or
                saved.get('source_sha256') != source_hash or
                saved.get('checkpoint_sha256') != file_sha256(path)):
            raise ValueError(f'Invalid round-{round_number} checkpoint manifest')
    prior = json.loads(PRIOR.read_text(encoding='utf-8'))
    if prior.get('method') != 'fedit' or set(prior.get('per_client', {})) != {str(i) for i in range(8)}:
        raise ValueError('Original FedIT generation evaluation is missing or incompatible')
    return config, paths, prior


def output_manifest(paths, prior, tokenizer, batch_size):
    protocol = generation_metadata(tokenizer)
    if protocol != prior['generation']:
        raise ValueError('Generation protocol differs from original round-10 evaluation')
    return {
        'experiment': 'post_aggregation_fedit_round_test_evaluation',
        'rounds_completed': list(ROUNDS),
        'checkpoint_sha256': {str(k): file_sha256(v) for k, v in paths.items()},
        'config_sha256': file_sha256(ORIGINAL / 'training_config.json'),
        'test_sha256': {str(i): file_sha256(ROOT / f'dataset/zsre/test/{i}.jsonl')
                        for i in range(8)},
        'generation': protocol,
        'answer_nll': 'mean of per-example mean gold-answer-plus-EOS token NLL',
        'batch_size': batch_size,
        'round_10_reference': PRIOR.relative_to(ROOT).as_posix(),
    }


def load_round_adapter(model, path):
    state = torch.load(path, map_location='cpu', weights_only=True)
    verify_checkpoint(model, state, rank=8)
    current = model.state_dict()
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError(f'Non-finite adapter weights: {path}')
    model.load_state_dict({key: value.to(dtype=current[key].dtype) for key, value in state.items()},
                          strict=False)


def check_round_10(metrics, prior):
    """A saved adapter must reproduce the previous generation evaluation."""
    for i in range(8):
        for key in ('em', 'f1', 'eos_rate', 'max_tokens_rate', 'average_generated_tokens'):
            if abs(metrics[i][key] - prior['per_client'][str(i)][key]) > 1e-5:
                raise RuntimeError(f'Round 10 does not reproduce prior evaluation: client {i}, {key}; stop')


def summarize(all_rounds):
    macros = {}
    for round_number in ROUNDS:
        rows = [all_rounds[round_number][i] for i in range(8)]
        if any(row['num_samples'] != 500 for row in rows):
            raise ValueError(f'Round {round_number}: expected 8x500 test examples')
        macros[round_number] = {key: sum(row[key] for row in rows) / 8 for key in SCORE_KEYS}
        macros[round_number]['num_samples'] = 4000
    lines = [
        '# FedIT post-aggregation global adapters on zsRE test', '',
        'Each saved global adapter was evaluated on all eight clients’ 500 rephrased test questions under the original zsre_chat_eos_v2 greedy 32-token protocol. Round 10 is the pre-continuation reference.', '',
        '| Completed round | EM | F1 | Answer NLL | EOS % | Max-token % | Mean generated tokens |',
        '|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for round_number in ROUNDS:
        row = macros[round_number]
        lines.append(f"| {round_number} | {row['em']:.2f} | {row['f1']:.2f} | "
                     f"{row['answer_nll']:.4f} | {row['eos_rate']:.2f} | "
                     f"{row['max_tokens_rate']:.2f} | {row['average_generated_tokens']:.2f} |")
    lines += ['', 'Per-client values and checkpoint/data hashes are in metrics.json and manifest.json.']
    atomic_json(OUTPUT / 'metrics.json', {'per_round': {str(k): {'per_client': {str(i): all_rounds[k][i]
                                   for i in range(8)}, 'macro_average': macros[k]} for k in ROUNDS},
                                   'protocol': 'zsre_chat_eos_v2'})
    (OUTPUT / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return macros


def run(batch_size, resume):
    config, paths, prior = preflight()
    args = SimpleNamespace(**config)
    set_global_seed(args.seed, device=args.device, deterministic=bool(args.deterministic))
    tokenizer = load_tokenizer(args)
    manifest = output_manifest(paths, prior, tokenizer, batch_size)
    manifest_path = OUTPUT / 'manifest.json'
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        if not resume or not manifest_path.exists() or json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
            raise FileExistsError(f'Existing output differs or --resume not supplied: {OUTPUT}')
    else:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        atomic_json(manifest_path, manifest)
    (OUTPUT / 'predictions').mkdir(exist_ok=True)
    (OUTPUT / 'client_metrics').mkdir(exist_ok=True)
    test_rows = [read_jsonl(ROOT / f'dataset/zsre/test/{i}.jsonl') for i in range(8)]
    if any(len(rows) != 500 for rows in test_rows):
        raise ValueError('Expected 8 test files of 500 examples each')
    encoded_rows = [[encode_zsre_training_example(tokenizer, row) for row in rows]
                    for rows in test_rows]
    model = get_peft_model(load_model(args), load_lora_config(args))
    all_rounds = {}
    for round_number in ROUNDS:
        load_round_adapter(model, paths[round_number])
        all_rounds[round_number] = {}
        for i in range(8):
            metric_path = OUTPUT / 'client_metrics' / f'round_{round_number:03d}_client_{i}.json'
            prediction_path = OUTPUT / 'predictions' / f'round_{round_number:03d}_client_{i}.jsonl'
            if metric_path.exists() and prediction_path.exists():
                row = json.loads(metric_path.read_text(encoding='utf-8'))
                if row.get('round') != round_number or row.get('client') != i or row.get('num_samples') != 500:
                    raise ValueError(f'Invalid cached round {round_number}, client {i}')
                if len(read_jsonl(prediction_path)) != 500:
                    raise ValueError(f'Incomplete cached predictions: round {round_number}, client {i}')
            else:
                atomic_json(OUTPUT / 'evaluation_state.json', {
                    'status': 'running', 'round': round_number, 'client': i,
                })
                nll = answer_nll(model, encoded_rows[i], batch_size)
                scores, predictions = generate_and_score(model, tokenizer, test_rows[i], batch_size)
                row = {'round': round_number, 'client': i, 'answer_nll': nll, **scores}
                write_predictions(prediction_path, predictions)
                atomic_json(metric_path, row)
            all_rounds[round_number][i] = row
            print(f"round {round_number} client {i}: EM={row['em']:.2f} F1={row['f1']:.2f} "
                  f"NLL={row['answer_nll']:.4f}", flush=True)
        if round_number == 10:
            check_round_10(all_rounds[10], prior)
            print('Round-10 generation baseline reproduced; evaluating resumed checkpoints.', flush=True)
    macros = summarize(all_rounds)
    atomic_json(OUTPUT / 'evaluation_state.json', {'status': 'completed', 'rounds': list(ROUNDS)})
    print(f'Complete: {OUTPUT / "summary.md"}', flush=True)
    return macros


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--resume', action='store_true')
    cli = parser.parse_args()
    if cli.batch_size < 1:
        parser.error('batch size must be positive')
    try:
        run(cli.batch_size, cli.resume)
    except Exception as exc:
        if OUTPUT.exists() and (OUTPUT / 'manifest.json').exists():
            atomic_json(OUTPUT / 'evaluation_state.json', {
                'status': 'failed', 'error': repr(exc), 'traceback': traceback.format_exc(),
            })
        raise


if __name__ == '__main__':
    main()
