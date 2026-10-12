"""One-shot support-normalized aggregation of the captured FedIT round-1 updates.

No training or edits to alg/ are performed. The original capture and prior
evaluation are read-only. This script may be restarted with --resume; completed
condition/client evaluations are validated against the input manifest and reused.

PowerShell:
  & 'D:\\sofrware\\anaconda3\\envs\\wzm\\python.exe' -X utf8 -u tools/support_norm_oneshot.py --wait-for-fedit-30
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval_zsre import generate_and_score, generation_metadata, read_jsonl
from tools.fedit_round_conflict_diagnostic import SCALE, state_pairs
from tools.fedfh_module1_diagnostic import answer_nll, atomic_json, file_sha256, write_predictions
from utils.model_utils import load_model, load_tokenizer
from utils.seed_utils import set_global_seed
from utils.zsre_utils import encode_zsre_training_example

OLD = ROOT / 'exp/zsre8_chat_eos_v2/diagnostics/fedit_round1_update_conflict'
CAPTURE = OLD / 'capture'
PRIOR = OLD / 'evaluation'
CONFIG = ROOT / 'exp/zsre8_chat_eos_v2/fedit/adapter/fedit_zsre_Qwen3-1.7B_8c_1E_lr0.0001/training_config.json'
OUT = ROOT / 'exp/zsre8_chat_eos_v2/diagnostics/fedit_round1_support_norm'
TRAIN_STATE = ROOT / 'exp/zsre8_chat_eos_v2/fedit_resume_30r/sequence_state.json'
SEEDS = (42, 43, 44)
TAUS = (0.1, 0.2, 0.3)
ETAS = (1.5, 2.0, 3.0, 4.0)
PAIRS = ((0, 4), (3, 6))
FIELDS = ('condition', 'client', 'num_samples', 'em', 'f1', 'nll', 'eos_rate',
          'max_tokens_rate', 'average_generated_tokens')


def product_svd(left: torch.Tensor, right: torch.Tensor):
    """Double-precision thin SVD of left@right, never a dense model matrix SVD."""
    left, right = left.double(), right.double()
    ql, rl = torch.linalg.qr(left, mode='reduced')
    qr, rr = torch.linalg.qr(right.T, mode='reduced')
    uc, s, vhc = torch.linalg.svd(rl @ rr.T, full_matrices=False)
    keep = s > (s[0] * 1e-13 if s.numel() else 0)
    return ql @ uc[:, keep], s[keep], vhc[keep] @ qr.T


def update_factors(global_pair, local_pairs, ids):
    bg, ag = (tensor.double() for tensor in global_pair)
    chosen = [(local_pairs[i][0].double(), local_pairs[i][1].double()) for i in ids]
    left = torch.cat([SCALE * b for b, _ in chosen] + [-SCALE * len(ids) * bg], dim=1)
    right = torch.cat([a for _, a in chosen] + [ag], dim=0)
    return left, right


def contributions(u, vh, global_pair, local_pairs, ids):
    v = vh.T
    bg, ag = (tensor.double() for tensor in global_pair)
    global_c = ((u.T @ bg) * (ag @ v).T).sum(dim=1)
    return torch.stack([
        SCALE * (((u.T @ local_pairs[i][0].double()) *
                  (local_pairs[i][1].double() @ v).T).sum(dim=1) - global_c)
        for i in ids
    ])


def support_count_a(c: torch.Tensor) -> torch.Tensor:
    denominator = c.square().sum(dim=0)
    n = torch.where(denominator > 0, c.abs().sum(dim=0).square() /
                    denominator.clamp_min(torch.finfo(torch.float64).tiny),
                    torch.ones_like(denominator))
    return n.clamp(1.0, c.shape[0])


def support_count_b(c: torch.Tensor, tau: float) -> torch.Tensor:
    if not 0 < tau <= 1:
        raise ValueError('tau must be in (0,1]')
    maximum = c.abs().max(dim=0).values
    return ((c.abs() >= tau * maximum) & (maximum > 0)).sum(dim=0).clamp_min(1).double()


def reconstruct(u, s, vh, n):
    return (u * (s / n).unsqueeze(0)) @ vh


def low_rank_truncate(left: torch.Tensor, right: torch.Tensor, rank: int) -> torch.Tensor:
    u, s, vh = product_svd(left, right)
    count = min(rank, s.numel())
    return (u[:, :count] * s[:count]) @ vh[:count]


class Layer:
    def __init__(self, name, global_pair, local_pairs):
        self.name = name
        self.global_pair = global_pair
        self.local_pairs = local_pairs
        left, right = update_factors(global_pair, local_pairs, tuple(range(8)))
        self.u, self.s, self.vh = product_svd(left, right)
        self.c = contributions(self.u, self.vh, global_pair, local_pairs, range(8))
        # In the SVD of S, sum_i u^T Δ_i v must equal the singular value.
        if not torch.allclose(self.c.sum(0), self.s, atol=1e-9, rtol=1e-9):
            raise ValueError(f'Contribution identity failed: {name}')
        self.n_a = support_count_a(self.c)
        self.n_b = {tau: support_count_b(self.c, tau) for tau in TAUS}
        self.norm_ratio = (self.s / self.n_a).norm().item() / (self.s / 8).norm().item()
        self.pair_svd = {}

    def pair(self, ids):
        if ids not in self.pair_svd:
            left, right = update_factors(self.global_pair, self.local_pairs, ids)
            u, s, vh = product_svd(left, right)
            c = contributions(u, vh, self.global_pair, self.local_pairs, ids)
            if not torch.allclose(c.sum(0), s, atol=1e-9, rtol=1e-9):
                raise ValueError(f'Pair contribution identity failed: {self.name}, {ids}')
            self.pair_svd[ids] = (u, s, vh, support_count_a(c))
        return self.pair_svd[ids]

    def n_for(self, condition):
        if condition == 'support_A' or condition == 'norm_matched_average':
            return self.n_a
        if condition.startswith('support_B_'):
            return self.n_b[float(condition.rsplit('_', 1)[1])]
        if condition.startswith('shuffled_support_'):
            seed = int(condition.rsplit('_', 1)[1])
            generator = torch.Generator().manual_seed(seed + 1009 * sum(map(ord, self.name)))
            return self.n_a[torch.randperm(self.n_a.numel(), generator=generator)]
        raise ValueError(condition)

    def factors_for(self, condition):
        if condition in ('global', 'average', 'direct_sum'):
            multiplier = {'global': 0.0, 'average': 1 / 8, 'direct_sum': 1.0}[condition]
            return self.u, self.s * multiplier, self.vh
        if condition.startswith('scaled_average_'):
            return self.u, self.s * (float(condition.rsplit('_', 1)[1]) / 8), self.vh
        if condition == 'norm_matched_average':
            return self.u, self.s * (self.norm_ratio / 8), self.vh
        if condition.startswith(('support_A', 'support_B_', 'shuffled_support_')):
            return self.u, self.s / self.n_for(condition), self.vh
        if condition.startswith('pair_'):
            _, mode, a, b = condition.split('_')
            u, s, vh, n = self.pair((int(a), int(b)))
            if mode == 'average':
                return u, s / 2, vh
            if mode == 'sum':
                return u, s, vh
            if mode == 'supportA':
                return u, s / n, vh
        raise ValueError(condition)

    def update(self, condition, device):
        if condition == 'ties_merging':
            return ties_merge(self.global_pair, self.local_pairs, device)
        if condition.startswith('rank8_'):
            underlying = condition.removeprefix('rank8_')
            u, values, vh = self.factors_for(underlying)
            bg, ag = self.global_pair
            left = torch.cat((SCALE * bg.double(), u * values.unsqueeze(0)), dim=1)
            right = torch.cat((ag.double(), vh), dim=0)
            effective = low_rank_truncate(left, right, 8)
            return effective.to(device=device, dtype=torch.float32), True
        u, values, vh = self.factors_for(condition)
        dense = ((u.to(device=device, dtype=torch.float32) *
                  values.to(device=device, dtype=torch.float32).unsqueeze(0)) @
                 vh.to(device=device, dtype=torch.float32))
        return dense, False

    def diagnostics(self):
        energy = self.s.square()
        total = energy.sum().item()
        n = self.n_a
        return {
            'layer': self.name, 'rank': len(self.s),
            'energy_n_lt_2': energy[n < 2].sum().item() / total,
            'energy_n_2_to_6': energy[(n >= 2) & (n <= 6)].sum().item() / total,
            'energy_n_gt_6': energy[n > 6].sum().item() / total,
            'support_over_average_norm': self.norm_ratio,
        }


def ties_merge(global_pair, local_pairs, device, keep_fraction=0.2):
    bg, ag = (x.to(device=device, dtype=torch.float32) for x in global_pair)
    global_product = bg @ ag
    deltas = []
    for bi, ai in local_pairs:
        delta = SCALE * (bi.to(device=device, dtype=torch.float32) @
                         ai.to(device=device, dtype=torch.float32) - global_product)
        flat = delta.flatten()
        selected = torch.topk(flat.abs(), math.ceil(keep_fraction * flat.numel()), sorted=False).indices
        trimmed = torch.zeros_like(flat)
        trimmed[selected] = flat[selected]
        deltas.append(trimmed.view_as(delta))
    kept = torch.stack(deltas)
    elected = kept.sum(dim=0).sign()
    agree = (kept.sign() == elected) & kept.ne(0) & elected.ne(0)
    counts = agree.sum(dim=0)
    return torch.where(counts > 0, (kept * agree).sum(dim=0) / counts.clamp_min(1),
                       torch.zeros_like(global_product)), False


class Patcher:
    def __init__(self, model, global_state, local_states):
        modules = dict(model.named_modules())
        gp = state_pairs(global_state)
        lp = [state_pairs(state) for state in local_states]
        if len(gp) != 56 or any(set(p) != set(gp) for p in lp):
            raise ValueError('Expected same 56 q/v LoRA layers for all nine checkpoints')
        self.layers = {}
        self.weights = {}
        for name in sorted(gp):
            if name not in modules:
                raise ValueError(f'Model lacks LoRA layer {name}')
            layer = Layer(name, gp[name], [p[name] for p in lp])
            self.layers[name] = layer
            weight = modules[name].weight
            self.weights[name] = (weight, weight.detach().clone())

    @torch.no_grad()
    def apply(self, condition):
        for name, layer in self.layers.items():
            weight, base = self.weights[name]
            device = weight.device
            if condition == 'average':
                # Match the prior diagnostic's float32 accumulation order
                # exactly, so the baseline gate tests protocol rather than
                # a harmless change in matrix-multiplication rounding.
                merged = base.float()
                for bi, ai in layer.local_pairs:
                    merged.addmm_(bi.to(device, torch.float32), ai.to(device, torch.float32),
                                  beta=1.0, alpha=SCALE / 8)
                weight.data.copy_(merged.to(dtype=weight.dtype))
                continue
            update, includes_global = layer.update(condition, device)
            if includes_global:
                merged = base.float() + update
            else:
                bg, ag = layer.global_pair
                global_product = SCALE * (bg.to(device, torch.float32) @ ag.to(device, torch.float32))
                merged = base.float() + global_product + update
            weight.data.copy_(merged.to(dtype=weight.dtype))

    @torch.no_grad()
    def restore(self):
        for weight, base in self.weights.values():
            weight.data.copy_(base)


def preflight():
    capture_manifest = json.loads((CAPTURE / 'manifest.json').read_text(encoding='utf-8'))
    previous_manifest = json.loads((PRIOR / 'manifest.json').read_text(encoding='utf-8'))
    if capture_manifest['target_round_zero_based'] != 1 or capture_manifest['actual_lora_scale'] != SCALE:
        raise ValueError('Captured round or LoRA scaling mismatch')
    if capture_manifest['sampled_clients'] != list(range(8)):
        raise ValueError('Capture did not select all eight clients')
    if file_sha256(CONFIG) != capture_manifest['train_config_sha256']:
        raise ValueError('Original training configuration changed')
    if file_sha256(CAPTURE / 'manifest.json') != previous_manifest['capture_manifest_sha256']:
        raise ValueError('Prior evaluation used another capture manifest')
    for name, digest in capture_manifest['checkpoint_sha256'].items():
        if file_sha256(CAPTURE / name) != digest:
            raise ValueError(f'Captured checkpoint hash mismatch: {name}')
    for i in range(8):
        if file_sha256(ROOT / f'dataset/zsre/train/{i}.jsonl') != capture_manifest['train_data_sha256'][str(i)]:
            raise ValueError(f'Train data hash mismatch: client {i}')
        if file_sha256(ROOT / f'dataset/zsre/test/{i}.jsonl') != previous_manifest['validation_sha256'][str(i)]:
            raise ValueError(f'Test data hash mismatch: client {i}')
    config = json.loads(CONFIG.read_text(encoding='utf-8'))
    return config, capture_manifest, previous_manifest


def manifest_for(capture_manifest, previous_manifest, batch_size):
    return {
        'experiment': 'fedit_round1_support_normalized_one_shot',
        'source_capture_manifest_sha256': file_sha256(CAPTURE / 'manifest.json'),
        'source_checkpoint_sha256': capture_manifest['checkpoint_sha256'],
        'source_train_sha256': capture_manifest['train_data_sha256'],
        'source_test_sha256': previous_manifest['validation_sha256'],
        'previous_joint_sha256': file_sha256(PRIOR / 'joint.csv'),
        'source_round_zero_based': 1, 'lora_scaling': SCALE,
        'svd_dtype': 'float64', 'support_A': '(sum_i abs(c_ik))^2 / sum_i c_ik^2, clamped [1,8]',
        'support_B_taus': list(TAUS), 'scaled_average_etas': list(ETAS),
        'shuffled_support_seeds': list(SEEDS), 'pairs': [list(x) for x in PAIRS],
        'ties': 'per-client per-layer exact top 20% absolute elements; sign of trimmed sum; mean of same-sign nonzero elements',
        'rank8': 'ordinary truncated SVD of scale*BgAg + aggregated matrix',
        'generation': previous_manifest['protocol'],
        'nll': 'per-example mean supervised answer+EOS token NLL',
        'nll_batch_size': batch_size, 'generation_batch_size': batch_size,
        'bootstrap_seed': 20261012, 'bootstrap_replicates': 1000,
        'bootstrap_unit': 'paired test question, pooled 8x500 in client order',
    }


def condition_order(best_eta=None):
    first = ['average', 'global', 'direct_sum', 'support_A']
    first += [f'support_B_{tau:g}' for tau in TAUS]
    first += [f'scaled_average_{eta:g}' for eta in ETAS]
    first += ['norm_matched_average']
    first += [f'shuffled_support_{seed}' for seed in SEEDS]
    first += ['ties_merging']
    first += [f'pair_{mode}_{i}_{j}' for i, j in PAIRS
              for mode in ('average', 'sum', 'supportA')]
    if best_eta is not None:
        first += ['rank8_average', 'rank8_support_A', f'rank8_scaled_average_{best_eta:g}']
    return first


def reference_average():
    with (PRIOR / 'joint.csv').open('r', encoding='utf-8-sig', newline='') as handle:
        rows = [row for row in csv.DictReader(handle) if row['condition'] == 'matrix_average']
    if len(rows) != 8:
        raise ValueError('Prior joint.csv lacks eight matrix_average rows')
    return (sum(float(row['nll']) for row in rows) / 8,
            sum(float(row['em']) for row in rows) / 8)


def check_average_baseline(new_nll, new_em, old_nll, old_em):
    if abs(new_nll - old_nll) >= 0.005 or abs(new_em - old_em) >= 0.005:
        raise RuntimeError(f'Average baseline mismatch: new NLL={new_nll:.4f}, EM={new_em:.2f}; '
                           f'prior NLL={old_nll:.4f}, EM={old_em:.2f}; stopping before other conditions')


def load_cached(condition, client):
    metric_path = OUT / 'metrics' / f'{condition}_client_{client}.json'
    prediction_path = OUT / 'predictions' / f'{condition}_client_{client}.jsonl'
    if not metric_path.exists() or not prediction_path.exists():
        # A crash between writing predictions and metrics leaves one partial
        # file. Re-evaluate that client; never trust a half-written result.
        return None
    metric = json.loads(metric_path.read_text(encoding='utf-8'))
    predictions = read_jsonl(prediction_path)
    if metric['condition'] != condition or metric['client'] != client or len(predictions) != 500:
        raise ValueError(f'Invalid cached evaluation for {condition}, client {client}')
    return metric, predictions


def write_results():
    rows = []
    for path in sorted((OUT / 'metrics').glob('*_client_*.json')):
        row = json.loads(path.read_text(encoding='utf-8'))
        rows.append(row)
    temporary = OUT / 'results.csv.tmp'
    with temporary.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(OUT / 'results.csv')


def bootstrap_ci(reference, candidate, indices):
    difference = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    interval = 100 * difference[indices].mean(axis=1)
    return tuple(float(x) for x in np.percentile(interval, [2.5, 97.5]))


def summarize(all_metrics, all_predictions, layers, best_eta):
    order = condition_order(best_eta)
    rng = np.random.default_rng(20261012)
    indices = rng.integers(0, 4000, size=(1000, 4000), dtype=np.int32)
    correct = {name: [int(row['em']) for i in range(8) for row in all_predictions[name][i]]
               for name in order}
    base = correct['average']
    macros = {}
    for name in order:
        rows = [all_metrics[name][i] for i in range(8)]
        macros[name] = {field: sum(row[field] for row in rows) / 8
                        for field in FIELDS[3:]}
        macros[name]['delta_em_vs_average'] = macros[name]['em'] - macros['average']['em'] if 'average' in macros else 0.0
        macros[name]['ci_vs_average'] = bootstrap_ci(base, correct[name], indices)
    chosen_scaled = f'scaled_average_{best_eta:g}'
    rivals = [chosen_scaled, 'norm_matched_average'] + [f'shuffled_support_{seed}' for seed in SEEDS]
    controls = {name: {rival: {'delta_em': macros[name]['em'] - macros[rival]['em'],
                               'ci': bootstrap_ci(correct[rival], correct[name], indices)}
                       for rival in rivals}
                for name in ['support_A'] + [f'support_B_{tau:g}' for tau in TAUS]}
    diag = [layer.diagnostics() for layer in layers.values()]
    with (OUT / 'layer_diagnostics.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(diag[0]))
        writer.writeheader()
        writer.writerows(diag)
    def fmt(name):
        row = macros[name]
        lo, hi = row['ci_vs_average']
        return (f"| {name} | {row['em']:.2f} | {row['f1']:.2f} | {row['nll']:.3f} | "
                f"{row['eos_rate']:.1f} | {row['max_tokens_rate']:.1f} | "
                f"{row['average_generated_tokens']:.2f} | {row['delta_em_vs_average']:+.2f} "
                f"[{lo:+.2f}, {hi:+.2f}] |")
    lines = ['# FedIT round-1 support normalization: one-shot evaluation', '',
             'All metrics are macro means across 8 clients × 500 held-out rephrased questions. EM/F1/EOS/max-token rates and EM differences are percentage points.', '',
             '| Condition | EM | F1 | NLL | EOS % | max-token % | mean tokens | ΔEM vs average [paired 95% CI] |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    lines += [fmt(name) for name in order]
    lines += ['', f'Best scaled_average η by macro EM: {best_eta:g} (selected on these same test examples; comparison is exploratory).', '',
              '## Per-layer support-A geometry', '',
              '| Layer | rank | σ² energy N<2 | σ² energy 2≤N≤6 | σ² energy N>6 | ‖support_A‖/‖average‖ |',
              '|---|---:|---:|---:|---:|---:|']
    for row in diag:
        lines.append(f"| {row['layer']} | {row['rank']} | {row['energy_n_lt_2']:.3f} | {row['energy_n_2_to_6']:.3f} | {row['energy_n_gt_6']:.3f} | {row['support_over_average_norm']:.2f} |")
    lines += ['', '## Paired combinations (all eight clients evaluated)', '',
              '| Pair | mode | EM | F1 | NLL | EOS % |', '|---|---|---:|---:|---:|---:|']
    for i, j in PAIRS:
        for mode in ('average', 'sum', 'supportA'):
            name = f'pair_{mode}_{i}_{j}'
            row = macros[name]
            lines.append(f"| {i}+{j} | {mode} | {row['em']:.2f} | {row['f1']:.2f} | {row['nll']:.3f} | {row['eos_rate']:.1f} |")
    lines += ['', '## Support variants against amplification and shuffled directions', '',
              '| Support | comparator | ΔEM | paired 95% CI |', '|---|---|---:|---:|']
    for name in controls:
        for rival, item in controls[name].items():
            lo, hi = item['ci']
            lines.append(f"| {name} | {rival} | {item['delta_em']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    lines += ['', 'Bootstrap resamples individual test questions jointly across conditions (1000 resamples of 4000). Conditions and thresholds were fixed before evaluation. These are diagnostic comparisons on the same test set, not a separate tuning/confirmation split.', '']
    (OUT / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    atomic_json(OUT / 'summary.json', {'macro': macros, 'control_comparisons': controls,
                                       'best_scaled_eta': best_eta, 'layers': diag})


def wait_for_training():
    while True:
        if not TRAIN_STATE.exists():
            raise FileNotFoundError(f'FedIT continuation state missing: {TRAIN_STATE}')
        state = json.loads(TRAIN_STATE.read_text(encoding='utf-8'))
        if state['status'] == 'completed' and state.get('completed_round') == 30:
            return
        if state['status'] == 'failed':
            raise RuntimeError(f"FedIT continuation failed: {state.get('error')}")
        print(f"Waiting for FedIT continuation: round {state.get('current_round')}/30", flush=True)
        time.sleep(60)


def run(cli):
    config, captured, prior_manifest = preflight()
    manifest = manifest_for(captured, prior_manifest, cli.batch_size)
    manifest_path = OUT / 'manifest.json'
    if OUT.exists() and any(OUT.iterdir()):
        if not cli.resume or not manifest_path.is_file() or json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
            raise FileExistsError(f'Existing results must have a matching manifest and --resume: {OUT}')
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        atomic_json(manifest_path, manifest)
    (OUT / 'metrics').mkdir(exist_ok=True)
    (OUT / 'predictions').mkdir(exist_ok=True)
    if cli.wait_for_fedit_30:
        wait_for_training()

    args = SimpleNamespace(**config)
    set_global_seed(args.seed, device=args.device, deterministic=bool(args.deterministic))
    tokenizer = load_tokenizer(args)
    if generation_metadata(tokenizer) != prior_manifest['protocol']:
        raise ValueError('Current tokenizer generation settings differ from prior evaluation')
    model = load_model(args)  # plain model: no active PEFT adapter can be double-counted
    global_state = torch.load(CAPTURE / 'global_before.pt', map_location='cpu', weights_only=True)
    local_states = [torch.load(CAPTURE / f'client_{i}.pt', map_location='cpu', weights_only=True)
                    for i in range(8)]
    patcher = Patcher(model, global_state, local_states)
    validation = [read_jsonl(ROOT / f'dataset/zsre/test/{i}.jsonl') for i in range(8)]
    if any(len(rows) != 500 for rows in validation):
        raise ValueError('Expected 8 x 500 validation questions')
    encoded = [[encode_zsre_training_example(tokenizer, row) for row in rows]
               for rows in validation]

    all_metrics, all_predictions = {}, {}
    def measure(condition):
        all_metrics[condition], all_predictions[condition] = {}, {}
        pending = [i for i in range(8) if load_cached(condition, i) is None]
        if pending:
            patcher.apply(condition)
        for i in range(8):
            cached = load_cached(condition, i)
            if cached is None:
                nll = answer_nll(model, encoded[i], cli.batch_size)
                scores, predictions = generate_and_score(model, tokenizer, validation[i], cli.batch_size)
                row = {'condition': condition, 'client': i, 'num_samples': 500, 'nll': nll, **scores}
                write_predictions(OUT / 'predictions' / f'{condition}_client_{i}.jsonl', predictions)
                atomic_json(OUT / 'metrics' / f'{condition}_client_{i}.json', row)
                write_results()
            else:
                row, predictions = cached
            all_metrics[condition][i], all_predictions[condition][i] = row, predictions
            print(f"{condition}: client {i} EM={row['em']:.2f} NLL={row['nll']:.4f}", flush=True)

    try:
        # Hard gate: do not run any proposed method until the old matrix-average
        # EM and NLL are reproduced under the identical evaluation protocol.
        measure('average')
        old_nll, old_em = reference_average()
        new_nll = sum(all_metrics['average'][i]['nll'] for i in range(8)) / 8
        new_em = sum(all_metrics['average'][i]['em'] for i in range(8)) / 8
        check_average_baseline(new_nll, new_em, old_nll, old_em)
        print(f'Average baseline reproduced: NLL={new_nll:.4f}, EM={new_em:.2f}', flush=True)
        for condition in condition_order():
            if condition != 'average':
                measure(condition)
        best_eta = max(ETAS, key=lambda eta: (sum(all_metrics[f'scaled_average_{eta:g}'][i]['em']
                                             for i in range(8)) / 8, -eta))
        for condition in condition_order(best_eta)[-3:]:
            measure(condition)
        summarize(all_metrics, all_predictions, patcher.layers, best_eta)
        print(f'Complete: {OUT / "summary.md"}', flush=True)
    finally:
        patcher.restore()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--wait-for-fedit-30', action='store_true',
                        help='Wait for the concurrent 30-round FedIT training to finish before allocating GPU')
    cli = parser.parse_args()
    if cli.batch_size < 1:
        parser.error('--batch-size must be positive')
    run(cli)


if __name__ == '__main__':
    main()
