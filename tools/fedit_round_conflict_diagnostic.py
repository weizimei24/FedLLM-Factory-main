"""Replay one authentic zsRE FedIT round, then diagnose product-space updates.

The original zsRE run kept only its final global adapter, so historical client
states cannot be recovered. ``capture`` deterministically replays the original
FedIT training path through --round (default 1), saves the *pre-aggregation*
global and all eight clients from that same round, and never writes to exp/fedit.
``evaluate`` uses the held-out rephrased test questions, not training prompts.

PowerShell:
  & 'D:\\sofrware\\anaconda3\\envs\\wzm\\python.exe' -X utf8 -u tools/fedit_round_conflict_diagnostic.py capture
  & 'D:\\sofrware\\anaconda3\\envs\\wzm\\python.exe' -X utf8 -u tools/fedit_round_conflict_diagnostic.py evaluate
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_zsre import generate_and_score, generation_metadata, read_jsonl
from tools.fedfh_module1_diagnostic import (
    answer_nll, atomic_json, file_sha256, verify_checkpoint, write_predictions,
)
from utils.model_utils import load_model, load_tokenizer
from utils.seed_utils import set_global_seed
from utils.zsre_utils import encode_zsre_training_example


DEFAULT_OUT = Path('exp/zsre8_chat_eos_v2/diagnostics/fedit_round1_update_conflict')
DEFAULT_ORIGINAL = Path('exp/zsre8_chat_eos_v2/fedit/adapter/fedit_zsre_Qwen3-1.7B_8c_1E_lr0.0001')
SCALE = 4.0
ALPHAS = (0.0, 0.125, 0.25, 0.5, 1.0)


def lora_state(model) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            if '.lora_A.' in k or '.lora_B.' in k}


def module_name(key: str) -> str:
    name = key.split('.lora_A.')[0]
    if name.startswith('base_model.model.'):
        return name[len('base_model.model.'):]
    raise ValueError(f'Unexpected LoRA key: {key}')


def state_pairs(state: dict) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    pairs = {}
    for key in sorted(state):
        if '.lora_A.' in key:
            b_key = key.replace('.lora_A.', '.lora_B.')
            if b_key not in state:
                raise KeyError(b_key)
            pairs[module_name(key)] = (state[b_key].float(), state[key].float())
    if not pairs or any(not name.endswith(('q_proj', 'v_proj')) for name in pairs):
        raise ValueError('Expected q/v LoRA layers')
    return pairs


def product_change(global_pair, local_pair, scale=SCALE):
    """Actual dense update s(B_i A_i - B_g A_g), not a factor difference."""
    bg, ag = global_pair
    bi, ai = local_pair
    return scale * (bi.float() @ ai.float() - bg.float() @ ag.float())


def product_rank(left: torch.Tensor, right: torch.Tensor) -> int:
    """Rank of left@right via small QR core, without a huge dense SVD."""
    ql, rl = torch.linalg.qr(left.double(), mode='reduced')
    qr, rr = torch.linalg.qr(right.double().T, mode='reduced')
    singular = torch.linalg.svdvals(rl @ rr.T)
    tol = max(left.shape[0], right.shape[1]) * torch.finfo(torch.float64).eps * singular.max()
    return int((singular > tol).sum())


def condition_rank(global_pair, local_pairs, coefficients: dict[int, float]) -> int:
    active = [(i, c) for i, c in sorted(coefficients.items()) if c]
    if not active:
        return 0
    bg, ag = global_pair
    left = torch.cat([c * local_pairs[i][0].float() for i, c in active] +
                     [-sum(c for _, c in active) * bg.float()], dim=1)
    right = torch.cat([local_pairs[i][1].float() for i, _ in active] + [ag.float()], dim=0)
    return product_rank(left, right)


class RoundPatcher:
    """Temporary dense merge: W = W0 + s[BgAg + sum_i c_i(BiAi-BgAg)]."""

    def __init__(self, plain_model, global_state: dict, local_states: list[dict]):
        self.modules = dict(plain_model.named_modules())
        self.global_pairs = state_pairs(global_state)
        self.local_pairs = [state_pairs(state) for state in local_states]
        keys = set(self.global_pairs)
        if any(set(pairs) != keys for pairs in self.local_pairs):
            raise ValueError('All clients must have exactly the global q/v layer set')
        self.base = {}
        self.factors = {}
        for name in sorted(keys):
            if name not in self.modules:
                raise KeyError(f'Model has no module {name}')
            weight = self.modules[name].weight
            self.base[name] = weight.detach().clone()
            gp = self.global_pairs[name]
            lp = [pairs[name] for pairs in self.local_pairs]
            self.factors[name] = (
                tuple(t.to(weight.device, torch.float32) for t in gp),
                [tuple(t.to(weight.device, torch.float32) for t in pair) for pair in lp],
            )

    @torch.no_grad()
    def apply(self, coefficients: dict[int, float]) -> None:
        if any(i < 0 or i >= len(self.local_pairs) or not math.isfinite(c)
               for i, c in coefficients.items()):
            raise ValueError('Invalid client coefficient')
        active = [(i, float(c)) for i, c in sorted(coefficients.items()) if c]
        total = sum(c for _, c in active)
        for name, (gp, lp) in self.factors.items():
            bg, ag = gp
            updated = self.base[name].float() + SCALE * (1.0 - total) * (bg @ ag)
            for i, c in active:
                bi, ai = lp[i]
                updated.addmm_(bi, ai, beta=1.0, alpha=SCALE * c)
            self.modules[name].weight.data.copy_(updated.to(self.modules[name].weight.dtype))

    @torch.no_grad()
    def restore_base(self):
        for name, weight in self.base.items():
            self.modules[name].weight.data.copy_(weight)

    def ranks(self, coefficients: dict[int, float]) -> dict:
        ranks = [condition_rank(self.global_pairs[name],
                                [p[name] for p in self.local_pairs], coefficients)
                 for name in self.global_pairs]
        return {'min_rank': min(ranks), 'max_rank': max(ranks),
                'mean_rank': sum(ranks) / len(ranks), 'layer_ranks': ranks,
                'algebraic_rank_bound': 8 * (1 + sum(bool(c) for c in coefficients.values()))}


def capture(cli) -> None:
    from alg.fedit import Client, Server

    original_config = DEFAULT_ORIGINAL / 'training_config.json'
    if not original_config.is_file():
        raise FileNotFoundError(original_config)
    saved = json.loads(original_config.read_text(encoding='utf-8'))
    expected = {'alg': 'fedit', 'dataset': 'zsre', 'model': 'Qwen3-1.7B',
                'cn': 8, 'sr': 1.0, 'lora_rank': 8, 'lora_alpha': 32,
                'epoch': 1, 'step': 63, 'grad_accum': 8}
    if any(saved.get(key) != value for key, value in expected.items()):
        raise ValueError('Saved FedIT configuration does not match zsRE 8-client run')
    output = cli.out / 'capture'
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Capture output already exists: {output}')
    output.mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(**saved)
    args.suffix = str(cli.out)
    set_global_seed(args.seed, device=args.device, deterministic=args.deterministic)
    clients = [Client(i, args) for i in range(8)]
    # main.FedSim reseeds after client data loading, before server/model creation.
    set_global_seed(args.seed, device=args.device, deterministic=args.deterministic)
    server = Server(args, clients)
    for rnd in range(cli.round):
        server.round = rnd
        print(f'Replaying FedIT round {rnd} to reconstruct global start', flush=True)
        server.run()
    server.round = cli.round
    server.sample()
    if [c.id for c in server.sampled_clients] != list(range(8)):
        raise ValueError('Expected all eight clients selected in the target round')
    before = lora_state(server.model)
    torch.save(before, output / 'global_before.pt')
    print(f'Capturing target round {cli.round}: same global sent to all clients', flush=True)
    server.local_run()
    for client in server.sampled_clients:
        verify_checkpoint(server.model, client.lora, 8)
        torch.save({k: v.cpu() for k, v in client.lora.items()}, output / f'client_{client.id}.pt')
    server.aggregate()
    torch.save(lora_state(server.model), output / 'global_after_fedit.pt')
    manifest = {
        'kind': 'deterministic_replay_of_original_fedit_training',
        'target_round_zero_based': cli.round, 'sampled_clients': list(range(8)),
        'all_clients_share': 'global_before.pt', 'local_epochs': args.epoch,
        'local_optimizer_steps': args.step, 'lr_at_round': args.lr * .99 ** cli.round,
        'lora_rank': 8, 'lora_alpha': 32, 'actual_lora_scale': SCALE,
        'train_config_sha256': file_sha256(original_config),
        'train_data_sha256': {str(i): file_sha256(Path('dataset/zsre/train') / f'{i}.jsonl') for i in range(8)},
        'checkpoint_sha256': {p.name: file_sha256(p) for p in sorted(output.glob('*.pt'))},
        'note': 'Not recovered from the old run: it saved only the final adapter. No test examples were trained on.',
    }
    atomic_json(output / 'manifest.json', manifest)
    print(f'Saved round {cli.round} pre-global, 8 local states, post-global to {output}', flush=True)


def choose_pairs(s: list[list[float]]) -> dict[str, list[int]]:
    # Exploratory selection on held-out NLL: max and min *mutual* cross effect.
    pairs = [(s[i][j] + s[j][i], i, j) for i in range(8) for j in range(i + 1, 8)]
    adverse = min(pairs)
    helpful = max(pair for pair in pairs if pair[1:] != adverse[1:])
    return {'most_adverse_cross_effect': [adverse[1], adverse[2]],
            'most_helpful_cross_effect': [helpful[1], helpful[2]]}


def result_summary(base, singles, dilution, joint, pairs):
    s = [[base[str(j)]['nll'] - singles[i][j]['nll'] for i in range(8)] for j in range(8)]
    off = [s[j][i] for j in range(8) for i in range(8) if i != j]
    diagonal = [s[i][i] for i in range(8)]
    own_full = [dilution[i]['1']['nll'] for i in range(8)]
    own_eighth = [dilution[i]['0.125']['nll'] for i in range(8)]
    return {
        'interaction_matrix_rows_eval_j_columns_update_i': s,
        'diagonal': {'mean': sum(diagonal) / 8, 'positive': sum(x > 0 for x in diagonal)},
        'off_diagonal': {'mean': sum(off) / len(off), 'positive': sum(x > 0 for x in off),
                         'negative': sum(x < 0 for x in off), 'count': len(off)},
        'mean_nll_increase_from_full_to_one_eighth': sum(a-b for a,b in zip(own_eighth, own_full))/8,
        'selected_pairs': pairs,
        'joint_macro': {name: {'nll': sum(row[str(i)]['nll'] for i in range(8))/8,
                               'em': sum(row[str(i)]['em'] for i in range(8))/8}
                        for name, row in joint.items()},
    }


def evaluate(cli) -> None:
    cap = cli.out / 'capture'
    manifest_path = cap / 'manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError('Complete same-round capture is required; run capture first')
    cap_manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    for name, digest in cap_manifest['checkpoint_sha256'].items():
        if file_sha256(cap / name) != digest:
            raise ValueError(f'Captured checkpoint changed: {name}')
    for i in range(8):
        data = Path('dataset/zsre/train') / f'{i}.jsonl'
        if file_sha256(data) != cap_manifest['train_data_sha256'][str(i)]:
            raise ValueError(f'Training file changed: {data}')
    if not math.isclose(cap_manifest['actual_lora_scale'], SCALE):
        raise ValueError('LoRA scaling mismatch')
    original_config = DEFAULT_ORIGINAL / 'training_config.json'
    if file_sha256(original_config) != cap_manifest['train_config_sha256']:
        raise ValueError('Original training configuration changed')
    saved = json.loads(original_config.read_text(encoding='utf-8'))
    args = SimpleNamespace(**saved)
    set_global_seed(args.seed, device=args.device, deterministic=args.deterministic)
    tokenizer = load_tokenizer(args)
    model = load_model(args)  # plain backbone, no active PEFT adapter
    global_state = torch.load(cap / 'global_before.pt', map_location='cpu', weights_only=True)
    local_states = [torch.load(cap / f'client_{i}.pt', map_location='cpu', weights_only=True)
                    for i in range(8)]
    post_state = torch.load(cap / 'global_after_fedit.pt', map_location='cpu', weights_only=True)
    if set(post_state) != set(global_state) or any(set(state) != set(global_state)
                                                   for state in local_states):
        raise ValueError('Captured round has incompatible LoRA keys')
    for key in global_state:
        expected = sum(state[key].float() / 8 for state in local_states)
        if not torch.allclose(post_state[key].float(), expected, rtol=1e-5, atol=1e-6):
            raise ValueError(f'Post-round FedIT factor aggregation mismatch: {key}')
    patcher = RoundPatcher(model, global_state, local_states)
    if len(patcher.global_pairs) != 56:
        raise ValueError('Expected 56 q/v LoRA layers for Qwen3-1.7B')
    validation = [read_jsonl(Path('dataset/zsre/test') / f'{i}.jsonl') for i in range(8)]
    training = [read_jsonl(Path('dataset/zsre/train') / f'{i}.jsonl') for i in range(8)]
    for i in range(8):
        if len(validation[i]) != 500 or len(training[i]) != 500:
            raise ValueError('Expected 500 train and 500 validation questions per client')
        if [row['sample_id'] for row in validation[i]] != [row['sample_id'] for row in training[i]]:
            raise ValueError('zsRE train and rephrased validation facts/order differ')
        if any(a['input_ids'] == b['input_ids'] for a,b in zip(training[i], validation[i])):
            raise ValueError('A training question is identical to its validation rephrase')
    encoded = [[encode_zsre_training_example(tokenizer, row) for row in group]
               for group in validation]
    output = cli.out / 'evaluation'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'metrics').mkdir(exist_ok=True)
    (output / 'predictions').mkdir(exist_ok=True)
    eval_manifest = {
        'capture_manifest_sha256': file_sha256(manifest_path),
        'protocol': generation_metadata(tokenizer), 'split': 'dataset/zsre/test/<client>.jsonl',
        'validation_relation': 'held-out rephrasings of the same facts as local training; no identical questions',
        'validation_sha256': {str(i): file_sha256(Path('dataset/zsre/test') / f'{i}.jsonl')
                              for i in range(8)},
        'target_round_zero_based': cap_manifest['target_round_zero_based'],
        'nll': 'per-example mean gold answer plus EOS token NLL',
        'matrix_average': 'mean of actual scaled dense client delta matrices; not FedIT factor-average checkpoint',
        'pair_selection': 'min/max sum of two cross-client NLL improvements; exploratory, selected on this validation set',
    }
    eval_manifest_path = output / 'manifest.json'
    if eval_manifest_path.is_file():
        if json.loads(eval_manifest_path.read_text(encoding='utf-8')) != eval_manifest:
            raise ValueError('Evaluation manifest differs; use a new output directory')
    else:
        atomic_json(eval_manifest_path, eval_manifest)

    def measure(name, coeffs, clients, generate=False):
        patcher.apply(coeffs)
        rank_path = output / 'metrics' / f'{name}_rank.json'
        if not rank_path.is_file():
            atomic_json(rank_path, patcher.ranks(coeffs))
        result = {}
        for j in clients:
            path = output / 'metrics' / f'{name}_client_{j}.json'
            pred_path = output / 'predictions' / f'{name}_client_{j}.jsonl'
            if path.is_file():
                cached = json.loads(path.read_text(encoding='utf-8'))
                if not generate or ('em' in cached and pred_path.is_file()):
                    result[str(j)] = cached
                    continue
            nll = answer_nll(model, encoded[j], cli.batch_size)
            row = {'nll': nll, 'num_samples': len(validation[j]), 'client_id': j,
                   'condition': name, 'rank': json.loads(rank_path.read_text(encoding='utf-8'))['max_rank']}
            if generate:
                metrics, predictions = generate_and_score(model, tokenizer, validation[j], cli.batch_size)
                row.update(metrics)
                write_predictions(pred_path, predictions)
            atomic_json(path, row)
            result[str(j)] = row
            print(f'{name} eval={j} NLL={nll:.4f}' +
                  (f" EM={row['em']:.2f}" if generate else ''), flush=True)
        return result

    try:
        base = measure('global', {}, range(8), generate=True)
        singles = [measure(f'single_{i}', {i: 1.0}, range(8)) for i in range(8)]
        s = [[base[str(j)]['nll'] - singles[i][str(j)]['nll'] for i in range(8)] for j in range(8)]
        pairs = choose_pairs(s)
        atomic_json(output / 'selected_pairs.json', pairs)
        dilution = {}
        for i in range(8):
            dilution[i] = {}
            for alpha in ALPHAS:
                key = f'{alpha:g}'
                name = 'global' if alpha == 0 else (f'single_{i}' if alpha == 1 else f'scale_{i}_{key}')
                # For alpha=1, add generation to the existing A NLL result.
                dilution[i][key] = measure(name, {} if alpha == 0 else {i: alpha},
                                            [i], generate=True)[str(i)]
        pair_ids = {i for pair in pairs.values() for i in pair}
        joint = {'global': base}
        for i in sorted(pair_ids):
            joint[f'single_{i}'] = measure(f'single_{i}', {i: 1.0}, range(8), generate=True)
        joint['matrix_average'] = measure('matrix_average', {i: .125 for i in range(8)},
                                           range(8), generate=True)
        joint['direct_sum'] = measure('direct_sum', {i: 1.0 for i in range(8)},
                                       range(8), generate=True)
        for tag, pair in pairs.items():
            joint[f'pair_{tag}_{pair[0]}_{pair[1]}'] = measure(
                f'pair_{tag}_{pair[0]}_{pair[1]}', {pair[0]: 1.0, pair[1]: 1.0},
                range(8), generate=True)
    finally:
        patcher.restore_base()

    compact_singles = [[singles[i][str(j)] for j in range(8)] for i in range(8)]
    compact_dilution = [{key: dilution[i][key] for key in dilution[i]} for i in range(8)]
    summary = result_summary(base, compact_singles, compact_dilution, joint, pairs)
    atomic_json(output / 'summary.json', summary)
    with (output / 'interaction_nll_gain.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        writer.writerow(['eval_client_j / update_client_i'] + [f'update_{i}' for i in range(8)])
        for j, row in enumerate(s):
            writer.writerow([f'eval_{j}'] + row)
    with (output / 'dilution.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        writer.writerow(['client', 'alpha', 'nll', 'em', 'rank'])
        for i in range(8):
            for alpha in ALPHAS:
                row = dilution[i][f'{alpha:g}']
                writer.writerow([i, alpha, row['nll'], row['em'], row['rank']])
    with (output / 'joint.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        writer.writerow(['condition', 'client', 'nll', 'em', 'rank'])
        for name, rows in joint.items():
            for j in range(8):
                row = rows[str(j)]
                writer.writerow([name, j, row['nll'], row['em'], row['rank']])
    print('Completed A, B, C. Summary:', json.dumps(summary, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('capture', 'evaluate'))
    parser.add_argument('--round', type=int, default=1, help='zero-based target FedIT round; replays all prior rounds')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--batch-size', type=int, default=8)
    cli = parser.parse_args()
    if cli.round < 0 or cli.round >= 10 or cli.batch_size < 1:
        parser.error('Require 0 <= --round < 10 and positive --batch-size')
    if cli.stage == 'capture':
        capture(cli)
    else:
        evaluate(cli)


if __name__ == '__main__':
    main()
