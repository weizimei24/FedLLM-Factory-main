"""Offline diagnostic of FedFH module-one input directions on trained zsRE LoRAs.

Run at the repository root with the same Python environment as training::

    python -u tools/fedfh_module1_diagnostic.py

This script never trains, changes base weights, or writes to existing runs. It
uses the exact local-only checkpoint and zsre_chat_eos_v2 generation protocol.
Gold-answer NLL is the mean per-example, per-target-token negative log
likelihood under teacher forcing; the supervised EOS token is included.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import get_peft_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_zsre import adapter_path, generate_and_score, generation_metadata, read_jsonl
from utils.fh_utils import collect_input_geometry, geometry_group, module_name_from_lora_key
from utils.model_utils import load_lora_config, load_model, load_tokenizer
from utils.seed_utils import set_global_seed
from utils.zsre_utils import ZSRE_PROTOCOL, encode_zsre_training_example


VARIANTS = ("Original", "Top-32", "Rest", "Random-32")
METRIC_FIELDS = ("em", "f1", "answer_nll", "eos_rate", "max_tokens_rate")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def stable_random_basis(width: int, k: int, seed: int, client_id: int, group: str) -> torch.Tensor:
    """One orthonormal basis per client/input group, independent of call order."""
    if not 0 < k <= width:
        raise ValueError(f"Invalid random subspace dimension {k} for input width {width}")
    label = f"{seed}:{client_id}:{group}".encode("utf-8")
    generator = torch.Generator(device="cpu").manual_seed(int.from_bytes(hashlib.sha256(label).digest()[:8], "little") % (2**63))
    q, _ = torch.linalg.qr(torch.randn(width, k, generator=generator), mode="reduced")
    return q.contiguous()


def project_a(a: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """A P with P = QQ^T; preserve the trained A shape and dtype."""
    if a.ndim != 2 or q.ndim != 2 or a.shape[1] != q.shape[0]:
        raise ValueError(f"Incompatible A/Q shapes: {tuple(a.shape)}, {tuple(q.shape)}")
    if not torch.allclose(q.T @ q, torch.eye(q.shape[1]), atol=1e-3, rtol=1e-3):
        raise ValueError("Projection basis is not orthonormal")
    return ((a.float() @ q.float()) @ q.float().T).to(a.dtype)


def projected_states(original: dict, geometry: dict, top_k: int, seed: int, client_id: int) -> dict:
    """Build four separate LoRA states; B factors and all other tensors stay fixed."""
    states = {variant: {key: value.clone() for key, value in original.items()} for variant in VARIANTS}
    a_keys = [key for key in original if ".lora_A." in key]
    if not a_keys:
        raise ValueError("Checkpoint contains no LoRA A factors")
    random_bases = {}
    for key in a_keys:
        group = geometry_group(module_name_from_lora_key(key))
        if group not in geometry:
            raise KeyError(f"Missing input geometry for {group}")
        a = original[key]
        q = geometry[group]["Q"]
        if q.shape != (a.shape[1], top_k):
            raise ValueError(f"Expected Q ({a.shape[1]}, {top_k}) for {key}, got {tuple(q.shape)}")
        random_bases.setdefault(group, stable_random_basis(a.shape[1], top_k, seed, client_id, group))
        top = project_a(a, q)
        states["Top-32"][key] = top
        states["Rest"][key] = (a.float() - top.float()).to(a.dtype)
        states["Random-32"][key] = project_a(a, random_bases[group])
    return states


@torch.no_grad()
def answer_nll(model, encoded_rows: list[dict], batch_size: int) -> float:
    """Macro mean over samples of mean target-token NLL, including EOS."""
    if not encoded_rows or batch_size < 1:
        raise ValueError("Need nonempty encoded rows and a positive NLL batch size")
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    for start in range(0, len(encoded_rows), batch_size):
        chunk = encoded_rows[start:start + batch_size]
        width = max(sum(row["attention_mask"]) for row in chunk)
        ids = torch.tensor([row["input_ids"][:width] for row in chunk], dtype=torch.long, device=device)
        mask = torch.tensor([row["attention_mask"][:width] for row in chunk], dtype=torch.long, device=device)
        labels = torch.tensor([row["labels"][:width] for row in chunk], dtype=torch.long, device=device)
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, :-1]
        targets = labels[:, 1:]
        valid = targets.ne(-100)
        counts = valid.sum(dim=1)
        if torch.any(counts == 0):
            raise ValueError("An example has no supervised answer/EOS tokens")
        token_losses = F.cross_entropy(logits[valid].float(), targets[valid], reduction="none")
        sample_index = valid.nonzero(as_tuple=True)[0]
        by_sample = torch.zeros(len(chunk), dtype=torch.float32, device=device)
        by_sample.scatter_add_(0, sample_index, token_losses)
        total_loss += float((by_sample / counts).sum().item())
    return total_loss / len(encoded_rows)


def check_data(train: list[dict], test: list[dict], client_id: int, full_run: bool) -> None:
    if full_run and (len(train) != 500 or len(test) != 500):
        raise ValueError(f"Client {client_id}: expected 500 train and 500 test rows")
    if not train or not test:
        raise ValueError(f"Client {client_id}: empty train/test data")
    for rows in (train, test):
        if any(row.get("client_id") != client_id for row in rows):
            raise ValueError(f"Client {client_id}: row has wrong client_id")
    if [r["sample_id"] for r in train] != [r["sample_id"] for r in test]:
        raise ValueError(f"Client {client_id}: train/test facts or ordering differ from zsRE rephrase protocol")
    if [r["label"] for r in train] != [r["label"] for r in test]:
        raise ValueError(f"Client {client_id}: train/test gold answers differ")


def verify_checkpoint(model, checkpoint: dict, rank: int) -> None:
    state = model.state_dict()
    expected = {key for key in state if ".lora_A." in key or ".lora_B." in key}
    if set(checkpoint) != expected:
        missing, extra = expected - set(checkpoint), set(checkpoint) - expected
        raise ValueError(f"Checkpoint LoRA key mismatch: {len(missing)} missing, {len(extra)} extra")
    for key, value in checkpoint.items():
        if value.shape != state[key].shape:
            raise ValueError(f"Checkpoint shape mismatch: {key}")
        if ".lora_A." in key and value.shape[0] != rank:
            raise ValueError(f"Checkpoint rank mismatch: {key}")


def write_predictions(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def build_summary(output: Path, num_clients: int) -> dict:
    per_client = {}
    records = []
    for client_id in range(num_clients):
        per_client[str(client_id)] = {}
        for variant in VARIANTS:
            path = output / "metrics" / f"client_{client_id}_{variant}.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            row = json.loads(path.read_text(encoding="utf-8"))
            per_client[str(client_id)][variant] = row
            records.append(row)
    macro = {}
    for variant in VARIANTS:
        subset = [per_client[str(i)][variant] for i in range(num_clients)]
        macro[variant] = {key: sum(row[key] for row in subset) / num_clients for key in METRIC_FIELDS}
        macro[variant]["num_samples"] = sum(row["num_samples"] for row in subset)
        records.append({"client_id": "macro", "variant": variant, **macro[variant]})
    csv_path = output / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["client_id", "variant", "num_samples", *METRIC_FIELDS], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    summary = {"per_client": per_client, "macro": macro}
    atomic_json(output / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="zsre8_chat_eos_v2")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset/zsre"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stat-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--nll-batch-size", type=int, default=8)
    parser.add_argument("--clients", type=int, nargs="+", default=list(range(8)), help="subset for a separate smoke run")
    parser.add_argument("--max-test-samples", type=int, default=None, help="smoke-test prefix; omit for full 500")
    parser.add_argument("--resume", action="store_true", help="resume only a matching, incomplete output directory")
    cli = parser.parse_args()
    if cli.top_k != 32:
        raise ValueError("This diagnostic specifically tests Top-32; --top-k must be 32")
    if any(v < 1 for v in (cli.stat_batch_size, cli.eval_batch_size, cli.nll_batch_size)):
        raise ValueError("Batch sizes must be positive")
    if sorted(set(cli.clients)) != sorted(cli.clients) or any(i not in range(8) for i in cli.clients):
        raise ValueError("--clients must be unique integers from 0 to 7")
    full_run = cli.clients == list(range(8)) and cli.max_test_samples is None
    output = cli.output_dir or Path("exp") / cli.run / "diagnostics" / "fedfh_module1_top32"
    run_dir = Path("exp") / cli.run
    adapter_root = run_dir / "local" / "adapter"
    configs = list(adapter_root.glob("*/training_config.json"))
    if len(configs) != 1:
        raise ValueError(f"Expected exactly one Local-only training config under {adapter_root}, found {len(configs)}")
    config_path = configs[0]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (config.get("alg"), config.get("dataset"), config.get("cn"), config.get("lora_rank")) != ("localonly", "zsre", 8, 8):
        raise ValueError("Unexpected training method, dataset, client count or LoRA rank")
    if config.get("lora_alpha") != 32 or config.get("task_type") != "CAUSAL_LM":
        raise ValueError("Unexpected trained LoRA scaling or task type")
    args = SimpleNamespace(**config)
    expected_adapter_dir = adapter_path(args)
    if expected_adapter_dir.resolve() != config_path.parent.resolve():
        raise ValueError("Training config and adapter path disagree")
    paths = {i: config_path.parent / f"client_{i}.pt" for i in range(8)}
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("One or more of the eight trained Local-only adapters is missing")
    baseline_path = run_dir / "local" / "evaluation" / "zsre" / "metrics.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline["generation"]["protocol"] != ZSRE_PROTOCOL:
        raise ValueError("Prior Local-only evaluation uses another zsRE protocol")
    manifest = {
        "experiment": "fedfh_module1_top32_direction_diagnostic",
        "run": cli.run, "protocol": ZSRE_PROTOCOL, "clients": cli.clients,
        "max_test_samples": cli.max_test_samples, "top_k": cli.top_k,
        "seed": cli.seed, "stat_batch_size": cli.stat_batch_size,
        "eval_batch_size": cli.eval_batch_size, "nll_batch_size": cli.nll_batch_size,
        "nll_definition": "mean per-example mean gold answer token NLL including supervised EOS",
        "checkpoint_config_sha256": file_sha256(config_path),
        "checkpoint_sha256": {str(i): file_sha256(paths[i]) for i in range(8)},
        "train_sha256": {str(i): file_sha256(cli.data_dir / "train" / f"{i}.jsonl") for i in cli.clients},
        "test_sha256": {str(i): file_sha256(cli.data_dir / "test" / f"{i}.jsonl") for i in cli.clients},
    }
    manifest_path = output / "manifest.json"
    if output.exists() and any(output.iterdir()):
        if not cli.resume or not manifest_path.is_file() or json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise FileExistsError(f"Output exists; use a new --output-dir, or --resume with matching manifest: {output}")
    else:
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(manifest_path, manifest)
    (output / "metrics").mkdir(exist_ok=True)
    (output / "predictions").mkdir(exist_ok=True)
    (output / "geometry").mkdir(exist_ok=True)

    set_global_seed(cli.seed, device=args.device, deterministic=bool(args.deterministic))
    tokenizer = load_tokenizer(args)
    model = get_peft_model(load_model(args), load_lora_config(args))
    if generation_metadata(tokenizer) != baseline["generation"]:
        raise ValueError("Current tokenizer generation protocol differs from prior Local-only evaluation")
    for client_id in cli.clients:
        print(f"Client {client_id}: loading trained adapter and computing train-input geometry", flush=True)
        train = read_jsonl(cli.data_dir / "train" / f"{client_id}.jsonl")
        test = read_jsonl(cli.data_dir / "test" / f"{client_id}.jsonl")
        check_data(train, test, client_id, full_run)
        if cli.max_test_samples is not None:
            if cli.max_test_samples < 1:
                raise ValueError("--max-test-samples must be positive")
            test = test[:cli.max_test_samples]
        checkpoint = torch.load(paths[client_id], map_location="cpu", weights_only=True)
        verify_checkpoint(model, checkpoint, args.lora_rank)
        model.load_state_dict(checkpoint, strict=False)
        formatted_train = Dataset.from_list([encode_zsre_training_example(tokenizer, row) for row in train])
        geometry = collect_input_geometry(model, formatted_train, top_k=cli.top_k, batch_size=cli.stat_batch_size)
        groups = {}
        for name, entry in geometry.items():
            coverage = float(entry["Lam"].sum().item() / entry["trace"]) if entry["trace"] > 0 else 0.0
            groups[name] = {"input_dim": entry["Q"].shape[0], "k": entry["Q"].shape[1],
                            "trace": entry["trace"], "top32_covariance_coverage": coverage,
                            "tokens": entry["tokens"]}
        atomic_json(output / "geometry" / f"client_{client_id}.json", {"groups": groups,
                    "mean_top32_covariance_coverage": sum(v["top32_covariance_coverage"] for v in groups.values()) / len(groups)})
        states = projected_states(checkpoint, geometry, cli.top_k, cli.seed, client_id)
        encoded_test = [encode_zsre_training_example(tokenizer, row) for row in test]
        for variant in VARIANTS:
            metric_path = output / "metrics" / f"client_{client_id}_{variant}.json"
            prediction_path = output / "predictions" / f"client_{client_id}_{variant}.jsonl"
            if cli.resume and metric_path.is_file() and prediction_path.is_file():
                print(f"Client {client_id} {variant}: already complete, skipping", flush=True)
                continue
            model.load_state_dict(states[variant], strict=False)
            metrics, predictions = generate_and_score(model, tokenizer, test, cli.eval_batch_size)
            nll = answer_nll(model, encoded_test, cli.nll_batch_size)
            if not math.isfinite(nll):
                raise FloatingPointError(f"Nonfinite answer NLL: client {client_id}, {variant}")
            metrics.update({"client_id": client_id, "variant": variant, "answer_nll": nll})
            if variant == "Original" and full_run:
                reference = baseline["per_client"][str(client_id)]
                for key in ("em", "f1", "num_samples", "eos_rate", "max_tokens_rate"):
                    if not math.isclose(metrics[key], reference[key], abs_tol=1e-6):
                        raise AssertionError(f"Original client {client_id} {key} does not reproduce the saved evaluation")
            write_predictions(prediction_path, predictions)
            atomic_json(metric_path, metrics)
            print(f"Client {client_id} {variant}: EM {metrics['em']:.2f} F1 {metrics['f1']:.2f} NLL {nll:.4f}", flush=True)
        del geometry, states, formatted_train, encoded_test
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if full_run:
        summary = build_summary(output, 8)
        print("Macro results:", flush=True)
        for variant in VARIANTS:
            row = summary["macro"][variant]
            print(f"{variant}: EM {row['em']:.2f} F1 {row['f1']:.2f} NLL {row['answer_nll']:.4f}", flush=True)
    else:
        print("Partial smoke run complete; full-run summary intentionally omitted.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
