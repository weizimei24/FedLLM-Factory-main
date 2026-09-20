"""Re-evaluate saved global adapters for the four-domain MRQA QA pilot.

Examples:
    python -X utf8 eval_cross_domain.py --config config.cross_domain_fedit.yaml --rounds all
    python -X utf8 eval_cross_domain.py --config config.cross_domain_fedrot.yaml --rounds 0,4
"""

from __future__ import annotations

import importlib
from pathlib import Path

import torch
from peft import get_peft_model

from utils.cross_domain_qa import CrossDomainQAEvaluator
from utils.model_utils import load_lora_config, load_model, load_tokenizer
from utils.options import build_parser
from utils.seed_utils import set_global_seed


def _adapter_root(args) -> Path:
    name = f"{args.alg}_{args.dataset}_{args.model}_{args.cn}c_{args.epoch}E_lr{args.lr}"
    return Path(args.suffix) / "adapter" / name


def _round_paths(root: Path, requested: str) -> list[tuple[int, Path]]:
    if requested == "all":
        candidates = sorted(root.glob("round_*"))
        paths = []
        for candidate in candidates:
            try:
                paths.append((int(candidate.name.removeprefix("round_")), candidate))
            except ValueError:
                continue
        if not paths:
            raise FileNotFoundError(f"No round_* adapters found under {root}")
        return paths
    rounds = [int(value.strip()) for value in requested.split(",") if value.strip()]
    paths = [(round_idx, root / f"round_{round_idx:03d}") for round_idx in rounds]
    missing = [str(path) for _, path in paths if not (path / "lora_weights.pt").is_file()]
    if missing:
        raise FileNotFoundError("Missing adapter snapshots: " + ", ".join(missing))
    return paths


def main() -> None:
    parser = build_parser()
    parser.add_argument("--rounds", default="all", help="all, or comma-separated zero-based round indices")
    args, _ = parser.parse_known_args()
    alg_module = importlib.import_module(f"alg.{args.alg}")
    if hasattr(alg_module, "add_args"):
        alg_module.add_args(parser)
    args = parser.parse_args()
    if not args.cross_domain_qa or args.dataset != "cross_domain_mrqa":
        raise ValueError("This entry point only supports the cross_domain_mrqa configuration")

    set_global_seed(args.seed, device=args.device, deterministic=args.deterministic)
    args.suffix = f"exp/{args.suffix}"
    root = _adapter_root(args)
    round_paths = _round_paths(root, args.rounds)
    tokenizer = load_tokenizer(args)
    model = get_peft_model(load_model(args), load_lora_config(args))
    evaluator = CrossDomainQAEvaluator(args, reset_results=True)
    for round_idx, path in round_paths:
        weights = torch.load(path / "lora_weights.pt", map_location="cpu")
        model.load_state_dict(weights, strict=False)
        metrics = evaluator.evaluate(model, tokenizer, round_idx=round_idx, method=args.alg)
        print(f"[Round {round_idx}] macro EM={metrics['macro_em']:.2f} F1={metrics['macro_f1']:.2f}")


if __name__ == "__main__":
    main()
