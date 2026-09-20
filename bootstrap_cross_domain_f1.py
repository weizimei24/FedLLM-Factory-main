"""Bootstrap uncertainty for saved cross-domain QA prediction records.

The script resamples the 200 test examples *within each domain*.  It reports
both method-specific F1 uncertainty and a paired FedRot-LoRA minus FedIT
difference: paired resampling is valid because both methods predict on the
same fixed examples, and is substantially more informative than comparing two
independent bootstrap intervals.

Examples:
    python -X utf8 bootstrap_cross_domain_f1.py --rounds all
    python -X utf8 bootstrap_cross_domain_f1.py --rounds 9 --num-resamples 20000

All reported F1 values and standard deviations are on the 0--100 scale.
Bootstrap reflects uncertainty from the finite 200-example test sample only;
it does not estimate variation from random training seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


DOMAINS = ("squad", "newsqa", "triviaqa", "naturalquestions")
PREDICTION_PATTERN = re.compile(r"round_(\d+)_(squad|newsqa|triviaqa|naturalquestions)\.jsonl$")
FIELDS = (
    "round", "domain", "num_samples", "fedit_f1", "fedit_bootstrap_sd",
    "fedit_ci95_low", "fedit_ci95_high", "fedrot_f1", "fedrot_bootstrap_sd",
    "fedrot_ci95_low", "fedrot_ci95_high", "fedrot_minus_fedit",
    "difference_bootstrap_sd", "difference_ci95_low", "difference_ci95_high",
    "p_bootstrap_difference_gt_zero",
)


def _discover(prediction_dir: Path) -> dict[tuple[int, str], Path]:
    if not prediction_dir.is_dir():
        raise FileNotFoundError(f"Prediction directory does not exist: {prediction_dir}")
    paths: dict[tuple[int, str], Path] = {}
    for path in prediction_dir.glob("round_*_*.jsonl"):
        match = PREDICTION_PATTERN.match(path.name)
        if match:
            key = (int(match.group(1)), match.group(2))
            if key in paths:
                raise ValueError(f"Duplicate prediction file for round/domain {key}: {path}")
            paths[key] = path
    if not paths:
        raise FileNotFoundError(f"No cross-domain prediction files found in {prediction_dir}")
    return paths


def _read_f1_by_id(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            sample_id = str(row.get("id", ""))
            if not sample_id:
                raise ValueError(f"Missing id in {path}:{line_number}")
            if sample_id in values:
                raise ValueError(f"Duplicate id {sample_id!r} in {path}")
            try:
                values[sample_id] = float(row["f1"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid F1 in {path}:{line_number}") from error
    if not values:
        raise ValueError(f"No predictions in {path}")
    return values


def _parse_rounds(value: str, common_rounds: list[int]) -> list[int]:
    if value == "all":
        return common_rounds
    if value == "last":
        return [max(common_rounds)]
    selected = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    missing = sorted(set(selected) - set(common_rounds))
    if missing:
        raise ValueError(f"Requested rounds are unavailable for one or both methods: {missing}")
    return selected


def _summary(samples: np.ndarray) -> tuple[float, float, float]:
    return (
        float(np.std(samples, ddof=1)),
        float(np.percentile(samples, 2.5)),
        float(np.percentile(samples, 97.5)),
    )


def _bootstrap_row(
    round_label: str | int,
    domain: str,
    fedit_scores: np.ndarray,
    fedrot_scores: np.ndarray,
    num_resamples: int,
    generator: np.random.Generator,
) -> dict:
    """Compute method and paired-difference bootstrap summaries in percent."""
    indices = generator.integers(0, len(fedit_scores), size=(num_resamples, len(fedit_scores)))
    fedit_bootstrap = fedit_scores[indices].mean(axis=1) * 100.0
    fedrot_bootstrap = fedrot_scores[indices].mean(axis=1) * 100.0
    difference_bootstrap = fedrot_bootstrap - fedit_bootstrap
    fedit_sd, fedit_low, fedit_high = _summary(fedit_bootstrap)
    fedrot_sd, fedrot_low, fedrot_high = _summary(fedrot_bootstrap)
    difference_sd, difference_low, difference_high = _summary(difference_bootstrap)
    return {
        "round": round_label,
        "domain": domain,
        "num_samples": len(fedit_scores),
        "fedit_f1": fedit_scores.mean() * 100.0,
        "fedit_bootstrap_sd": fedit_sd,
        "fedit_ci95_low": fedit_low,
        "fedit_ci95_high": fedit_high,
        "fedrot_f1": fedrot_scores.mean() * 100.0,
        "fedrot_bootstrap_sd": fedrot_sd,
        "fedrot_ci95_low": fedrot_low,
        "fedrot_ci95_high": fedrot_high,
        "fedrot_minus_fedit": (fedrot_scores - fedit_scores).mean() * 100.0,
        "difference_bootstrap_sd": difference_sd,
        "difference_ci95_low": difference_low,
        "difference_ci95_high": difference_high,
        "p_bootstrap_difference_gt_zero": float(np.mean(difference_bootstrap > 0.0)),
    }


def _print_row(row: dict) -> None:
    print(
        f"round={str(row['round']):<20} domain={row['domain']:<16} "
        f"FedRot−FedIT={row['fedrot_minus_fedit']:+.3f} "
        f"SE={row['difference_bootstrap_sd']:.3f} "
        f"95% CI=[{row['difference_ci95_low']:+.3f}, {row['difference_ci95_high']:+.3f}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired bootstrap F1 analysis for cross-domain QA predictions")
    parser.add_argument(
        "--fedit-dir",
        default="exp/cross_domain_mrqa_fedit/evaluation/cross_domain/predictions",
        help="FedIT prediction directory",
    )
    parser.add_argument(
        "--fedrot-dir",
        default="exp/cross_domain_mrqa_fedrot/evaluation/cross_domain/predictions",
        help="FedRot-LoRA prediction directory",
    )
    parser.add_argument("--rounds", default="all", help="all, last, or comma-separated zero-based rounds")
    parser.add_argument("--num-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-round-average", action=argparse.BooleanOptionalAction, default=True,
        help="also bootstrap each method's per-example mean over all selected rounds",
    )
    parser.add_argument(
        "--output",
        default="exp/cross_domain_mrqa_bootstrap/bootstrap_f1.csv",
        help="CSV output path",
    )
    args = parser.parse_args()
    if args.num_resamples < 2:
        raise ValueError("--num-resamples must be at least 2")

    fedit_paths = _discover(Path(args.fedit_dir))
    fedrot_paths = _discover(Path(args.fedrot_dir))
    common_rounds = sorted({round_idx for round_idx, _ in fedit_paths} & {round_idx for round_idx, _ in fedrot_paths})
    if not common_rounds:
        raise ValueError("FedIT and FedRot-LoRA have no common evaluated rounds")
    rounds = _parse_rounds(args.rounds, common_rounds)
    generator = np.random.default_rng(args.seed)
    output_rows: list[dict] = []
    selected_score_vectors: dict[str, list[tuple[list[str], np.ndarray, np.ndarray]]] = {
        domain: [] for domain in DOMAINS
    }

    for round_idx in rounds:
        for domain in DOMAINS:
            key = (round_idx, domain)
            if key not in fedit_paths or key not in fedrot_paths:
                raise ValueError(f"Missing {domain} predictions for round {round_idx} in one method")
            fedit = _read_f1_by_id(fedit_paths[key])
            fedrot = _read_f1_by_id(fedrot_paths[key])
            if fedit.keys() != fedrot.keys():
                only_fedit = len(fedit.keys() - fedrot.keys())
                only_fedrot = len(fedrot.keys() - fedit.keys())
                raise ValueError(
                    f"Cannot pair {domain} round {round_idx}: "
                    f"{only_fedit} FedIT-only and {only_fedrot} FedRot-only IDs"
                )
            ids = sorted(fedit)
            fedit_scores = np.asarray([fedit[sample_id] for sample_id in ids], dtype=float)
            fedrot_scores = np.asarray([fedrot[sample_id] for sample_id in ids], dtype=float)
            selected_score_vectors[domain].append((ids, fedit_scores, fedrot_scores))
            row = _bootstrap_row(round_idx, domain, fedit_scores, fedrot_scores, args.num_resamples, generator)
            output_rows.append(row)
            _print_row(row)

    if args.include_round_average:
        label = "mean_selected_rounds"
        for domain, vectors in selected_score_vectors.items():
            base_ids = vectors[0][0]
            if any(ids != base_ids for ids, _, _ in vectors[1:]):
                raise ValueError(f"Cannot average rounds for {domain}: prediction IDs changed across rounds")
            fedit_scores = np.stack([scores for _, scores, _ in vectors]).mean(axis=0)
            fedrot_scores = np.stack([scores for _, _, scores in vectors]).mean(axis=0)
            row = _bootstrap_row(label, domain, fedit_scores, fedrot_scores, args.num_resamples, generator)
            output_rows.append(row)
            _print_row(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Saved {len(output_rows)} rows to {output}")


if __name__ == "__main__":
    main()
