"""Plot per-domain F1 trajectories from FedIT and FedRot-LoRA result CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def _read(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fedit", default="exp/cross_domain_mrqa_fedit/evaluation/cross_domain/results.csv")
    parser.add_argument("--fedrot", default="exp/cross_domain_mrqa_fedrot/evaluation/cross_domain/results.csv")
    parser.add_argument("--output", default="exp/cross_domain_mrqa_f1_by_round.png")
    args = parser.parse_args()
    figure, axis = plt.subplots(figsize=(10, 5))
    colours = {"squad": "C0", "newsqa": "C1", "triviaqa": "C2", "naturalquestions": "C3"}
    for label, path, style in (("FedIT", Path(args.fedit), "-"), ("FedRot-LoRA", Path(args.fedrot), "--")):
        grouped: dict[str, list[dict]] = {}
        for row in _read(path):
            if row["domain"] != "macro_average":
                grouped.setdefault(row["domain"], []).append(row)
        for domain, rows in grouped.items():
            rows.sort(key=lambda row: int(row["round"]))
            axis.plot([int(row["round"]) for row in rows], [float(row["f1"]) for row in rows],
                      linestyle=style, marker="o", color=colours.get(domain), label=f"{label} — {domain}")
    axis.set(xlabel="Communication round (zero-based)", ylabel="F1 (%)", title="Cross-domain MRQA global-adapter F1")
    axis.grid(True, alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
