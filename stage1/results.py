import csv
import json
from pathlib import Path


MAIN_METRICS = (
    "eval_loss", "perplexity", "rouge1", "rouge2", "rougeL",
    "avg_generation_tokens", "eos_rate", "max_tokens_rate",
)
METHOD_ORDER = ("base", "local", "centralized", "fedit", "fedrotlora")


def write_json(path: str | Path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def macro_average(per_domain: dict[str, dict]) -> dict:
    keys = per_domain[next(iter(per_domain))].keys()
    return {
        key: sum(float(metrics[key]) for metrics in per_domain.values()) / len(per_domain)
        for key in keys if all(isinstance(metrics.get(key), (int, float)) for metrics in per_domain.values())
    }


def save_method_result(run_dir: Path, method: str, per_domain: dict, extra: dict | None = None):
    payload = {"method": method, "per_domain": per_domain, "macro_average": macro_average(per_domain)}
    if extra:
        payload.update(extra)
    write_json(run_dir / method / "metrics.json", payload)
    refresh_main_table(run_dir)


def refresh_main_table(run_dir: Path):
    available = {}
    for method in METHOD_ORDER:
        path = run_dir / method / "metrics.json"
        if path.exists():
            available[method] = json.loads(path.read_text(encoding="utf-8"))
    if not available:
        return
    domains = list(next(iter(available.values()))["per_domain"].keys())
    path = run_dir / "main_table.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "metric", *domains, "macro_average"])
        for method in METHOD_ORDER:
            if method not in available:
                continue
            result = available[method]
            for metric in MAIN_METRICS:
                writer.writerow([
                    method,
                    metric,
                    *(result["per_domain"][domain].get(metric, "") for domain in domains),
                    result["macro_average"].get(metric, ""),
                ])


def save_transfer_tables(run_dir: Path, transfer: dict[str, dict[str, dict]]):
    domains = list(transfer.keys())
    write_json(run_dir / "local" / "transfer_matrix.json", transfer)
    for metric in MAIN_METRICS:
        path = run_dir / "local" / f"transfer_{metric}.csv"
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(["local_model", *domains])
            for train_domain in domains:
                writer.writerow([
                    train_domain,
                    *(transfer[train_domain][test_domain].get(metric, "") for test_domain in domains),
                ])

