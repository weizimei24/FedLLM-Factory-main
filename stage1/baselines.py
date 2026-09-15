import json
from pathlib import Path

from stage1.config import DOMAINS
from stage1.data import load_domain_rows
from stage1.evaluator import Stage1Evaluator
from stage1.federated import FederatedRunner
from stage1.model import load_base_model, load_lora_model, load_tokenizer, release_memory
from stage1.results import save_method_result, write_json
from stage1.trainer import set_seed, train_phase


def load_experiment_rows(config):
    result = {}
    for domain, _ in DOMAINS:
        train = load_domain_rows(config.data_dir, domain, "train")
        test = load_domain_rows(config.data_dir, domain, "test")
        if config.smoke:
            # Smoke metrics are not paper results. Short examples exercise every
            # path while keeping the 6x6 Local transfer check practical.
            key = lambda row: len(row["question"]) + len(row["answer"])
            train = sorted(train, key=key)
            test = sorted(test, key=key)
        result[domain] = {
            "train": train[:config.train_samples],
            "test": test[:config.test_samples],
        }
    return result


def _evaluate_domains(model, evaluator, rows, output_dir: Path, prefix: str = "") -> dict:
    result = {}
    for domain, splits in rows.items():
        print(f"Evaluating {prefix or 'model'} on {domain}")
        result[domain] = evaluator.evaluate(
            model, splits["test"], output_dir / f"{prefix}{domain}.jsonl"
        )
        print(json.dumps(result[domain], ensure_ascii=False))
    return result


def run_base(config, rows):
    tokenizer = load_tokenizer(config.model_path)
    model = load_base_model(config.model_path, for_training=False)
    evaluator = Stage1Evaluator(tokenizer, config)
    per_domain = _evaluate_domains(model, evaluator, rows, config.run_dir / "base" / "predictions")
    save_method_result(config.run_dir, "base", per_domain)
    del model
    release_memory()


def run_local(config, rows):
    tokenizer = load_tokenizer(config.model_path)
    evaluator = Stage1Evaluator(tokenizer, config)
    per_domain = {}
    history = {}
    for client_index, (train_domain, splits) in enumerate(rows.items()):
        # Every independent Local run starts from the same LoRA initialization.
        set_seed(config.seed)
        model = load_lora_model(config)
        history[train_domain] = []
        for phase in range(config.phases):
            history[train_domain].append(train_phase(
                model, splits["train"], tokenizer, config,
                config.seed + phase * 100 + client_index,
                f"local {train_domain} phase={phase}",
            ))
        adapter_dir = config.run_dir / "local" / "adapters" / train_domain
        model.save_pretrained(adapter_dir)
        print(f"Evaluating local {train_domain} on {train_domain}")
        per_domain[train_domain] = evaluator.evaluate(
            model,
            splits["test"],
            config.run_dir / "local" / "predictions" / f"{train_domain}.jsonl",
        )
        print(json.dumps(per_domain[train_domain], ensure_ascii=False))
        del model
        release_memory()
    save_method_result(config.run_dir, "local", per_domain, {"training_history": history})


def run_centralized(config, rows):
    tokenizer = load_tokenizer(config.model_path)
    model = load_lora_model(config)
    combined = [row for domain in rows.values() for row in domain["train"]]
    history = []
    for phase in range(config.phases):
        history.append(train_phase(
            model, combined, tokenizer, config, config.seed + phase * 100,
            f"centralized phase={phase}",
        ))
    adapter_dir = config.run_dir / "centralized" / "adapter"
    model.save_pretrained(adapter_dir)
    evaluator = Stage1Evaluator(tokenizer, config)
    per_domain = _evaluate_domains(
        model, evaluator, rows, config.run_dir / "centralized" / "predictions"
    )
    save_method_result(config.run_dir, "centralized", per_domain, {"training_history": history})
    del model
    release_memory()


def run_federated(config, rows, method: str):
    tokenizer = load_tokenizer(config.model_path)
    model = load_lora_model(config)
    runner = FederatedRunner(
        model, tokenizer, {domain: split["train"] for domain, split in rows.items()}, config, method
    )
    history = runner.run()
    adapter_dir = config.run_dir / method / "adapter"
    model.save_pretrained(adapter_dir)
    evaluator = Stage1Evaluator(tokenizer, config)
    per_domain = _evaluate_domains(model, evaluator, rows, config.run_dir / method / "predictions")
    save_method_result(config.run_dir, method, per_domain, {"training_history": history})
    del model
    release_memory()


def run_method(config, method: str):
    # Keep the base adapter initialization identical across methods.
    set_seed(config.seed)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.run_dir / "config.json", config.to_dict())
    rows = load_experiment_rows(config)
    if method == "base":
        run_base(config, rows)
    elif method == "local":
        run_local(config, rows)
    elif method == "centralized":
        run_centralized(config, rows)
    elif method in {"fedit", "fedrotlora"}:
        run_federated(config, rows, method)
    else:
        raise ValueError(method)
