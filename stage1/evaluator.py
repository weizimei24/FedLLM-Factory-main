import json
import math
from pathlib import Path

import torch
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader

from stage1.data import QACollator, QADataset, chat_token_ids


class Stage1Evaluator:
    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config
        self.scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    @torch.inference_mode()
    def evaluate(self, model, rows: list[dict], predictions_path: str | Path | None = None) -> dict:
        model.eval()
        model.config.use_cache = True
        dataset = QADataset(rows, self.tokenizer, self.config.max_length)
        loader = DataLoader(
            dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            collate_fn=QACollator(self.tokenizer.pad_token_id),
        )
        total_nll = 0.0
        total_tokens = 0
        for batch in loader:
            batch = {key: value.to(model.device) for key, value in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                output = model(**batch)
            tokens = int((batch["labels"][:, 1:] != -100).sum().item())
            total_nll += output.loss.float().item() * tokens
            total_tokens += tokens

        sums = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        generation_lengths = []
        eos_count = 0
        max_count = 0
        records = []
        eos_ids = model.generation_config.eos_token_id
        eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [self.tokenizer.eos_token_id])

        batch_size = self.config.generation_batch_size
        for start in range(0, len(rows), batch_size):
            row_batch = rows[start:start + batch_size]
            prompts = [chat_token_ids(self.tokenizer, row["question"]) for row in row_batch]
            input_length = max(map(len, prompts))
            # Decoder-only batched generation requires left padding.
            padded = [[self.tokenizer.pad_token_id] * (input_length - len(ids)) + ids for ids in prompts]
            masks = [[0] * (input_length - len(ids)) + [1] * len(ids) for ids in prompts]
            input_ids = torch.tensor(padded, dtype=torch.long, device=model.device)
            attention_mask = torch.tensor(masks, dtype=torch.long, device=model.device)
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                max_new_tokens=self.config.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=list(eos_ids),
            )
            for index, row in enumerate(row_batch):
                generated = outputs[index, input_length:].tolist()
                eos_position = next((i for i, token in enumerate(generated) if token in eos_ids), None)
                met_eos = eos_position is not None
                if met_eos:
                    generated = generated[:eos_position + 1]
                hit_max = len(generated) >= self.config.max_new_tokens and not met_eos
                prediction = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
                scores = self.scorer.score(row["answer"], prediction)
                for key in sums:
                    sums[key] += scores[key].fmeasure
                generation_lengths.append(len(generated))
                eos_count += int(met_eos)
                max_count += int(hit_max)
                records.append({
                    "domain": row["domain"],
                    "qid": row["qid"],
                    "reference": row["answer"],
                    "prediction": prediction,
                    "generation_tokens": len(generated),
                    "met_eos": met_eos,
                    "hit_max_new_tokens": hit_max,
                })

        count = len(rows)
        mean_nll = total_nll / max(total_tokens, 1)
        result = {
            "eval_loss": mean_nll,
            "perplexity": math.exp(mean_nll) if mean_nll < 20 else float("inf"),
            "rouge1": sums["rouge1"] / max(count, 1),
            "rouge2": sums["rouge2"] / max(count, 1),
            "rougeL": sums["rougeL"] / max(count, 1),
            "avg_generation_tokens": sum(generation_lengths) / max(count, 1),
            "eos_rate": eos_count / max(count, 1),
            "max_tokens_rate": max_count / max(count, 1),
            "eval_answer_tokens": total_tokens,
            "samples": count,
        }
        if predictions_path is not None:
            path = Path(predictions_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return result
