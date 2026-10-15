# FedCA core experiment snapshot

Snapshot date: 2026-10-15 (Asia/Shanghai). Model: Qwen3-1.7B; seed: 42;
8 clients; 10 planned rounds; rank-8 q/v LoRA; `zsre_chat_eos_v2` generation protocol.
Each completed evaluation contains 8 x 500 test questions. EM and F1 are macro percentages.

| Experiment | Training status | Final EM | Final F1 | Evaluated saved rounds |
|---|---|---:|---:|---|
| `fedca` | Complete (10/10 rounds) | 15.850 | 22.846 | 2, 4, 6, 8 |
| `fedca_k1` | Complete (10/10 rounds) | 12.125 | 19.378 | 2, 4, 6, 8 |
| `srv_eta2` | Complete (10/10 rounds) | 11.850 | 19.217 | 2, 4, 6, 8 |
| `srv_eta3` | Complete (10/10 rounds) | 9.625 | 14.534 | 2, 4, 6, 8 |
| `fedexp` | Partial local snapshot (9/10 rounds) | - | - | - |
| `probe_only` | No local results yet | - | - | - |
| `ntd_only` | No local results yet | - | - | - |

`fedexp` has records/checkpoints through round 9, but no final adapter or generation
evaluation in this snapshot. `probe_only` and `ntd_only` have no local results yet.
The table reports observed files; it does not imply a running job has stopped.

## Included artifacts

- `*/adapter/*/fedca_rounds.jsonl`: per-round client diagnostics, selected server steps and probe scores.
- `*/adapter/*/training_config.json`: resolved configuration, when saved by training.
- `*/logs/*.log` and experiment logs: available training/evaluation logs, including the partial FedExP run.
- `*/evaluation/zsre/metrics.json` and `rounds/`: final and saved-round metrics, summaries and per-client predictions.
- `snapshot.json`: machine-readable completion status and final metrics.

Adapter weights and round checkpoint weights (`.pt`, `.pth`, `.bin`, `.safetensors`)
remain local and are ignored by Git. No checkpoints are deleted.
For an incomplete run, the base configuration and per-experiment overrides are in
[the suite runner](../../../../tools/run_fedca_suite.py) and
[the configuration](../../../../config.zsre_fedca.yaml).

## Reproduction and restart

From the repository root:

```powershell
python -X utf8 -u tools/run_fedca_suite.py --stage core --rounds-eval 2,4,6,8
python -X utf8 -u tools/run_fedca_suite.py --summary
```

A local restart skips training when its final `lora_weights.pt` exists, skips a final
evaluation when `metrics.json` exists, and skips completed saved-round evaluations.
A fresh clone has no adapter weights and must train before evaluating.
The runner cannot resume an incomplete training run; preserve partial artifacts
before arranging any retry. See [the method notes](../../../../README.fedca.md).

## Snapshot validation

- Python syntax parsed successfully for all 6 related source/test files.
- Validated 20 evaluation payloads and 80,000 prediction records:
  each has 8 clients x 500 unique samples, consistent per-client/macro metrics,
  and saved-round summaries consistent with their metric files.
- The summary command completed successfully.
- The CPU unit-test command was attempted in the current local Python environment,
  but could not import `datasets` (`ModuleNotFoundError`); unit tests were not rerun
  successfully as part of this upload.
