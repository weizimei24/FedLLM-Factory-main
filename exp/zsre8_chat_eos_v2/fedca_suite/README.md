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
| `fedexp` | Complete (10/10 rounds) | 9.400 | 16.273 | 2, 4, 6, 8 |
| `probe_only` | No local results yet | - | - | - |
| `ntd_only` | No local results yet | - | - | - |

`fedexp` was continued from the saved global adapter after 9 completed rounds;
only round 10 was trained. Final and round 2/4/6/8 generation evaluations completed
on 2026-10-15 at 11:34 (Asia/Shanghai). `probe_only` and `ntd_only` still have no local results.

## FedExP continuation

The original round-9 checkpoint was preserved; its SHA-256 matches the continuation
record in [`fedexp/resume_state.json`](fedexp/resume_state.json). The zero-based round
index was 9, with learning rate `0.0001 * 0.99**9 = 0.00009135172474836409`.
The original trainer creates a new AdamW optimizer each round and derives client
RNG/shuffle seeds from the client id and round index. The continuation uses those same paths.

| Completed round | EM | F1 |
|---|---:|---:|
| 2 | 4.025 | 10.169 |
| 4 | 7.425 | 14.069 |
| 6 | 7.975 | 14.735 |
| 8 | 8.825 | 15.744 |
| 10 | 9.400 | 16.273 |

## Included artifacts

- `*/adapter/*/fedca_rounds.jsonl`: per-round client diagnostics, selected server steps and probe scores.
- `*/adapter/*/training_config.json`: resolved training configuration.
- `*/logs/*.log` and experiment logs: training/evaluation logs, including separate FedExP continuation logs.
- `*/evaluation/zsre/metrics.json` and `rounds/`: final and saved-round metrics, summaries and per-client predictions.
- `fedexp/resume_state.json`: source checkpoint hash, configuration and continuation completion record.
- `snapshot.json`: machine-readable experiment completion status and final metrics.

Adapter and round checkpoint weights (`.pt`, `.pth`, `.bin`, `.safetensors`) remain
local and are ignored by Git. No checkpoints are deleted. Source checkpoints must
be available locally to resume; a fresh clone needs training before evaluation.

## Reproduction and restart

From the repository root:

```powershell
python -X utf8 -u tools/run_fedca_suite.py --stage core --rounds-eval 2,4,6,8
python -X utf8 -u tools/run_fedca_suite.py --summary
# Continue a partial run from a saved completed round:
python -X utf8 -u tools/run_fedca_suite.py --only fedexp --resume-training --rounds-eval 2,4,6,8
```

An ordinary local restart skips completed training/final evaluations and existing
saved-round evaluations. For continuation, use `--resume-training` with the original
configuration and experiment flags; only full-participation prototype runs with
every round saved are supported. See [the method notes](../../../../README.fedca.md).

## Snapshot validation

- Python syntax parsed successfully for all 8 related source/test files.
- Validated 25 evaluation payloads and 100,000 prediction records:
  each has 8 clients x 500 unique samples, consistent per-client/macro metrics,
  and saved-round summaries consistent with their metric files.
- All prediction prompts and targets exactly match the zsRE test data already
  present in this repository on `origin/main`.
- The summary command completed successfully.
- All 21 FedCA and continuation tests passed in the local `wzm` environment
  (`torch 2.10.0.dev20251013+cu128`, `transformers 4.57.6`, `peft 0.20.0`,
  `datasets 4.2.0`). The tiny CPU continuation test produces bitwise identical
  adapter weights to uninterrupted training.
