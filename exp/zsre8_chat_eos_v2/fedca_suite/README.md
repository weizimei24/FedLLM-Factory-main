# FedCA core experiment snapshot

Snapshot date: 2026-10-16 (Asia/Shanghai). Model: Qwen3-1.7B; seed: 42;
8 clients; 10 planned rounds; rank-8 q/v LoRA; `zsre_chat_eos_v2` generation protocol.
Each completed evaluation contains 8 x 500 test questions. EM and F1 are macro percentages.

| Experiment | Training status | Final EM | Final F1 | Evaluated saved rounds |
|---|---|---:|---:|---|
| `fedca` | Complete (10/10 rounds) | 15.850 | 22.846 | 2, 4, 6, 8 |
| `fedca_k1` | Complete (10/10 rounds) | 12.125 | 19.378 | 2, 4, 6, 8 |
| `srv_eta2` | Complete (10/10 rounds) | 11.850 | 19.217 | 2, 4, 6, 8 |
| `srv_eta3` | Complete (10/10 rounds) | 9.625 | 14.534 | 2, 4, 6, 8 |
| `fedexp` | Complete (10/10 rounds) | 9.400 | 16.273 | 2, 4, 6, 8 |
| `probe_only` | Complete (10/10 rounds) | 11.650 | 19.011 | 2, 4, 6, 8 |
| `ntd_only` | Complete (10/10 rounds) | 8.700 | 15.798 | 2, 4, 6, 8 |

`fedexp` was continued from the saved global adapter after 9 completed rounds;
only round 10 was trained. Final and round 2/4/6/8 generation evaluations completed
on 2026-10-15 at 11:34 (Asia/Shanghai). All seven core experiments now have complete
10-round training records and final plus round 2/4/6/8 generation evaluations.

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

## Ablation results

Both ablations have 10 completed rounds. `probe_only` disables the local not-true
KL constraint (`ca_lambda=0`) and retains probe-selected server steps; `ntd_only`
retains the constraint and fixes the server step to 1. Each evaluation uses the
same 8 x 500 zsRE test questions and generation protocol as the other methods.

| Completed round | probe_only EM | probe_only F1 | ntd_only EM | ntd_only F1 |
|---|---:|---:|---:|---:|
| 2 | 5.050 | 11.471 | 3.175 | 9.242 |
| 4 | 7.600 | 14.445 | 5.500 | 12.001 |
| 6 | 9.200 | 16.240 | 7.050 | 14.103 |
| 8 | 10.625 | 17.957 | 8.000 | 14.986 |
| 10 | 11.650 | 19.011 | 8.700 | 15.798 |

`probe_only` completed its tenth aggregation before the original command was
interrupted during the per-round loss/perplexity check. Its saved final adapter
was subsequently evaluated without retraining the completed rounds.

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
- Validated 35 evaluation payloads and 140,000 prediction records:
  each has 8 clients x 500 unique samples, consistent per-client/macro metrics,
  and saved-round summaries consistent with their metric files.
- All prediction prompts and targets exactly match the zsRE test data already
  present in this repository on `origin/main`.
- The summary command completed successfully.
- All 21 FedCA and continuation tests passed in the local `wzm` environment
  (`torch 2.10.0.dev20251013+cu128`, `transformers 4.57.6`, `peft 0.20.0`,
  `datasets 4.2.0`). The tiny CPU continuation test produces bitwise identical
  adapter weights to uninterrupted training.
