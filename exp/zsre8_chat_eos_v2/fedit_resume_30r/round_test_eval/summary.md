# FedIT post-aggregation global adapters on zsRE test

Each saved global adapter was evaluated on all eight clients’ 500 rephrased test questions under the original zsre_chat_eos_v2 greedy 32-token protocol. Round 10 is the pre-continuation reference.

| Completed round | EM | F1 | Answer NLL | EOS % | Max-token % | Mean generated tokens |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 9.45 | 16.55 | 1.8067 | 99.97 | 0.03 | 3.93 |
| 15 | 11.32 | 18.67 | 1.6851 | 100.00 | 0.00 | 3.98 |
| 20 | 13.05 | 20.46 | 1.6059 | 100.00 | 0.00 | 4.00 |
| 25 | 14.33 | 21.41 | 1.5649 | 100.00 | 0.00 | 4.09 |
| 30 | 15.65 | 22.70 | 1.5463 | 100.00 | 0.00 | 4.12 |

Per-client values and checkpoint/data hashes are in metrics.json and manifest.json.
