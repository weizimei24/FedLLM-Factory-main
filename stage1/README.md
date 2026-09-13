# Stage 1 StackExchange experiment

Run these commands from the repository root in the `wzm` environment. `-X utf8`
works around the original repository's encoding-free read of `utils/eval.yaml` on
Chinese Windows installations.

```powershell
conda activate wzm
python -X utf8 prepare_stackexchange.py

python -X utf8 run_stage1.py --method base --run-name paper_seed42
python -X utf8 run_stage1.py --method local --run-name paper_seed42
python -X utf8 run_stage1.py --method centralized --run-name paper_seed42
python -X utf8 run_stage1.py --method fedit --run-name paper_seed42
python -X utf8 run_stage1.py --method fedrotlora --run-name paper_seed42
```

The defaults are the paper protocol: 1000/200 examples per domain, 10 phases or
rounds, BF16, rank 16 LoRA on `q_proj` and `v_proj`, training batch size 1,
gradient accumulation 32, maximum sequence length 1152, and generation length
256. Outputs from all five commands are combined under
`results/stage1/paper_seed42`; `main_table.csv` is refreshed after each method.

For a functional check, `--smoke` selects the shortest 50/20 examples from each
already-fixed domain split, runs one phase or round, and uses 32 generation
tokens. Smoke results are diagnostics and must not be reported as paper results.
