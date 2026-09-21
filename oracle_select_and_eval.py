import argparse, json
import torch
from peft import get_peft_model
from utils.cross_domain_qa import CrossDomainQAEvaluator, DOMAINS
from utils.model_utils import load_lora_config, load_model, load_tokenizer
from utils.options import build_parser
from utils.seed_utils import set_global_seed
from compute_response_matrix import load_client_delta, build_candidates, scaling  # reuse

def pick_oracle(rows):
    """Minimize worst-domain R_i, tie-break by mean R_i (both: lower is better)."""
    by_candidate = {}
    for r in rows:
        by_candidate.setdefault(r["candidate"], []).append(r["R_i"])
    scored = [(max(v), sum(v) / len(v), name) for name, v in by_candidate.items()]
    return min(scored)[2]

def main():
    parser = build_parser()
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--response_matrix", required=True)
    args, _ = parser.parse_known_args()
    args.suffix = f"exp/{args.suffix}"
    set_global_seed(args.seed, device=args.device, deterministic=args.deterministic)

    rows = json.load(open(args.response_matrix))
    best_name = pick_oracle(rows)
    print(f"Oracle-selected candidate: {best_name}")

    tokenizer = load_tokenizer(args)
    model = get_peft_model(load_model(args), load_lora_config(args))
    adapter_root = f"{args.suffix}/adapter/{args.alg}_{args.dataset}_{args.model}_{args.cn}c_{args.epoch}E_lr{args.lr}"

    client_dir = f"{adapter_root}/round_{args.round:03d}/clients"
    client_deltas = {i: load_client_delta(f"{client_dir}/client_{i}.pt", args) for i in range(args.cn)}
    a_keys = [k for k in client_deltas[0] if "lora_A" in k]
    global_weights = torch.load(f"{adapter_root}/round_{args.round:03d}/lora_weights.pt", map_location="cpu")
    candidates, _ = build_candidates(client_deltas, a_keys, scaling(args), args.lora_rank, {args.alg: global_weights})

    model.load_state_dict(candidates[best_name], strict=False)
    evaluator = CrossDomainQAEvaluator(args, reset_results=False)  # append, don't wipe existing FedIT/FedRot rows
    metrics = evaluator.evaluate(model, tokenizer, round_idx=args.round, method="oracle_response")
    print(metrics)

if __name__ == "__main__":
    main()