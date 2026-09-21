"""Build a candidate bank and measure geometry vs. real domain response.

Usage:
    python -X utf8 compute_response_matrix.py --config config.cross_domain_fedit.yaml --round 5
"""
import argparse, itertools, json, os
import torch
from peft import get_peft_model
from utils.model_utils import load_lora_config, load_model, load_tokenizer
from utils.options import build_parser
from utils.seed_utils import set_global_seed
from utils.response_probe import DOMAINS, probe_loss_all_domains

def scaling(args):
    return args.lora_alpha / args.lora_rank

def load_client_delta(path, args):
    """Return {a_key: A_i, b_key: B_i} for one saved client."""
    return torch.load(path, map_location="cpu")

def materialize_dense(lora_dict, a_keys, s):
    """{a_key: dense ΔW = s * B @ A} for every lora_A/_B pair. Keyed by the
    original a_key itself, mirroring alg/fedrotlora.py's own pairing style."""
    out = {}
    for a_key in a_keys:
        b_key = a_key.replace("lora_A", "lora_B")
        A, B = lora_dict[a_key].float(), lora_dict[b_key].float()
        out[a_key] = s * (B @ A)
    return out

def flat_cosine(dense_1, dense_2):
    keys = sorted(dense_1)
    v1 = torch.cat([dense_1[k].flatten() for k in keys])
    v2 = torch.cat([dense_2[k].flatten() for k in keys])
    return torch.nn.functional.cosine_similarity(v1, v2, dim=0).item()

def refactorize_to_rank(dense_dict, r, model_scaling):
    """SVD each layer's dense ΔW back to rank r, pre-dividing A by
    model_scaling so that PEFT's own forward-time `scaling * B @ A`
    reproduces `dense_dict` (up to truncation error), not `scaling * dense_dict`."""
    patch = {}
    for a_key, dense in dense_dict.items():
        b_key = a_key.replace("lora_A", "lora_B")
        U, S, Vh = torch.linalg.svd(dense, full_matrices=False)
        U_r, S_r, Vh_r = U[:, :r], S[:r], Vh[:r, :]
        B_new = U_r * S_r.sqrt()
        A_new = (S_r.sqrt().unsqueeze(1) * Vh_r) / model_scaling
        patch[a_key] = A_new.half()
        patch[b_key] = B_new.half()
    return patch

def build_candidates(client_deltas, a_keys, s, r, extra_globals):
    dense_per_client = {cid: materialize_dense(lora, a_keys, s) for cid, lora in client_deltas.items()}
    candidates = {f"local_{cid}": client_deltas[cid] for cid in client_deltas}
    candidates.update(extra_globals)  # e.g. saved FedIT/FedRot global adapters for this round

    layers = list(next(iter(dense_per_client.values())).keys())
    mean_dense = {l: sum(dense_per_client[c][l] for c in dense_per_client) / len(dense_per_client) for l in layers}
    candidates["naive_mean"] = refactorize_to_rank(mean_dense, r, s)

    # a couple of simple weighted pairwise combinations (equal weight, all pairs)
    for c1, c2 in itertools.combinations(dense_per_client.keys(), 2):
        pair_dense = {l: 0.5 * dense_per_client[c1][l] + 0.5 * dense_per_client[c2][l] for l in layers}
        candidates[f"pair_{c1}_{c2}"] = refactorize_to_rank(pair_dense, r, s)

    return candidates, dense_per_client

def main():
    parser = build_parser()
    parser.add_argument("--round", type=int, required=True)
    args, _ = parser.parse_known_args()
    args.suffix = f"exp/{args.suffix}"
    set_global_seed(args.seed, device=args.device, deterministic=args.deterministic)

    tokenizer = load_tokenizer(args)
    model = get_peft_model(load_model(args), load_lora_config(args))
    s = scaling(args)

    adapter_root = f"{args.suffix}/adapter/{args.alg}_{args.dataset}_{args.model}_{args.cn}c_{args.epoch}E_lr{args.lr}"
    client_dir = f"{adapter_root}/round_{args.round:03d}/clients"
    client_deltas = {
        i: load_client_delta(f"{client_dir}/client_{i}.pt", args) for i in range(args.cn)
    }
    a_keys = [k for k in client_deltas[0] if "lora_A" in k]

    global_weights = torch.load(f"{adapter_root}/round_{args.round:03d}/lora_weights.pt", map_location="cpu")
    candidates, dense_per_client = build_candidates(
        client_deltas, a_keys, s, args.lora_rank, {args.alg: global_weights}
    )

    # W (pre-aggregation global reference for this round) baseline probe loss
    prev_root = f"{adapter_root}/round_{args.round-1:03d}/lora_weights.pt" if args.round > 0 else None
    if prev_root and os.path.exists(prev_root):
        model.load_state_dict(torch.load(prev_root, map_location="cpu"), strict=False)
    baseline = probe_loss_all_domains(model, tokenizer, f"dataset/{args.dataset}")

    rows = []
    for name, patch in candidates.items():
        model.load_state_dict(patch, strict=False)
        response = probe_loss_all_domains(model, tokenizer, f"dataset/{args.dataset}")
        for i, domain_i in enumerate(DOMAINS):
            row = {
                "round": args.round, "candidate": name, "domain": domain_i,
                "R_i": response[domain_i] - baseline[domain_i],
            }
            if name.startswith("local_"):
                cid = int(name.split("_")[1])
                row["cos_with_own_domain"] = flat_cosine(dense_per_client[cid], dense_per_client[i]) if cid != i else None
            rows.append(row)

    out_path = f"{args.suffix}/response_matrix_round{args.round:03d}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(rows, open(out_path, "w"), indent=2)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()