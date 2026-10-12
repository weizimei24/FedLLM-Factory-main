"""CPU checks of same-global, product-space FedIT update composition."""

import copy

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM

from tools.fedit_round_conflict_diagnostic import (
    RoundPatcher, SCALE, choose_pairs, condition_rank, product_change, result_summary, state_pairs,
)


def tiny_model():
    torch.manual_seed(9)
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=8, max_position_embeddings=64)
    return Qwen3ForCausalLM(config).float().eval()


def fake_state(model, seed):
    generator = torch.Generator().manual_seed(seed)
    state = {}
    for name, module in model.named_modules():
        if name.endswith(('q_proj', 'v_proj')):
            out_features, in_features = module.weight.shape
            prefix = f'base_model.model.{name}'
            state[f'{prefix}.lora_A.default.weight'] = torch.randn(8, in_features, generator=generator) * .05
            state[f'{prefix}.lora_B.default.weight'] = torch.randn(out_features, 8, generator=generator) * .05
    return state


def test_product_difference_and_rank():
    model = tiny_model()
    global_state, local = fake_state(model, 1), fake_state(model, 2)
    gp, lp = state_pairs(global_state), state_pairs(local)
    name = next(iter(gp))
    bg, ag = gp[name]
    bi, ai = lp[name]
    torch.testing.assert_close(product_change(gp[name], lp[name]),
                               SCALE * (bi @ ai - bg @ ag))
    assert condition_rank(gp[name], [lp[name]], {0: 1.0}) <= 16
    assert condition_rank(gp[name], [lp[name]], {}) == 0


def test_dense_merge_matches_peft_global_local_and_combination():
    base = tiny_model()
    global_state = fake_state(base, 1)
    local_states = [fake_state(base, 2), fake_state(base, 3)]
    model = copy.deepcopy(base)
    patcher = RoundPatcher(model, global_state, local_states)
    ids = torch.tensor([[3, 4, 5, 6]])
    original = model(input_ids=ids).logits
    config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32,
                        lora_dropout=0.0, target_modules=['q_proj', 'v_proj'])
    peft = get_peft_model(copy.deepcopy(base), config).eval()
    for coefficients, state in (({}, global_state), ({0: 1.0}, local_states[0])):
        patcher.apply(coefficients)
        peft.load_state_dict(state, strict=False)
        torch.testing.assert_close(model(input_ids=ids).logits,
                                   peft(input_ids=ids).logits, atol=1e-5, rtol=1e-5)
    name = next(iter(patcher.global_pairs))
    base_weight = patcher.base[name].float()
    global_pair = patcher.global_pairs[name]
    local_pairs = [pairs[name] for pairs in patcher.local_pairs]
    for coefficients in ({0: .125}, {0: .5, 1: .5}, {0: 1.0, 1: 1.0}):
        patcher.apply(coefficients)
        expected = base_weight + SCALE * (global_pair[0] @ global_pair[1])
        for i, c in coefficients.items():
            expected += c * product_change(global_pair, local_pairs[i])
        torch.testing.assert_close(patcher.modules[name].weight, expected, atol=1e-6, rtol=1e-6)
    patcher.restore_base()
    torch.testing.assert_close(model(input_ids=ids).logits, original)


def test_pair_selection_is_deterministic():
    s = [[0.0] * 8 for _ in range(8)]
    s[0][1], s[1][0] = -3.0, -2.0
    s[6][7], s[7][6] = 3.0, 2.0
    pairs = choose_pairs(s)
    assert pairs['most_adverse_cross_effect'] == [0, 1]
    assert pairs['most_helpful_cross_effect'] == [6, 7]


def test_result_summary_uses_eval_rows_and_update_columns():
    base = {str(j): {'nll': 2.0} for j in range(8)}
    singles = [[{'nll': 1.0 if i == j else 2.1} for j in range(8)] for i in range(8)]
    dilution = [{'1': {'nll': 1.0}, '0.125': {'nll': 1.8}} for _ in range(8)]
    joint = {'global': {str(j): {'nll': 2.0, 'em': 0.0} for j in range(8)}}
    summary = result_summary(base, singles, dilution, joint, {'example': [0, 1]})
    matrix = summary['interaction_matrix_rows_eval_j_columns_update_i']
    assert matrix[0][0] == 1.0 and abs(matrix[0][1] + .1) < 1e-10
    assert summary['diagonal']['positive'] == 8
    assert abs(summary['mean_nll_increase_from_full_to_one_eighth'] - .8) < 1e-10
