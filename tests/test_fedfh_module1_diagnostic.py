"""Small CPU correctness tests for the standalone FedFH direction diagnostic."""

import math
import unittest

import torch

from tools.fedfh_module1_diagnostic import answer_nll, project_a, projected_states, stable_random_basis


class TinyCausalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(9, 9)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.eye(9))

    def forward(self, input_ids, attention_mask, use_cache=False):
        return type("Output", (), {"logits": self.embedding(input_ids)})()


class FedFHModuleOneDiagnosticTests(unittest.TestCase):
    def test_projected_effects_are_complementary_and_rank_is_unchanged(self):
        torch.manual_seed(7)
        prefix = "base_model.model.model.layers.0.self_attn"
        state = {}
        geometry = {}
        for leaf in ("q_proj", "v_proj"):
            name = f"{prefix}.{leaf}"
            state[f"{name}.lora_A.default.weight"] = torch.randn(3, 40)
            state[f"{name}.lora_B.default.weight"] = torch.randn(24, 3)
        group = f"{prefix}.<qkv_input>"
        q, _ = torch.linalg.qr(torch.randn(40, 32), mode="reduced")
        geometry[group] = {"Q": q}
        variants = projected_states(state, geometry, top_k=32, seed=42, client_id=2)
        for key, a in state.items():
            if ".lora_B." in key:
                for variant in variants:
                    self.assertTrue(torch.equal(variants[variant][key], a))
                continue
            top = variants["Top-32"][key]
            rest = variants["Rest"][key]
            random = variants["Random-32"][key]
            b = state[key.replace("lora_A", "lora_B")]
            self.assertEqual(top.shape, a.shape)
            self.assertTrue(torch.allclose(top + rest, a, atol=1e-5))
            self.assertTrue(torch.allclose(b @ top + b @ rest, b @ a, atol=1e-4))
            self.assertTrue(torch.allclose(rest @ q, torch.zeros(3, 32), atol=1e-5))
            self.assertTrue(torch.allclose(random, project_a(a, stable_random_basis(40, 32, 42, 2, group))))
        self.assertTrue(torch.equal(state[f"{prefix}.q_proj.lora_A.default.weight"], variants["Original"][f"{prefix}.q_proj.lora_A.default.weight"]))

    def test_random_basis_is_reproducible_and_orthogonal(self):
        q1 = stable_random_basis(40, 32, 42, 0, "group")
        q2 = stable_random_basis(40, 32, 42, 0, "group")
        q3 = stable_random_basis(40, 32, 43, 0, "group")
        self.assertTrue(torch.equal(q1, q2))
        self.assertFalse(torch.equal(q1, q3))
        self.assertTrue(torch.allclose(q1.T @ q1, torch.eye(32), atol=1e-5))

    def test_nll_scores_only_gold_target_tokens_including_eos(self):
        model = TinyCausalModel()
        # First target is token 2, second target represents EOS (token 3).
        rows = [
            {"input_ids": [0, 1, 2, 3, 8], "attention_mask": [1, 1, 1, 1, 0], "labels": [-100, -100, 2, 3, -100]},
            {"input_ids": [4, 5, 6, 8, 8], "attention_mask": [1, 1, 1, 0, 0], "labels": [-100, -100, 6, -100, -100]},
        ]
        observed = answer_nll(model, rows, batch_size=2)
        logits = torch.eye(9)
        first = (-torch.log_softmax(logits[1], 0)[2] - torch.log_softmax(logits[2], 0)[3]) / 2
        second = -torch.log_softmax(logits[5], 0)[6]
        self.assertTrue(math.isclose(observed, float((first + second) / 2), abs_tol=1e-6))


if __name__ == "__main__":
    unittest.main()
