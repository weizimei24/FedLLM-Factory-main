from types import SimpleNamespace

from alg.fedrotlora import _align_factors
from alg.ftbase import FTBaseServer
from stage1.model import lora_state_dict
from stage1.trainer import train_phase


def align_with_original_fedrot(local_lora: dict, global_lora: dict, round_index: int,
                               lam: float) -> dict:
    if round_index == 0:
        return local_lora
    aligned = {}
    for a_key in (key for key in local_lora if "lora_A" in key):
        b_key = a_key.replace("lora_A", "lora_B")
        aligned[a_key], aligned[b_key] = _align_factors(
            A_i=local_lora[a_key],
            B_i=local_lora[b_key],
            A_ref=global_lora[a_key],
            B_ref=global_lora[b_key],
            t=round_index,
            lam=lam,
        )
    return aligned


class FederatedRunner:
    """Thin data/training wrapper around the authors' aggregation functions."""

    def __init__(self, model, tokenizer, client_rows: dict[str, list[dict]], config, method: str):
        if method not in {"fedit", "fedrotlora"}:
            raise ValueError(method)
        sizes = {len(rows) for rows in client_rows.values()}
        if len(sizes) != 1:
            raise ValueError(f"All clients must have equal sizes for 1/6 FedAvg weights: {sizes}")
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.method = method
        self.global_lora = lora_state_dict(model)
        self.clients = [
            SimpleNamespace(id=index, domain=domain, rows=rows, dataset={"train": rows}, lora={})
            for index, (domain, rows) in enumerate(client_rows.items())
        ]
        self.sampled_clients = self.clients
        self.history = []

    def run(self):
        for round_index in range(self.config.rounds):
            round_history = []
            for client in self.clients:
                # This explicit restore is equivalent to FTBaseServer.local_run:
                # each client starts from the previous global adapter.
                self.model.load_state_dict(self.global_lora, strict=False)
                train_metrics = train_phase(
                    self.model,
                    client.rows,
                    self.tokenizer,
                    self.config,
                    phase_seed=self.config.seed + round_index * 100 + client.id,
                    label=f"{self.method} round={round_index} client={client.id} {client.domain}",
                )
                local = lora_state_dict(self.model)
                if self.method == "fedrotlora":
                    local = align_with_original_fedrot(
                        local, self.global_lora, round_index, self.config.fedrot_lambda
                    )
                client.lora = local
                round_history.append(train_metrics)

            # Directly invoke the original FedIT/FedRotLoRA base server FedAvg.
            # Equal client sizes make every coefficient exactly 1/6.
            FTBaseServer.aggregate(self)
            self.history.append(round_history)
        return self.history

