from typing import Literal

import einx
import torch
from jaxtyping import Float, Int


class CrossEntropyLoss(torch.nn.Module):
    def forward(
        self, logits: Float[torch.Tensor, "*prefix vocab_size"], targets: Int[torch.Tensor, "*prefix"], reduction: Literal["mean", "none"] = "mean",
    ) -> Float[torch.Tensor, "1"]:
        logits = logits - logits.max(dim=-1, keepdim=True).values
        numerator = einx.get_at("... [vocab_size], ... -> ...", logits, targets)
        denumerator = einx.sum("... [vocab_size]", logits.exp()).log()
        loss = -numerator + denumerator
        match reduction:
            case "mean":
                return loss.mean()
            case "none":
                return loss
        raise ValueError(f"Unknown reduction: {reduction}")
    
def perplexity(logits: Float[torch.Tensor, "*prefix seq_len vocab_size"], targets: Int[torch.Tensor, "*prefix seq_len"]) -> Float[torch.Tensor, "*prefix"]:
    loss: Float[torch.Tensor, "*prefix seq_len"] = CrossEntropyLoss()(logits, targets, reduction="none")
    print(loss.shape)
    return torch.exp(loss.mean(dim=-1))
