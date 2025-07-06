import math

import einx
import torch
from jaxtyping import Float, Int, Bool


class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None) -> None:
        super().__init__()
        weight = torch.empty(out_features, in_features, device=device, dtype=dtype)
        std = math.sqrt(2 / (in_features * out_features))
        torch.nn.init.trunc_normal_(weight, std=std, a=-std * 3, b=std * 3)
        self.weight = torch.nn.Parameter(weight)

    def forward(self, x: Float[torch.Tensor, "*prefix in_features"]) -> Float[torch.Tensor, "*prefix out_features"]:
        return einx.dot("... in_features, out_features in_features -> ... out_features", x, self.weight)


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None) -> None:
        super().__init__()
        weight = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(weight, std=1, a=-3, b=3)
        self.weight = torch.nn.Parameter(weight)

    def forward(self, x: Int[torch.Tensor, "*prefix seq_len"]) -> Float[torch.Tensor, "*prefix seq_len embedding_dim"]:
        return self.weight[x]


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None) -> None:
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.weight = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Float[torch.Tensor, "*prefix d_model"]) -> Float[torch.Tensor, "*prefix d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        x_squared_sum = einx.sum("... d_model -> ... 1", x.pow(2))
        norm = torch.sqrt(x_squared_sum / self.d_model + self.eps)
        result = x / norm * self.weight
        return result.to(in_dtype)


def silu(x: Float[torch.Tensor, "*dims"]) -> Float[torch.Tensor, "*dims"]:
    return x * torch.sigmoid(x)


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None) -> None:
        super().__init__()
        self.linear1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.linear2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.linear3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Float[torch.Tensor, "*prefix d_model"]) -> Float[torch.Tensor, "*prefix d_model"]:
        return self.linear2(silu(self.linear1(x)) * self.linear3(x))


class RoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None) -> None:
        super().__init__()
        assert d_k % 2 == 0, "RoPE dimension should be even"
        positions = einx.rearrange("seq_len -> seq_len 1", torch.arange(max_seq_len))
        dimensions = einx.rearrange("half_d_k -> 1 half_d_k", torch.arange(d_k // 2))
        thetas: Float[torch.Tensor, "seq_len half_d_k"] = positions / (theta ** (2 * dimensions / d_k))
        rotations: Float[torch.Tensor, "seq_len half_d_k 2 2"] = einx.rearrange(
            "seq_len half_d_k, seq_len half_d_k, seq_len half_d_k, seq_len half_d_k -> seq_len half_d_k (1 + 1) (1 + 1)",
            torch.cos(thetas),
            -torch.sin(thetas),
            torch.sin(thetas),
            torch.cos(thetas),
        )
        self.rotations = torch.nn.parameter.Buffer(rotations)

    def forward(
        self, x: Float[torch.Tensor, "*prefix seq_len d_k"], token_positions: Int[torch.Tensor, " seq_len"]
    ) -> Float[torch.Tensor, "*prefix seq_len d_k"]:
        x = einx.rearrange("... seq_len (half_d_k two) -> ... seq_len half_d_k two", x, two=2)
        rotations = self.rotations[token_positions]
        result = einx.dot(
            "... seq_len half_d_k two_in, seq_len half_d_k two_out two_in -> ... seq_len half_d_k two_out", x, rotations
        )
        return einx.rearrange("... seq_len half_d_k two -> ... seq_len (half_d_k two)", result)


def softmax(x: Float[torch.Tensor, "*dims"], dim: int = -1) -> Float[torch.Tensor, "*dims"]:
    x = x - x.max(dim=dim, keepdim=True).values
    exps = torch.exp(x)
    denominators = exps.sum(dim=dim, keepdim=True)
    return exps / denominators


class ScaledDotProductAttention(torch.nn.Module):
    def forward(
        self,
        Q: Float[torch.Tensor, "... seq_len_q d_k"],
        K: Float[torch.Tensor, "... seq_len_kv d_k"],
        V: Float[torch.Tensor, "... seq_len_kv d_v"],
        mask: Bool[torch.Tensor, "... seq_len_q seq_len_kv"] | None = None,
    ) -> Float[torch.Tensor, "... seq_len_q d_v"]:
        d_k = K.shape[-1]
        pre_softmax = einx.dot("... seq_len_q d_k, ... seq_len_kv d_k -> ... seq_len_q seq_len_kv", Q, K) / math.sqrt(
            d_k
        )
        if mask is not None:
            pre_softmax = einx.where(
                "... seq_len_q seq_len_kv, ... seq_len_q seq_len_kv, ", mask, pre_softmax, -torch.inf
            )
        attention = softmax(pre_softmax)
        return einx.dot("... seq_len_q seq_len_kv, ... seq_len_kv d_v -> ... seq_len_q d_v", attention, V)


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: RoPE | None = None, device=None, dtype=None) -> None:
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_v = d_model // num_heads
        self.rope = rope

        self.scaled_dot_product_attention = ScaledDotProductAttention()

        self.weights = torch.nn.ParameterDict(
            {
                "Q": torch.nn.Parameter(
                    torch.empty(self.num_heads * self.d_k, self.d_model, device=device, dtype=dtype)
                ),
                "K": torch.nn.Parameter(
                    torch.empty(self.num_heads * self.d_k, self.d_model, device=device, dtype=dtype)
                ),
                "V": torch.nn.Parameter(
                    torch.empty(self.num_heads * self.d_v, self.d_model, device=device, dtype=dtype)
                ),
                "output": torch.nn.Parameter(
                    torch.empty(self.d_model, self.num_heads * self.d_v, device=device, dtype=dtype)
                ),
            }
        )

        def init_matmul_weight(weight, std):
            torch.nn.init.trunc_normal_(weight, std=std, a=-std * 3, b=std * 3)

        with torch.no_grad():
            init_matmul_weight(self.weights["Q"], math.sqrt(2 / (self.num_heads * self.d_k * d_model)))
            init_matmul_weight(self.weights["K"], math.sqrt(2 / (self.num_heads * self.d_k * d_model)))
            init_matmul_weight(self.weights["V"], math.sqrt(2 / (self.num_heads * self.d_v * d_model)))
            init_matmul_weight(self.weights["output"], math.sqrt(2 / (self.d_model * self.num_heads * self.d_v)))

        # TODO: Ensure proper dimensions order

    def forward(self, x: Float[torch.Tensor, "... seq_len_q d_in"]) -> Float[torch.Tensor, "... seq_len_q d_k"]:
        seq_len = x.shape[-2]
        batch_size = x.shape[0]

        Q = einx.dot("d_model d_in, ... seq_len d_in -> ... seq_len d_model", self.weights["Q"], x)
        K = einx.dot("d_model d_in, ... seq_len d_in -> ... seq_len d_model", self.weights["K"], x)
        V = einx.dot("d_model d_in, ... seq_len d_in -> ... seq_len d_model", self.weights["V"], x)

        Q = einx.rearrange("... seq_len (n_heads d_inner) -> ... n_heads seq_len d_inner", Q, n_heads=self.num_heads)
        K = einx.rearrange("... seq_len (n_heads d_inner) -> ... n_heads seq_len d_inner", K, n_heads=self.num_heads)
        V = einx.rearrange("... seq_len (n_heads d_inner) -> ... n_heads seq_len d_inner", V, n_heads=self.num_heads)
        mask = torch.tril(torch.ones(batch_size, self.num_heads, seq_len, seq_len, device=self.device, dtype=torch.bool))

        if self.rope is not None:
            token_positions = torch.arange(seq_len, device=self.device)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        attended = self.scaled_dot_product_attention(Q, K, V, mask=mask)
        attended = einx.rearrange("... n_heads seq_len d_inner -> ... seq_len (n_heads d_inner)", attended)
        result = einx.dot("d_in d_model, ... seq_len d_model -> ... seq_len d_in", self.weights["output"], attended)
        return result


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: RoPE | None = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.rms_norm1 = RMSNorm(d_model)
        self.mhsa = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, rope=rope, device=device, dtype=dtype)
        self.rms_norm2 = RMSNorm(d_model)
        self.swiglu = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(
        self, x: Float[torch.Tensor, "*prefix seq_len d_model"]
    ) -> Float[torch.Tensor, "*prefix seq_len d_model"]:
        x = x + self.mhsa(self.rms_norm1(x))
        x = x + self.swiglu(self.rms_norm2(x))
        return x


class Transformer(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: RoPE | None = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.emb = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)
        self.transformer_blocks = torch.nn.Sequential(*[
            TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, rope=rope, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.final_linear = Linear(d_model, vocab_size, device=device, dtype=dtype)

    
    def forward(self, x: Int[torch.Tensor, "*prefix seq_dim"]) -> Float[torch.Tensor, "*prefix seq_dim vocab_size"]:
        x = self.emb(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        x = self.final_linear(x)
        return x
    



def calc():
    return num_layers * (
        (8 * d_model + 2 * seq + 2) * bs * seq * d_model + 
        (6 * d_model + 2) * bs * seq * d_ff +
        7 * bs * seq * d_model + 7 * bs * seq * d_model
    ) + \
    7 * bs * seq * d_model + \
    2 * bs * seq * d_model * vocab_size
    