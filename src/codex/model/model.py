import os
import inspect
import torch.nn as nn
import torch.nn.functional as F
import torch
import hydra
import tiktoken
import time
import math
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchtune.modules import RotaryPositionalEmbeddings
from .utils import MOEManager

MANAGER = MOEManager()


# Adapted from https://github.com/DeepSeek-ai/DeepSeek-V3/blob/main/inference/model.py#L294
def precompute_freqs_cis(args: DeepSeekV3ModelArgs) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args (DeepSeekV3ModelArgs): Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(
        num_rotations: float, dim: int, base: float, max_seq_len: int
    ) -> float:
        """
        Computes the correction dimension for a given number of rotations in the rotary positional embedding.

        Args:
            num_rotations (float): Number of rotations to compute the correction for.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            float: The correction dimension based on the input parameters.
        """
        return (
            dim
            * math.log(max_seq_len / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    def find_correction_range(
        low_rot: float, high_rot: float, dim: int, base: float, max_seq_len: int
    ) -> tuple[int, int]:
        """
        Computes the range of correction dimensions for rotary positional embeddings.

        Args:
            low_rot (float): Lower bound for the number of rotations.
            high_rot (float): Upper bound for the number of rotations.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min: float, max: float, dim: int) -> torch.Tensor:
        """
        Computes a linear ramp function used to smooth values between a minimum and maximum range.

        Args:
            min (float): Minimum value for the ramp function.
            max (float): Maximum value for the ramp function.
            dim (int): Dimensionality of the ramp tensor.

        Returns:
            torch.Tensor: A tensor of shape (dim,) with values linearly interpolated between 0 and 1,
                clamped to the range [0, 1].
        """
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    # Basic RoPE frequency calculation
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    # YaRN scaling for extended context. YaRN is used to extend the context length after pre-training.
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, base, args.original_seq_len
        )
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    # Create position indices
    t = torch.arange(seqlen)

    # Outer product: [positions] × [frequencies]
    freqs = torch.outer(t, freqs)

    # Convert to complex exponentials: e^(i*freq*pos)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.RESIDUAL_SCALE = 1

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.rotary_pos_emb = RotaryPositionalEmbeddings(config.n_embd // config.n_head)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, C // self.n_head)
        k = k.view(B, T, self.n_head, C // self.n_head)

        q = self.rotary_pos_emb(q)
        k = self.rotary_pos_emb(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        q = self.rotary_pos_emb(q)
        k = self.rotary_pos_emb(k)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(attn)


# Adapted from https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/deepseek_v3/model/model.py#L147 to include gated attetion
class MultiHeadLatentAttention(nn.Module):
    """
    Multi-head attention (MLA) module.

    """

    def __init__(self, model_args):
        super().__init__()
        self.dim = model_args.dim
        self.n_heads = model_args.n_heads
        self.q_lora_rank = model_args.q_lora_rank
        self.kv_lora_rank = model_args.kv_lora_rank
        self.qk_nope_head_dim = model_args.qk_nope_head_dim
        self.qk_rope_head_dim = model_args.qk_rope_head_dim
        self.qk_head_dim = model_args.qk_nope_head_dim + model_args.qk_rope_head_dim
        self.v_head_dim = model_args.v_head_dim

        if self.q_lora_rank == 0:
            self.wq = nn.Linear(self.dim, self.n_heads * self.qk_head_dim, bias=False)
        else:
            self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False)
            self.q_norm = nn.RMSNorm(self.q_lora_rank, eps=model_args.norm_eps)
            self.wq_b = nn.Linear(
                self.q_lora_rank, self.n_heads * self.qk_head_dim, bias=False
            )
        self.wkv_a = nn.Linear(
            self.dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=False
        )
        self.kv_norm = nn.RMSNorm(self.kv_lora_rank, eps=model_args.norm_eps)
        self.wkv_b = nn.Linear(
            self.kv_lora_rank,
            self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim, bias=False)
        self.softmax_scale = self.qk_head_dim**-0.5

        # gate parameters for gatted attention
        self.wg = nn.Linear(self.dim, self.n_heads * self.v_head_dim, bias=False)

        if model_args.max_seq_len > model_args.original_seq_len:
            mscale = 0.1 * model_args.mscale * math.log(model_args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        self.use_flex_attn = model_args.use_flex_attn
        if self.use_flex_attn:
            self.inner_attention = FlexAttentionWrapper()
        else:
            self.inner_attention = ScaledDotProductAttentionWrapper()

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
    ):
        """
        Forward pass for the Multi-Head Latent Attention (MLA) Layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        bsz, seqlen, _ = x.size()

        # Query projection
        if self.q_lora_rank == 0:
            q = self.wq(x)  # (bsz, seqlen, n_heads * qk_head_dim)
        else:
            q = self.wq_a(x)
            q = self.wq_b(self.q_norm(q))
        # Use -1 instead of `n_heads` (or `n_kv_heads`) to infer the actual
        # local heads from sizes of q and kv as TP may have sharded them after
        # the above linear ops.
        q = q.view(bsz, seqlen, -1, self.qk_head_dim)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        q = torch.cat([q_nope, q_pe], dim=-1)  # (bsz, seqlen, n_heads, qk_head_dim)

        # Key-value projection
        kv = self.wkv_a(x)  # (bsz, seqlen, kv_lora_rank + qk_rope_head_dim)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        k_pe = apply_rotary_emb(
            k_pe.unsqueeze(2), freqs_cis
        )  # (bsz, seqlen, 1, qk_rope_head_dim)

        kv = self.wkv_b(
            self.kv_norm(kv)
        )  # (bsz, seqlen, n_heads * (qk_nope_head_dim + v_head_dim))
        kv = kv.view(bsz, seqlen, -1, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = torch.cat(
            [k_nope, k_pe.expand(-1, -1, self.n_heads, -1)], dim=-1
        )  # (bsz, seqlen, n_heads, qk_head_dim)

        # condition gate parameters with input from the residual stream
        g = self.wg(x)  # (bsz, seqlen, n_heads, v_head_dim)

        q = q.transpose(1, 2)  # (bsz, n_heads, seqlen, qk_head_dim)
        k = k.transpose(1, 2)  # (bsz, n_heads, seqlen, qk_head_dim)
        v = v.transpose(1, 2)  # (bsz, n_heads, seqlen, v_head_dim)

        if self.use_flex_attn:
            assert isinstance(attention_masks, BlockMask)
            output = self.inner_attention(
                q, k, v, block_mask=attention_masks, scale=self.softmax_scale
            )
        else:
            assert attention_masks is None
            output = self.inner_attention(q, k, v, scale=self.softmax_scale)

        # Reshape and project output
        output = output.transpose(
            1, 2
        ).contiguous()  # (bsz, seqlen, n_heads, v_head_dim)

        # apply gate to output
        output = (
            nn.sigmoid(g) * output
        )  # another implementation used output = output * nn.sigmoid(g) instead, just noting

        output = output.view(bsz, seqlen, -1)  # (bsz, seqlen, n_heads * v_head_dim)

        return self.wo(output)  # (bsz, seqlen, dim)

    def init_weights(self, init_std: float):
        linear_list = [
            self.wkv_a,
            self.wkv_b,
        ]
        if self.q_lora_rank > 0:
            linear_list.extend([self.wq_a, self.wq_b])
        else:
            linear_list.append(self.wq)

        for linear in linear_list:
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

        self.kv_norm.reset_parameters()
        if self.q_lora_rank > 0:
            self.q_norm.reset_parameters()


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        d_ff = 3 * config.n_embd
        assert d_ff % 2 == 0
        self.c_fc = nn.Linear(config.n_embd, d_ff)
        self.c_proj = nn.Linear(d_ff // 2, config.n_embd)
        self.c_proj.RESIDUAL_SCALE = 1
        self.silu = nn.SiLU()

    def forward(self, x):
        """
        swiglu based on flash attention implementation
        """
        y = self.c_fc(x)
        y, gate = y.chunk(2, dim=-1)
        return self.c_proj(self.silu(y) * gate)


class MLPExperts(nn.Module):

    def __init__(self, config):
        super().__init__()

        d_ff = 3 * config.n_embd
        assert d_ff % 2 == 0

        self.config = config
        self.c_fc = nn.Parameter(torch.empty(config.n_exp, config.n_embd, d_ff))
        # does this get added as part of the model in the accelerator even when it is not in use?
        self.bias_fc = (
            nn.Parameter(torch.empty(config.n_exp, 1, d_ff))
            if self.config.bias
            else None
        )
        self.c_proj = nn.Parameter(torch.empty(config.n_exp, d_ff // 2, config.n_embd))
        self.bias_proj = (
            nn.Parameter(torch.empty(config.n_exp, 1, config.n_embd))
            if self.config.bias
            else None
        )
        self.silu = nn.SiLU()
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = torch.bmm(x, self.c_fc)
        if self.config.bias:
            x = x + self.bias_fc
        x, gate = x.chunk(2, dim=-1)
        x = self.silu(x) * gate
        x = torch.bmm(x, self.c_proj)
        if self.config.bias:
            x = x + self.bias_proj
        x = self.dropout(x)
        return x


class Router(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config
        assert (
            config.top_k >= 1 and config.top_k <= config.n_exp
        ), f"top_k must be less than or equal to n_exp"

        self.router = nn.Linear(config.n_embd, config.n_exp, bias=False)
        self.router_noise = (
            nn.Linear(config.n_embd, config.n_exp, bias=False)
            if config.use_router_noise
            else None
        )

    def forward(self, x):
        B, T, _ = x.size()
        num_tokens = B * T

        router_logits = self.router(x)

        if self.config.use_router_noise:
            noise_weights = F.softplus(self.router_noise(x))
            noise = noise_weights * torch.rand_like(noise_weights)
            router_logits += noise

        if self.config.use_router_z_loss:
            z_loss = self.compute_router_z_loss(router_logits)
            MANAGER.add_router_z_loss(z_loss)

        topk_logits, topk_indices = router_logits.topk(
            self.config.top_k, dim=-1
        )  # B, T, top_k
        router_probs = torch.full_like(router_logits, float("-inf"))  # [B, C, n_exp]
        router_probs.scatter_(-1, topk_indices, topk_logits)
        router_probs = F.softmax(router_probs, dim=-1)

        if self.config.use_aux_loss:
            aux_loss = self.compute_aux_loss(router_probs, topk_indices)
            MANAGER.add_aux_loss(aux_loss)

        # compute expert capacity
        exp_cap = (
            (self.config.top_k * B * T) / self.config.n_exp
        ) * self.config.capacity_factor
        exp_cap += exp_cap % 2
        exp_cap = int(exp_cap)

        exp_mask = F.one_hot(
            topk_indices, num_classes=self.config.n_exp
        )  # [B, C, K, n_exp]
        exp_mask = exp_mask.view(
            num_tokens, self.config.top_k, self.config.n_exp
        )  # [B * C, K, n_exp]
        exp_mask = exp_mask.permute(1, 0, 2)  # [K, B * C, n_exp]

        exp_rank = exp_mask.reshape(
            self.config.top_k * num_tokens, self.config.n_exp
        )  # [K * B * C, n_exp]
        exp_rank = (
            torch.cumsum(exp_rank, dim=0) - 1
        )  # cumsum of expert selections [K * B * C, n_exp]
        exp_rank = exp_rank.reshape(self.config.top_k, num_tokens, self.config.n_exp)

        exp_mask *= torch.lt(exp_rank, exp_cap)  # [K, B * C, n_exp]
        used_cap = torch.sum(exp_mask, dim=(0, 1))

        # matrix storing token position in batch of corresponding expert
        exp_rank = torch.sum(exp_mask * exp_rank, dim=-1)  # [K, B * C]

        # mask probabilities to only include selected experts
        router_probs = router_probs.view(num_tokens, self.config.n_exp)[
            None, :
        ]  # [1, B * C, n_exp]
        exp_weights = exp_mask * router_probs  # [K, B * C, n_exp]

        # position of each token within the capacity of the selected expert
        exp_rank_sc = F.one_hot(
            exp_rank, num_classes=exp_cap
        )  # [K, B * C, exp_capacity]

        # weight of selected expert for each token at position the capacity of that expert
        cb_weight = torch.sum(
            exp_weights.unsqueeze(3) * exp_rank_sc.unsqueeze(2), dim=0
        )  # [B * C, n_exp, exp_capacity]
        sec_mask = cb_weight.bool()  # binary mask of selected experts for each token

        # reshape tokens into batches for each expert, return both weights and batches
        # [n_exp, exp_capacity, B * C] * [B * C, d] -> [n_exp, exp_capacity, n_embd]
        x = x.view(num_tokens, self.config.n_embd)

        return used_cap, cb_weight, sec_mask

    def compute_aux_loss(self, expert_probs: torch.Tensor, indices: torch.Tensor):
        """
        Computes Switch Transformer auxiliary loss (https://arxiv.org/abs/2101.03961)
        See equations (4)-(6) on page 7
        """

        # equation (5): compute ratio of tokens allocated to each expert
        # total number of tokens is defined as total tokens in batch * k
        # (k = 1) for the Switch Transformer
        with torch.no_grad():
            one_hot_indices = F.one_hot(
                indices, num_classes=self.config.n_exp
            )  # [B, T, k, n_exp]
            one_hot_indices = torch.sum(
                one_hot_indices.float(), dim=2
            )  # [B, T, n_exp] (sum over k dimension)
            tokens_per_expert = torch.mean(one_hot_indices.float(), dim=(0, 1))

        # equation (6): compute ratio of router probability allocated to each expert
        prob_per_expert = torch.mean(expert_probs.float(), dim=(0, 1))

        # equation (4): take a scaled dot product between prob/token allocation vectors
        # multiply the result by the number of experts
        return self.config.n_exp * torch.sum(prob_per_expert * tokens_per_expert)

    def compute_router_z_loss(self, logits: torch.Tensor):
        """
        Computes ST-MoE router z loss (https://arxiv.org/abs/2202.08906)
        See equation (5) on page 7
        """

        # exponentiate logits, sum logits of each expert, take log, and square
        # code below is the same as:
        # > z_loss = torch.exp(logits)
        # > z_loss = torch.sum(z_loss, dim=-1)
        # > z_loss = torch.log(z_loss) ** 2.0
        z_loss = torch.logsumexp(logits, dim=-1) ** 2.0  # [B, T, n_exp]

        # sum over all tokens and divide by total number of tokens
        return torch.mean(z_loss)


class Gate(nn.Module):

    def __init__(self, config):
        """
        top_k
        n_exp
        n_groups
        n_limited_groups
        func -> sigmoid, softmax
        bias
        """
        self.w_gate = nn.Linear(n_embd, n_exp, bias=bias)
        pass

    def forward(self, x):
        """
        x -> B*T, n_embd
        """
        gate_scores = self.w_gate(x)  # B*T, n_exp

        if self.func == "sigmoid":
            gate_scores = nn.sigmoid(gate, dim=-1)
        elif self.func == "softmax":
            gate_scores = nn.softmax(gate, dim=-1)

        original_scores = gate_scores

        if self.n_groups > 1:
            assert self.n_groups % n_exp == 0, "n_groups is indivisible by n_exp"
            n_group_exp = self.n_groups // n_exp
            assert (
                n_group_exp > self.top_k
            ), "number of experts in a group should be greater than topk"

            gate_scores = gate_scores.view(
                x.size(0), self.n_groups, -1
            )  # B*S, n_groups, n_group_exp

            # maximum score for each group
            if not bias:
                group_scores = gate_scores.amax(dim=-1)  # B*S, n_groups
            else:
                group_scores = gate_scores.topk(2, dim=-1)[0].sum(-1)

            # select top groups i.e maximum number of groups
            top_group_indices = group_scores.topk(self.n_limited_groups, dim=-1)[
                1
            ]  # B*S, n_limited_groups

            # create mask
            mask = gate_scores.new_ones(x.size(0), n_groups, dtype=bool).scatter_(
                1, top_group_indices, False
            )

            gate_scores = gate_scores.masked_fill_(
                mask.unsqueeze(-1), float("-inf")
            ).flatten(1)

        topk_indices = torch.topk(gate_scores, self.topk, dim=-1)[1]
        weights = gate_scores.gather(1, topk_indices)
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale

        # group tokens together by expert indices from 0 to num_experts and pass that to experts forward
        num_tokens_per_expert = torch.histc(
            topkindices.view(-1),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )

        return weights.type_as(x), topkindices, num_tokens_per_expert


class MOE(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.router = Gate(config)

        self.experts = MLPExperts(config)

    def forward(self, x):
        B, T, n_embd = x.size()
        num_tokens = B * T

        used_capacity, exp_weight, exp_mask = self.router(x)

        x = x.view(num_tokens, n_embd)
        exp_batches = exp_mask.permute(1, 2, 0).type_as(x) @ x

        exp_out = self.experts(exp_batches)

        exp_weight = exp_weight.view(num_tokens, -1)
        exp_out = exp_out.view(-1, self.config.n_embd)
        output = exp_weight @ exp_out
        output = output.view(B, T, self.config.n_embd)
        return output


class MoE(nn.Module):
    def __init__(self, moe_args: MoEArgs, dim: int, hidden_dim: int):
        super().__init__()

        num_experts = moe_args.num_experts
        from torchtitan.models.moe import GroupedExperts, FeedForward

        self.experts = GroupedExperts(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            use_grouped_mm=moe_args.use_grouped_mm,
        )
        self.router = Gate(
            dim=dim,
            num_experts=num_experts,
            top_k=moe_args.top_k,
            score_func=moe_args.score_func,
            route_norm=moe_args.route_norm,
            route_scale=moe_args.route_scale,
        )

        self.shared_experts = (
            FeedForward(dim=dim, hidden_dim=hidden_dim * moe_args.num_shared_experts)
            if moe_args.num_shared_experts > 0
            else None
        )

        self.score_before_experts = moe_args.score_before_experts

        # define fields for auxiliary-loss-free load balancing (https://arxiv.org/abs/2408.15664)
        # NOTE: tokens_per_expert is accumulated in the model forward pass.
        #       expert_bias is updated outside the model in an optimizer step pre hook
        #       to work with gradient accumulation.
        self.load_balance_coeff = moe_args.load_balance_coeff
        if self.load_balance_coeff is not None:
            assert self.load_balance_coeff > 0.0
            self.register_buffer(
                "expert_bias",
                torch.zeros(num_experts, dtype=torch.float32),
                persistent=True,
            )
        else:
            self.expert_bias = None
        # tokens_per_expert will be used to track expert usage and to update the expert bias for load balancing
        self.register_buffer(
            "tokens_per_expert",
            torch.zeros(num_experts, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs, slen, dim)``.

        Returns:
            out (torch.Tensor): Output tensor with shape ``(bs, slen, dim)``.
        """
        bs, slen, dim = x.shape
        x = x.view(-1, dim)

        # top_scores and selected_experts_indices shape (bs*slen*top_k,)
        # num_tokens_per_expert shape (num_experts,)
        (
            top_scores,
            selected_experts_indices,
            num_tokens_per_expert,
        ) = self.router(x, self.expert_bias)

        # tokens_per_expert will be used to update the expert bias for load balancing.
        # and also to count the expert usage
        # TODO: Activation Checkpointing has the side effect of double counting tokens_per_expert --
        #       first in the forward pass, and then in the backward pass. However, this has no
        #       effect on the expert bias update thanks to the torch.sign() operator.
        with torch.no_grad():
            self.tokens_per_expert.add_(num_tokens_per_expert)

        # top_scores and token_indices_experts_sorted shape (bs*slen*top_k,)
        # num_tokens_per_expert shape (num_experts,)
        # NOTE: the reason we need to compute num_tokens_per_expert again is:
        #       1st computation in router is to update self.tokens_per_expert
        #       which would be the same across all TP ranks.
        #       2nd computation in reorderer is for the actual routing and experts computation
        #       which would be sharded over TP ranks if expert_tensor_parallel_degree==1.
        #       If tensor_paralllel_degree == expert_tensor_parallel_degree, they agree.

        # shape (bs*slen*top_k, dim)
        token_indices_experts_sorted = token_indices_experts_sorted.reshape(
            -1, 1
        ).expand(-1, dim)

        # shape (bs*slen*top_k, dim)
        routed_input = torch.gather(x, dim=0, index=token_indices_experts_sorted)

        if self.score_before_experts:
            routed_input = (
                routed_input.to(torch.float32)
                * top_scores_experts_sorted.reshape(-1, 1)
            ).to(x.dtype)

        # shape (bs*slen*top_k, dim)
        routed_output = self.experts(routed_input, num_tokens_per_expert)

        if not self.score_before_experts:
            routed_output = (
                routed_output.to(torch.float32)
                * top_scores_experts_sorted.reshape(-1, 1)
            ).to(x.dtype)

        # shared expert
        if self.shared_experts is not None:
            out = self.shared_experts(x)
        else:
            out = torch.zeros_like(x)

        out = out.scatter_add(
            dim=0, index=token_indices_experts_sorted, src=routed_output
        )
        out = out.reshape(bs, slen, dim)
        return out

    def init_weights(
        self,
        init_std: float,
        buffer_device: torch.device,
    ):
        self.experts.init_weights(init_std)
        self.router.init_weights(init_std)
        if self.shared_experts is not None:
            self.shared_experts.init_weights(init_std)

        with torch.device(buffer_device):
            self.tokens_per_expert = torch.zeros(
                self.experts.num_experts, dtype=torch.float32
            )
            if self.load_balance_coeff is not None:
                self.expert_bias = torch.zeros(
                    self.experts.num_experts, dtype=torch.float32
                )


class Block(nn.Module):
    def __init__(self, config, use_moe=False, linear_attention=True):
        super().__init__()
        from fla.layers import GatedDeltaNet

        self.ln_1 = nn.RMSNorm(config.n_embd)
        if linear_attention:
            self.attn = GatedDeltaNet(
                hidden_size=config.n_embd, num_heads=config.n_head, mode="chunk"
            )
        else:
            self.attn = MultiHeadLatentAttention(config)
        self.ln_2 = nn.RMSNorm(config.n_embd)
        if use_moe:
            self.mlp = MoE(config)
        else:
            self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Codex(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                # use moe for every pth layer if use_moe is true
                # use linear attention for every layer and standard attention for every gth layer
                h=nn.ModuleList(
                    [
                        Block(
                            config,
                            use_moe=((i % config.p) == 0 and config.use_moe),
                            linear_attention=(i % config.g) != 0,
                        )
                        for i in range(config.n_layers)
                    ]
                ),
                ln_f=nn.RMSNorm(config.n_embd),
            )
        )

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self.__init_weights)

    def __init_weights(self, module):

        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "RESIDUAL_SCALE"):
                std *= (2 * self.config.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        elif isinstance(module, MLPExperts):
            torch.nn.init.normal_(module.c_fc, mean=0.0, std=0.02)
            torch.nn.init.normal_(module.c_proj, mean=0.0, std=0.02)
            if module.bias_fc is not None:
                torch.nn.init.zeros_(module.bias_fc)
            if module.bias_proj is not None:
                torch.nn.init.zeros_(module.bias_proj)

    def configure_optimizer(self, device_type=None):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        non_decay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {
                "params": decay_params,
                "weight_decay": self.config.optimizer.weight_decay,
            },
            {"params": non_decay_params, "weight_decay": 0.0},
        ]

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.config.optimizer.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=use_fused,
        )

        return optimizer

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert (
            T <= self.config.block_size
        ), f"Sequence length {T} is longer than the block size {self.config.block_size}"

        tok_emb = self.transformer.wte(idx)
        x = tok_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

            if self.config.use_moe and self.config.use_aux_loss:
                loss += self.config.aux_loss_weight * MANAGER.aggregate_aux_loss()
                MANAGER.reset_aux_loss()

            if self.config.use_moe and self.config.use_router_z_loss:
                loss += (
                    self.config.router_z_loss_weight * MANAGER.aggregate_router_z_loss()
                )
                MANAGER.reset_router_z_loss()

        return logits, loss


class DataloaderLite:

    def __init__(self, B, T, config, rank, world_size):
        self.B = B
        self.T = T
        self.rank = rank
        self.world_size = world_size
        self.current_idx = self.rank * T * B

        file_path = config.path

        with open(
            os.path.expanduser(file_path),
            "r",
        ) as f:
            text = f.read()

        self.enc = tiktoken.get_encoding("gpt2")
        self.tokens = torch.tensor(self.enc.encode(text))
        print(f"Total number of tokens {self.tokens.shape[0]}")

    def next_batch(self):
        buff = self.tokens[self.current_idx : self.current_idx + self.B * self.T + 1]
        x = buff[:-1].view(self.B, self.T)
        y = buff[1:].view(self.B, self.T)

        self.current_idx += self.B * self.T * self.world_size

        if self.current_idx + (self.B * self.T * self.world_size + 1) >= len(
            self.tokens
        ):
            self.current_idx = self.rank * self.T * self.B

        return x, y


def get_lr(step, config):

    # linear warmup
    if step < config.model.optimizer.warmup_steps:
        return (
            config.model.optimizer.max_lr
            * (step + 1)
            / config.model.optimizer.warmup_steps
        )

    if step > config.train.max_steps:
        return config.model.optimizer.min_lr

    # cosine decay
    # normalize decay ratio to 0-1
    decay_ratio = (step - config.model.optimizer.warmup_steps) / (
        config.train.max_steps - config.model.optimizer.warmup_steps
    )
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.model.optimizer.min_lr + coeff * (
        config.model.optimizer.max_lr - config.model.optimizer.min_lr
    )


def train(config, device="cpu", world_size=1, rank=0, local_rank=0, ddp=False):

    dataloader = DataloaderLite(8, 1024, config.data, rank, world_size)
    B, T = dataloader.B, dataloader.T
    total_batch_size = config.train.total_batch_size

    assert (
        total_batch_size % (B * T * world_size) == 0
    ), "total_batch_size must be divisible by the batch size x sequence length"

    gradient_accumulation_steps = total_batch_size // (B * T * world_size)

    model = Codex(config.model)
    model.to(device)

    if device == "cuda":
        model = torch.compile(model)

    if ddp:
        model = DDP(model, device_ids=[local_rank])

    raw_model = model.module if ddp else model

    optimizer = raw_model.configure_optimizer(device)

    scaler = torch.amp.GradScaler()

    for epoch in range(config.train.epochs):

        for step in range(config.train.max_steps):
            t0 = time.time()
            optimizer.zero_grad()
            loss_accum = 0.0
            for micro_step in range(gradient_accumulation_steps):
                x, y = dataloader.next_batch()
                x, y = x.to(device), y.to(device)
                # Autocast increases computation on the device because it adds dtype conversion operations that wouldn't exist otherwise. But the is little compared to the increase in flops and speed of loading data it provides.
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                    loss = loss / gradient_accumulation_steps
                    loss_accum += loss.detach()
                # we don't want ddp to sync gradients every micro_step
                if ddp:
                    model.require_backward_grad_sync = (
                        micro_step == gradient_accumulation_steps - 1
                    )
                scaler.scale(loss).backward()
            if ddp:
                # get loss_accum from all processes and average them, so we print the average loss for all processes and not just for rank 0
                dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            lr = get_lr(step, config)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if device == "cuda":
                torch.cuda.synchronize()
            if device == "mps":
                torch.mps.synchronize()
            t1 = time.time()
            dt = (t1 - t0) * 1000
            toks_sec = (B * T * gradient_accumulation_steps * world_size) / (t1 - t0)
            if rank == 0:
                print(
                    f"Epoch {epoch}: step: {step}, lr: {lr}, loss: {loss_accum.item()}, time: {dt}ms, toks/sec: {toks_sec}, norm: {norm}"
                )

    if ddp:
        dist.destroy_process_group()


@hydra.main(config_path="src/codex/config", config_name="config.yaml")
def main(config):

    ddp = int(os.environ.get("RANK", -1)) != -1
    print(f"DDP: {ddp}")
    if ddp:
        assert torch.cuda.is_available(), "For ddp training, cuda is required"
        dist.init_process_group(backend="nccl")
        rank = int(os.environ.get("RANK"))
        local_rank = int(os.environ.get("LOCAL_RANK"))
        world_size = int(os.environ.get("WORLD_SIZE"))
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        master_process = rank == 0
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        master_process = True
        device = "cpu"
        torch.manual_seed(1337)

        if torch.cuda.is_available():
            device = "cuda"
            torch.cuda.manual_seed(1337)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            torch.mps.manual_seed(1337)
        print(f"Using device: {device}")

    torch.set_float32_matmul_precision("high")
    # TODO: Add params to config
    train(config, device, world_size, rank, local_rank, ddp)

    if master_process:
        print(f"World size: {world_size}")


if __name__ == "__main__":
    main()
