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
from torchtitan.models.moe import GroupedExperts, FeedForward, MoEArgs
from torchtitan.models.attention import build_attention


# Adapted from https://github.com/DeepSeek-ai/DeepSeek-V3/blob/main/inference/model.py#L294
def precompute_freqs_cis(args) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args: Model arguments containing positional embedding parameters.

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
    def __init__(self, model_args):
        super().__init__()

        assert model_args.d_model % model_args.n_heads == 0

        self.c_attn = nn.Linear(model_args.d_model, 3 * model_args.d_model)
        self.c_proj = nn.Linear(model_args.d_model, model_args.d_model)
        self.c_proj.RESIDUAL_SCALE = 1

        self.n_head = model_args.n_heads
        self.n_embd = model_args.d_model
        self.rotary_pos_emb = RotaryPositionalEmbeddings(
            model_args.d_model // model_args.n_heads
        )

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

        attention_scale = 1.0 / q.size(-1) ** 0.5
        if self.model_args.use_mup:
            attention_scale = 1.0 / q.size(-1)

        attn = F.scaled_dot_product_attention(
            q, k, v, scale=attention_scale, is_causal=True
        )
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(attn)

    def init_weights(self, init_std, mup_multiplier, residual_scale):
        nn.init.normal_(self.c_attn.weight, mean=0.0, std=init_std * mup_multiplier)
        nn.init.normal_(
            self.c_proj.weight, mean=0.0, std=init_std * mup_multiplier * residual_scale
        )


# Adapted from https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/deepseek_v3/model/model.py#L147 to include gated attetion
class MultiHeadLatentAttention(nn.Module):
    """
    Multi-head attention (MLA) module.

    """

    def __init__(self, model_args):
        super().__init__()
        self.dim = model_args.d_model
        self.n_heads = model_args.n_heads
        self.q_lora_rank = model_args.q_lora_rank
        self.kv_lora_rank = model_args.kv_lora_rank
        self.qk_nope_head_dim = model_args.qk_nope_head_dim
        self.qk_rope_head_dim = model_args.qk_rope_head_dim
        # the RoPE dimension is just the subset of the head dimension that gets rotated
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

        # gate parameters for gatted attention
        self.wg = nn.Linear(self.dim, self.n_heads * self.v_head_dim, bias=False)

        self.softmax_scale = self.qk_head_dim**-0.5

        if model_args.max_seq_len > model_args.original_seq_len:
            mscale = 0.1 * model_args.mscale * math.log(model_args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        self.sdpa = build_attention(model_args.use_flex_attn, model_args.attn_mask_type)

        self.model_args = model_args

    def init_weights(self, init_std, mup_multiplier, residual_scale):
        mup_scale = mup_multiplier**-0.5

        # Handle wq (Dense or LoRA)
        if self.q_lora_rank == 0:
            nn.init.normal_(self.wq.weight, mean=0.0, std=init_std * mup_scale)
        else:
            # wq_a: Down projection (d -> r). Use mup_scale (1/sqrt(d)) for stable pre-norm activation
            nn.init.normal_(self.wq_a.weight, mean=0.0, std=init_std * mup_scale)
            # wq_b: Up projection (r -> d). Input is normalized (var=1). Scale by 1/sqrt(r) for stable output
            scale_b = init_std * (self.q_lora_rank**-0.5)
            nn.init.normal_(self.wq_b.weight, mean=0.0, std=scale_b)
            self.q_norm.reset_parameters()

        # Handle wkv (LoRA)
        # wkv_a: Down projection (d -> r). Use mup_scale (1/sqrt(d))
        nn.init.normal_(self.wkv_a.weight, mean=0.0, std=init_std * mup_scale)

        # wkv_b: Up projection (r -> d_out). Input is normalized. Scale by 1/sqrt(r)
        scale_b = init_std * (self.kv_lora_rank**-0.5)
        nn.init.normal_(self.wkv_b.weight, mean=0.0, std=scale_b)
        self.kv_norm.reset_parameters()

        # Gate projection
        nn.init.normal_(self.wg.weight, mean=0.0, std=init_std * mup_scale)

        # Output projection
        nn.init.normal_(
            self.wo.weight, mean=0.0, std=init_std * residual_scale * mup_scale
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
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
        g = self.wg(x)  # (bsz, seqlen, n_heads * v_head_dim)
        g = g.view(bsz, seqlen, self.n_heads, self.v_head_dim)

        q = q.transpose(1, 2)  # (bsz, n_heads, seqlen, qk_head_dim)
        k = k.transpose(1, 2)  # (bsz, n_heads, seqlen, qk_head_dim)
        v = v.transpose(1, 2)  # (bsz, n_heads, seqlen, v_head_dim)

        # softmax scale for mup
        if self.model_args.use_mup:
            self.softmax_scale = 1.0 / q.size(-1)

        output = self.sdpa(q, k, v, scale=self.softmax_scale)

        # Reshape and project output
        output = output.transpose(
            1, 2
        ).contiguous()  # (bsz, seqlen, n_heads, v_head_dim)

        # apply gate to output
        output = (
            torch.sigmoid(g) * output
        )  # another implementation used output = output * torch.sigmoid(g) instead, just noting

        output = output.view(bsz, seqlen, -1)  # (bsz, seqlen, n_heads * v_head_dim)

        return self.wo(output)  # (bsz, seqlen, dim)


class MLP(nn.Module):
    def __init__(self, model_args):
        super().__init__()

        d_ff = 3 * model_args.d_model
        assert d_ff % 2 == 0
        self.c_fc = nn.Linear(model_args.d_model, d_ff)
        self.c_proj = nn.Linear(d_ff // 2, model_args.d_model)
        self.c_proj.RESIDUAL_SCALE = 1
        self.silu = nn.SiLU()

    def init_weights(self, init_std, buffer_device=None):
        nn.init.trunc_normal_(self.c_fc.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.c_proj.weight, mean=0.0, std=init_std)

    def forward(self, x):
        """
        swiglu based on flash attention implementation
        """
        y = self.c_fc(x)
        y, gate = y.chunk(2, dim=-1)
        return self.c_proj(self.silu(y) * gate)


class CodexGroupedExperts(GroupedExperts):
    def __init__(self, dim, hidden_dim, num_experts, use_grouped_mm):
        super().__init__(dim, hidden_dim, num_experts, use_grouped_mm)

    def init_weights(self, init_std, mup_multiplier, residual_scale):

        mup_scale = mup_multiplier**-0.5
        nn.init.trunc_normal_(self.w1, mean=0.0, std=init_std * mup_scale)
        # w2 is the projection weight, it should be scaled by the mup scale and the residual scale
        nn.init.trunc_normal_(
            self.w2, mean=0.0, std=init_std * residual_scale * mup_scale
        )
        # w3 is the gate weight, i'm not convinced it should be scaled by the residual scale
        nn.init.trunc_normal_(self.w3, mean=0.0, std=init_std * mup_scale)


class CodexFeedForward(FeedForward):
    def __init__(self, dim, hidden_dim):
        super().__init__(dim, hidden_dim)

    def init_weights(
        self, init_std, mup_multiplier, residual_scale, buffer_device=None
    ):
        mup_scale = mup_multiplier**-0.5
        nn.init.normal_(self.w1.weight, mean=0.0, std=init_std * mup_scale)
        nn.init.normal_(
            self.w2.weight, mean=0.0, std=init_std * residual_scale * mup_scale
        )
        nn.init.normal_(self.w3.weight, mean=0.0, std=init_std * mup_scale)


class Gate(nn.Module):

    def __init__(
        self, dim, num_experts, top_k, score_func, route_norm, route_scale, model_args
    ):
        """
        top_k
        num_experts
        n_groups
        n_limited_groups
        func -> sigmoid, softmax
        bias
        """
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.func = score_func
        self.w_gate = nn.Linear(dim, num_experts, bias=False)
        self.n_groups = model_args.n_expert_groups
        self.n_limited_groups = model_args.n_limited_groups
        self.route_scale = route_scale

    def init_weights(self, init_std: float, mup_multiplier: float):
        mup_scale = mup_multiplier**-0.5
        nn.init.trunc_normal_(self.w_gate.weight, mean=0.0, std=init_std * mup_scale)

    def forward(self, x, expert_bias=None):
        """
        x -> B*T, n_embd
        """
        gate_scores = self.w_gate(x)  # B*T, n_exp

        if self.func == "sigmoid":
            gate_scores = gate_scores.sigmoid(dim=-1, dtype=torch.float32)
        elif self.func == "softmax":
            gate_scores = gate_scores.softmax(dim=-1, dtype=torch.float32)

        original_scores = gate_scores

        if expert_bias is not None:
            gate_scores = gate_scores + expert_bias

        if self.n_groups > 1:
            assert (
                self.num_experts % self.n_groups == 0
            ), "num_experts is indivisible by n_groups"
            n_group_exp = self.num_experts // self.n_groups
            # in deepseek implementation, tokens can choose from up to n_limited_groups groups, so n_group_exp doesn't need to be greater than top_k, but here we n_limited_groups is set to 1 ), so we need to assert that n_group_exp is greater than top_k
            if self.n_limited_groups == 1:
                assert (
                    n_group_exp > self.top_k
                ), "number of experts in a group should be greater than topk"

            gate_scores = gate_scores.view(x.size(0), self.n_groups, -1)

            if expert_bias is None:
                group_scores = gate_scores.amax(dim=-1)
            else:
                group_scores = gate_scores.topk(2, dim=-1)[0].sum(dim=-1)

            # select top groups i.e maximum number of groups a token can choose from
            top_group_indices = group_scores.topk(self.n_limited_groups, dim=-1)[
                1
            ]  # B*S, n_limited_groups

            # create mask
            mask = gate_scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(
                1, top_group_indices, False
            )

            gate_scores = gate_scores.masked_fill_(
                mask.unsqueeze(-1), float("-inf")
            ).flatten(1)

        topk_indices = torch.topk(gate_scores, self.top_k, dim=-1)[1]
        weights = original_scores.gather(1, topk_indices)
        if self.func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale

        # group tokens together by expert indices from 0 to num_experts and pass that to experts forward
        num_tokens_per_expert = torch.histc(
            topk_indices.view(-1),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )

        return weights.type_as(x), topk_indices, num_tokens_per_expert


class MoE(nn.Module):
    def __init__(self, moe_args: MoEArgs, dim: int, hidden_dim: int, model_args):
        super().__init__()

        num_experts = moe_args.num_experts

        self.experts = CodexGroupedExperts(
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
            model_args=model_args,
        )

        self.shared_experts = (
            CodexFeedForward(
                dim=dim, hidden_dim=hidden_dim * moe_args.num_shared_experts
            )
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

    def init_weights(
        self,
        init_std: float,
        mup_multiplier: float,
        residual_scale: float,
        buffer_device: torch.device,
    ):
        self.experts.init_weights(init_std, mup_multiplier, residual_scale)
        self.router.init_weights(init_std, mup_multiplier)
        if self.shared_experts is not None:
            self.shared_experts.init_weights(init_std, mup_multiplier, residual_scale)

        with torch.device(buffer_device):
            self.tokens_per_expert = torch.zeros(
                self.experts.num_experts, dtype=torch.float32
            )
            if self.load_balance_coeff is not None:
                self.expert_bias = torch.zeros(
                    self.experts.num_experts, dtype=torch.float32
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

        # top_scores and selected_experts_indices have shape (bs*slen, top_k)
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

        # NOTE: We do not really need to sort the tokens by experts here since we are using DeepExpertParallel and this uses a different all-to-all implementation that does not require sorting the tokens by experts.
        # However the tokens on each local rank needs to sorted after all-to-all communication(taken care of by DeepExpertParallel) to be able to use grouped mm.

        # shape (bs*slen*top_k, dim)
        token_indices = selected_experts_indices.reshape(-1, 1).expand(-1, dim)

        # shape (bs*slen*top_k, dim)
        routed_input = torch.gather(x, dim=0, index=token_indices)

        if self.score_before_experts:
            routed_input = (
                routed_input.to(torch.float32) * top_scores.reshape(-1, 1)
            ).to(x.dtype)

        # shape (bs*slen*top_k, dim)
        # TODO: I'm not sure num_tokens_per_expert is properly calibrated here.
        routed_output = self.experts(routed_input, num_tokens_per_expert)

        if not self.score_before_experts:
            routed_output = (
                routed_output.to(torch.float32) * top_scores.reshape(-1, 1)
            ).to(x.dtype)

        # shared expert
        if self.shared_experts is not None:
            out = self.shared_experts(x)
        else:
            out = torch.zeros_like(x)

        out = out.scatter_add(dim=0, index=token_indices, src=routed_output)
        out = out.reshape(bs, slen, dim)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, model_args, layer_id, use_moe=False, linear_attention=True):
        super().__init__()
        from fla.layers import GatedDeltaNet

        self.ln_1 = nn.RMSNorm(model_args.d_model)
        self.linear_attention = linear_attention
        if linear_attention:
            self.attn = GatedDeltaNet(
                hidden_size=model_args.d_model,
                num_heads=model_args.n_heads,
                mode="chunk",
            )
        else:
            if model_args.use_mla:
                self.attn = MultiHeadLatentAttention(model_args)
            else:
                self.attn = Attention(model_args)
        self.ln_2 = nn.RMSNorm(model_args.d_model)
        self.moe_enabled = use_moe
        if use_moe:
            self.moe = MoE(
                model_args.moe_args,
                model_args.d_model,
                model_args.moe_inter_dim,
                model_args,
            )
        else:
            self.mlp = CodexFeedForward(model_args.d_model, model_args.inter_dim)
        # residual scaling from gpt 2
        self.residual_scale = (
            ((2 * (model_args.n_layers)) ** -0.5)
            if model_args.use_residual_scaling
            else 1.0
        )
        self.mup_multiplier = (
            (model_args.d_model / model_args.mup_base_dim)
            if model_args.use_mup
            else 1.0
        )
        self.init_std = model_args.init_std
        # self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        self.layer_id = layer_id

    def init_weights(self, buffer_device):
        for norm in (self.ln_1, self.ln_2):
            norm.reset_parameters()
        if not self.linear_attention:
            self.attn.init_weights(
                self.init_std, self.mup_multiplier, self.residual_scale
            )

        if self.moe_enabled:
            self.moe.init_weights(
                self.init_std, self.mup_multiplier, self.residual_scale, buffer_device
            )
        else:
            self.mlp.init_weights(
                self.init_std, self.mup_multiplier, self.residual_scale, buffer_device
            )

    def forward(self, x, freqs_cis):
        attn_in = self.ln_1(x)
        if self.linear_attention:
            attn_out = self.attn(attn_in)
        else:
            attn_out = self.attn(attn_in, freqs_cis)
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]
        x = x + attn_out

        if self.moe_enabled:
            x = x + self.moe(self.ln_2(x))
        else:
            x = x + self.mlp(self.ln_2(x))
        return x


class Codex(nn.Module):

    def __init__(self, model_args):
        super().__init__()

        model_args.inter_dim = model_args.d_model * 4

        self.model_args = model_args

        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.d_model)

        self.register_buffer(
            "freqs_cis", precompute_freqs_cis(model_args), persistent=False
        )

        self.layers = nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(
                model_args,
                layer_id,
                use_moe=((layer_id % model_args.p) == 0 and model_args.use_moe),
                linear_attention=(layer_id % model_args.g) != 0,
            )

        self.norm = nn.RMSNorm(model_args.d_model)

        self.output = nn.Linear(model_args.d_model, model_args.vocab_size, bias=False)

        self.tok_embeddings.weight = self.output.weight

        self.mup_multiplier = (
            (model_args.d_model / model_args.mup_base_dim)
            if model_args.use_mup
            else 1.0
        )
        self.mup_input_alpha = model_args.mup_input_alpha if model_args.use_mup else 1.0
        self.mup_output_alpha = (
            model_args.mup_output_alpha if model_args.use_mup else 1.0
        )

        # We don't apply init_weights call here since we are using meta init

    def init_weights(self, buffer_device):
        buffer_device = buffer_device or self.freqs_cis.device
        with torch.device(buffer_device):
            self.freqs_cis = precompute_freqs_cis(self.model_args)
        if self.tok_embeddings is not None:
            nn.init.normal_(
                self.tok_embeddings.weight, mean=0.0, std=self.model_args.init_std
            )
        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights(buffer_device=buffer_device)
        if self.norm is not None:
            self.norm.reset_parameters()
        # TODO: confirm if this is correct, also we are using tied weights
        # final_out_std = self.model_args.d_model**-0.5
        # cutoff_factor = 3
        # if self.lm_head is not None:
        #     nn.init.trunc_normal_(
        #         self.lm_head.weight,
        #         mean=0.0,
        #         std=final_out_std,
        #         a=-cutoff_factor * final_out_std,
        #         b=cutoff_factor * final_out_std,
        #     )

    def forward(self, tokens, input_batch=None):
        B, T = tokens.size()
        x = self.tok_embeddings(tokens)

        if self.model_args.use_mup:
            x *= self.mup_input_alpha

        for block in self.layers.values():
            x = block(x, self.freqs_cis)

        x = self.norm(x)

        if self.model_args.use_mup:

            x *= self.mup_output_alpha / self.mup_multiplier

        logits = self.output(x)

        return logits
