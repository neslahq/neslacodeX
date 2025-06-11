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
from utils import MOEManager

MANAGER = MOEManager()


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


class MOE(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.router = Router(config)
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


class Block(nn.Module):
    def __init__(self, config, use_moe=False):
        super().__init__()

        self.ln_1 = nn.RMSNorm(config.n_embd)
        self.attn = Attention(config)
        self.ln_2 = nn.RMSNorm(config.n_embd)
        if use_moe:
            self.mlp = MOE(config)
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
                h=nn.ModuleList(
                    [
                        Block(config, use_moe=((i % config.p) == 0 and config.use_moe))
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

    def configure_optimizer(self, device_type):
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


def train(config, device, world_size, rank, local_rank, ddp):

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

    for epoch in range(config.train.epochs):

        for step in range(config.train.max_steps):
            t0 = time.time()
            optimizer.zero_grad()
            loss_accum = 0.0
            for micro_step in range(gradient_accumulation_steps):
                x, y = dataloader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                    loss = loss / gradient_accumulation_steps
                    loss_accum += loss.detach()
                # we don't want ddp to sync gradients every micro_step
                if ddp:
                    model.require_backward_grad_sync = (
                        micro_step == gradient_accumulation_steps - 1
                    )
                loss.backward()
            if ddp:
                dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            lr = get_lr(step, config)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            optimizer.step()
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


@hydra.main(config_path="config", config_name="config.yaml")
def main(config):

    ddp = int(os.environ.get("RANK", -1)) != -1
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
