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


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.RESIDUAL_SCALE = 1

        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(attn)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.c_fc = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(3 * config.n_embd, config.n_embd)
        self.c_proj.RESIDUAL_SCALE = 1
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
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
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
                ln_f=nn.LayerNorm(config.n_embd),
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

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
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
