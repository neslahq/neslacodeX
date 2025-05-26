import torch.nn as nn
import torch.nn.functional as F
import torch
import hydra
import tiktoken


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

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=None)
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

    def __init__(self, B, T):
        self.B = B
        self.T = T

        self.current_idx = 0

        with open(
            "/Users/tinuademargaret/Documents/explorer/neslacodeX/src/data/input.txt",
            "r",
        ) as f:
            text = f.read()

        self.enc = tiktoken.get_encoding("gpt2")
        self.tokens = torch.tensor(self.enc.encode(text))

    def next_batch(self):
        buff = self.tokens[self.current_idx : self.current_idx + self.B * self.T + 1]
        x = buff[:-1].view(self.B, self.T)
        y = buff[1:].view(self.B, self.T)

        self.current_idx += self.B * self.T

        if self.current_idx + (self.B * self.T + 1) >= len(self.tokens):
            self.current_idx = 0

        return x, y


def train(config, device):
    model = Codex(config.model)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    dataloader = DataloaderLite(4, 32)

    for epoch in range(50):
        optimizer.zero_grad()
        x, y = dataloader.next_batch()
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch} loss: {loss.item()}")


@hydra.main(config_path="config", config_name="config.yaml")
def main(config):
    device = "cpu"
    torch.manual_seed(1337)

    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.manual_seed(1337)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        torch.mps.manual_seed(1337)
    print(f"Using device: {device}")

    train(config, device)

    print("Model created")


if __name__ == "__main__":
    main()
