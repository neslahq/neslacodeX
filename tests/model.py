# n-layer vanilla fully connected network to play with.

import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(config.n_embd, config.n_embd) for _ in range(config.n_layers)])
        self.output_layer = nn.Linear(config.n_embd, config.vocab_size)
        self.relu = nn.ReLU()

    def forward(self, x, targets=None):
        for layer in self.layers:
            x = self.relu(layer(x))

        x = self.output_layer(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(x.view(-1, x.size(-1)), targets.view(-1))
        return x, loss

    def configure_optimizer(self, device_type=None):
        optimizer = torch.optim.AdamW(self.parameters(), lr=config.optimizer.learning_rate)
        return optimizer
