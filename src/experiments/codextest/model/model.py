# n-layer vanilla fully connected network with an embedding layer and an output layer to play with.

import torch
import torch.nn as nn
import torch.nn.functional as F


class CodexTest(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.embedding_layer = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = nn.ModuleList([nn.Linear(config.n_embd, config.n_embd) for _ in range(config.n_layers)])
        self.output_layer = nn.Linear(config.n_embd, config.vocab_size)
        self.relu = nn.ReLU()
        self.config = config

    def forward(self, x, targets=None):
        x = self.embedding_layer(x)
        for layer in self.layers:
            x = self.relu(layer(x))

        x = self.output_layer(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(x.view(-1, x.size(-1)), targets.view(-1))
        return x, loss

    def configure_optimizer(self, device_type=None):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.optimizer.learning_rate)
        return optimizer
