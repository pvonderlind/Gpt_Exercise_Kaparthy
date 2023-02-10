from typing import Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Arrange tensor into a batch by time by channel tensor (B,T,C) dim.
        logits = self.token_embedding_table(idx)
        # E.g. predict what comes next by learning the logits (probabilities of next token) for each token.

        # Reshape so cross entropy loss works
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        # idx is of shape (B, T) of token indices in the current context
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            # Get only last timestep from the predicted next tokens of shape (B, C)
            logits = logits[:, -1, :]
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample next token by logits distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append sampled next token to idx
            # Shape is then (B, T+1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

