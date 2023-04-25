from typing import Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int, block_size: int, device: str = 'cuda'):
        super().__init__()
        self.device = device
        self.block_size = block_size
        # Embedding for token content directly
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # Embedding for the position of the token
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_head = Head(head_size=n_embd, n_embd=n_embd, block_size=block_size)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = idx.shape

        # Arrange tensor into a batch by time by channel tensor (B,T,C) dim.
        token_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))  # (T,C)
        x = token_emb + pos_emb  # (B,T,C) --> created by broadcasting T,C to B dimension
        x = self.sa_head(x)  # Pass through self-attention
        logits = self.lm_head(x)  # (B,T,vocab_size)
        # E.g. predict what comes next by learning the logits (probabilities of next token) for each token.

        # Reshape so cross entropy loss works
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        # idx is of shape (B, T) of token indices in the current context
        for _ in range(max_new_tokens):
            # Crop context to the last block_size tokens due to positional encoding table now also being used!
            idx = idx[:, -self.block_size:]
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


class Head(nn.Module):
    """Head of self-attention"""

    def __init__(self, head_size: int, n_embd: int, block_size: int):
        super().__init__()
        # Note: it is important to de-activate bias as to not skew calculations!
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Register a buffer (e.g. a non-parameter variable in PyTorch)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        key = self.key(x)
        query = self.query(x)
        value = self.value(x)
        # Note: transpose is only necessary for the last two dimensions, as the first one is the batch dim.
        # Note: Normalize by sqrt of head size -> C ** -0.5!!
        wei = query * key.transpose(-2, -1) * C ** -0.5  # (B,T,C) @ (B,C,T) -> (B,T,T)
        # Here: Use a numerical trick to get 0 to be actually 0 after using the exp in the softmax!
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weu = F.softmax(wei, dim=-1)  # (B, T, T)

        # Weighted aggregation
        out = wei @ value
        return out
