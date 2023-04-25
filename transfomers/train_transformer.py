from transfomers import models
from data_loader import load_preprocessed_splits_for_txt, decoder
from typing import Tuple
import torch

# Context size of each input of the transformer (e.g. n tokens in a single example)

BLOCK_SIZE = 8
BATCH_SIZE = 32
EVAL_ITER = 200
N_EMB = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_random_batch(data: torch.Tensor, batch_size: int = BATCH_SIZE) -> Tuple[torch.Tensor, torch.Tensor]:
    rand_ixs = torch.randint(len(data) - BLOCK_SIZE, (batch_size,))
    x = torch.stack([data[i:i + BLOCK_SIZE] for i in rand_ixs])
    # --> Target for transformer is input shifted by 1.
    y = torch.stack([data[i + 1:i + 1 + BLOCK_SIZE] for i in rand_ixs])
    return x.to(device), y.to(device)


def train_bigram_language_model(data: torch.Tensor, vocab_size: int, batch_size: int = BATCH_SIZE, n_steps: int = 100)\
        -> models.BigramLanguageModel:
    model = models.BigramLanguageModel(vocab_size, block_size=BLOCK_SIZE, n_embd=N_EMB, device=device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = None
    for steps in range(n_steps):
        # Get random batch
        xb, yb = get_random_batch(data, batch_size)

        # Eval loss
        if steps % EVAL_ITER == 0:
            train_loss = eval_loss(model, train)
            val_loss = eval_loss(model, val)
            print(f"Iteration {steps} | Training loss: {train_loss:.4f} ; Validation loss: {val_loss:.4f}")

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return model


@torch.no_grad()
def eval_loss(model: torch.nn.Module, data: torch.Tensor):
    model.eval()
    losses = torch.zeros(EVAL_ITER)
    for k in range(EVAL_ITER):
        x, y = get_random_batch(data)
        logits, loss = model(x, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean()


if __name__ == "__main__":
    train, val, enc_dict, dec_dict = load_preprocessed_splits_for_txt('shakespeare.txt')
    bigram_model = train_bigram_language_model(train, vocab_size=len(enc_dict.keys()), n_steps=10000)
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decoder(bigram_model.generate(context, max_new_tokens=500)[0].tolist(), dec_dict))

