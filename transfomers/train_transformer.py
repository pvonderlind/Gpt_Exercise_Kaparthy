import os
# to allow keyboardinterrupt
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

from transfomers import models
from data_loader import load_preprocessed_splits_for_txt, decoder, load_crd3_text
from typing import Tuple
import torch

# Context size of each input of the transformer (e.g. n tokens in a single example)

BLOCK_SIZE = 256
BATCH_SIZE = 64
EVAL_ITER = 200
N_HEADS = 6
N_BLOCK_LAYERS = 6
N_EMB = 384
LR = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

N_STEPS = 5000
MAX_OUT_TOKENS = 500
N_SEQUENCES = 10
MODEL_PATH = 'crd3_lm_test'
TXT_OUTPUT = f'text_output_{MODEL_PATH}.txt'

torch.manual_seed(1337)


def get_random_batch(data: torch.Tensor, batch_size: int = BATCH_SIZE) -> Tuple[torch.Tensor, torch.Tensor]:
    rand_ixs = torch.randint(len(data) - BLOCK_SIZE, (batch_size,))
    x = torch.stack([data[i:i + BLOCK_SIZE] for i in rand_ixs])
    # --> Target for transformer is input shifted by 1.
    y = torch.stack([data[i + 1:i + 1 + BLOCK_SIZE] for i in rand_ixs])
    return x.to(device), y.to(device)


def train_bigram_language_model(data: torch.Tensor, vocab_size: int, n_steps: int = 100)\
        -> models.BigramLanguageModel:
    model = models.BigramLanguageModel(vocab_size=vocab_size, block_size=BLOCK_SIZE, n_embd=N_EMB, n_heads=N_HEADS,
                                       n_block_layers=N_BLOCK_LAYERS, device=device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss = None
    try:
        for steps in range(n_steps):
            # Get random batch
            xb, yb = get_random_batch(data, BATCH_SIZE)

            # Eval loss
            if steps % EVAL_ITER == 0:
                train_loss = eval_loss(model, train)
                val_loss = eval_loss(model, val)
                print(f"Iteration {steps} | Training loss: {train_loss:.4f} ; Validation loss: {val_loss:.4f}")

            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    except KeyboardInterrupt:
        print("Stopping early ...")
        return model
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
    train, val, enc_dict, dec_dict = load_crd3_text()
    if not os.path.exists(MODEL_PATH):
        bigram_model = train_bigram_language_model(train, vocab_size=len(enc_dict.keys()), n_steps=4000)
        torch.save(bigram_model, MODEL_PATH)
    else:
        print(f"Loading previous model from {MODEL_PATH}")
        bigram_model = torch.load(MODEL_PATH)
    context = torch.zeros((N_SEQUENCES, BLOCK_SIZE), dtype=torch.long, device=device)
    out_tokens = bigram_model.generate(context, max_new_tokens=MAX_OUT_TOKENS)
    out_tokens = out_tokens.view(out_tokens.shape[0] * out_tokens.shape[1])
    text = decoder(out_tokens.tolist(), dec_dict)
    with open(TXT_OUTPUT, 'w+') as file:
        file.write(text)
