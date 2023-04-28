import torch
from typing import List, Tuple
from datasets import load_dataset


def load_crd3_text() -> Tuple[torch.Tensor, torch.Tensor, dict, dict]:
    crd3 = load_dataset('crd3')
    train_lines = ''.join(crd3["train"]["chunk"])
    val_lines = ''.join(crd3["validation"]["chunk"])
    whole_ds = train_lines + val_lines
    enc_dict, dec_dict = _build_encoder_decoder_dicts(whole_ds)
    encoded_text = encode(whole_ds, enc_dict)
    encoded_text_tensor = torch.LongTensor(encoded_text)
    train, val = _create_train_val_split(encoded_text_tensor, 0.1)
    return train, val, enc_dict, dec_dict


def load_preprocessed_splits_for_txt(path_to_txt: str) -> Tuple[torch.Tensor, torch.Tensor, dict, dict]:
    with open(path_to_txt, 'r', encoding='utf-8') as f:
        lines = f.read()
    enc_dict, dec_dict = _build_encoder_decoder_dicts(lines)
    encoded_text = encode(lines, enc_dict)
    encoded_text_tensor = torch.LongTensor(encoded_text)
    train, val = _create_train_val_split(encoded_text_tensor, 0.1)
    return train, val, enc_dict, dec_dict


def _build_encoder_decoder_dicts(text: str):
    chars = sorted(list(set(text)))
    encoder_dict = {ch: i for i, ch in enumerate(chars)}
    decoder_dict = {i: ch for i, ch in enumerate(chars)}
    return encoder_dict, decoder_dict


def encode(text: str, encoder_dict: dict):
    return [encoder_dict[c] for c in text]


def decoder(encoding: List[int], decoder_dict: dict):
    return ''.join([decoder_dict[i] for i in encoding])


def _create_train_val_split(data: torch.Tensor, val_percentage: float) -> Tuple[torch.Tensor, torch.Tensor]:
    n = int((1 - val_percentage) * len(data))
    train = data[:n]
    val = data[n:]
    return train, val
