"""Tiny Shakespeare, as the characters ``train.py`` learns from.

    python prepare.py          # downloads data/input.txt; train.py does it on its own

About a megabyte of Shakespeare's plays, the corpus Andrej Karpathy put
together for char-rnn. The first 90% is for training, the rest for the eval.
The alphabet is every character in the text, sorted, so a character's number
is the same on every machine and in every run.
"""

from __future__ import annotations

import os
import sys
import urllib.request
from typing import List, Tuple

import torch

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download(directory: str) -> str:
    """The text's path, fetched first if it is not there yet."""
    path = os.path.join(directory, "input.txt")
    if not os.path.isfile(path):
        os.makedirs(directory, exist_ok=True)
        print(f"downloading Tiny Shakespeare into {directory}", flush=True)
        # Into a temporary name first, so a download cut short is not taken
        # for the whole text by the next run.
        partial = path + ".part"
        urllib.request.urlretrieve(URL, partial)
        os.replace(partial, path)
    return path


def load(directory: str) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """The training and eval characters as numbers, and the alphabet."""
    with open(download(directory), encoding="utf-8") as handle:
        text = handle.read()
    alphabet = sorted(set(text))
    index = {character: i for i, character in enumerate(alphabet)}
    numbers = torch.tensor([index[character] for character in text], dtype=torch.long)
    split = int(len(numbers) * 0.9)
    return numbers[:split], numbers[split:], alphabet


if __name__ == "__main__":
    train, val, alphabet = load(sys.argv[1] if len(sys.argv) > 1 else "data")
    print(f"{len(train):,} training and {len(val):,} eval characters, {len(alphabet)} distinct")
