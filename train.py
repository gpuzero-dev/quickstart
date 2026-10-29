"""A small GPT that learns to write like Shakespeare, one character at a time, under Ravex.

    python train.py                    # a CPU is enough: a few minutes
    python train.py --steps 5000       # longer, better text

Stop it whenever you like (Ctrl-C, `kill`, a spot instance taken away) and
run the same command again: it carries on from the step it stopped at, with
the same losses it would have had. That is the whole point of the one Ravex
line below, the decorator on `run()`.

The data is Tiny Shakespeare, about a megabyte of text, downloaded into
``data/`` by ``prepare.py`` the first time. Every flag is also a parameter in
``gpuzero.toml``, so the console's Launch page can change it.

**Batches are drawn by step, not by position.** Step ``s`` takes its windows
from a generator seeded with ``seed + s``, so a run resumed at step ``s`` sees
exactly the batches it would have seen, with nothing about the data to save in
the checkpoint.

**The checkpoints' location is left to the environment.** Locally they go to
``./checkpoints``; on a node, the agent names the run's store with
``RAVEX_STORAGE_PATH``. Setting it in the decorator would win over that.
"""

from __future__ import annotations

import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import ravex
from prepare import load

HERE = os.path.dirname(os.path.abspath(__file__))


class GPT(nn.Module):
    """A decoder-only transformer over characters: embeddings, causal
    self-attention blocks, and a head that predicts the next character."""

    def __init__(self, vocab: int, seq: int, width: int, layers: int, heads: int, dropout: float):
        super().__init__()
        self.embed = nn.Embedding(vocab, width)
        self.position = nn.Embedding(seq, width)
        block = nn.TransformerEncoderLayer(
            width, heads, dim_feedforward=4 * width, dropout=dropout, batch_first=True, norm_first=True
        )
        self.blocks = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        positions = torch.arange(length, device=x.device)
        mask = nn.Transformer.generate_square_subsequent_mask(length, device=x.device)
        h = self.embed(x) + self.position(positions)
        h = self.blocks(h, mask=mask, is_causal=True)
        return self.head(self.norm(h))

    @torch.no_grad()
    def write(self, start: torch.Tensor, characters: int, generator: torch.Generator) -> torch.Tensor:
        """Continue ``start`` by sampling one character at a time."""
        out = start
        for _ in range(characters):
            logits = self(out[:, -self.position.num_embeddings :])[:, -1]
            probabilities = F.softmax(logits, dim=-1)
            out = torch.cat([out, torch.multinomial(probabilities, 1, generator=generator)], dim=1)
        return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=2000, help="length of the whole training, across restarts")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seq", type=int, default=64, help="characters a window holds")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    train, val, alphabet = load(os.path.join(HERE, "data"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def batch_of(source: torch.Tensor, generator: torch.Generator):
        starts = torch.randint(len(source) - args.seq - 1, (args.batch,), generator=generator)
        window = torch.stack([source[s : s + args.seq + 1] for s in starts.tolist()]).to(device)
        return window[:, :-1], window[:, 1:]

    def lr_at(step: int) -> float:
        """Linear warm-up, then a cosine down to a tenth."""
        if step < args.warmup:
            return (step + 1) / args.warmup
        progress = min(1.0, (step - args.warmup) / max(1, args.steps - args.warmup))
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    # The one Ravex line. `max_steps` is the length of the whole training
    # across restarts, so a resumed run stops where the first one would have.
    @ravex.train_loop(max_steps=args.steps, checkpoint_every=100)
    def run() -> None:
        torch.manual_seed(args.seed)
        model = GPT(len(alphabet), args.seq, args.width, args.layers, args.heads, args.dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
        schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)

        def loss_of(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            logits = model(x)
            return F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))

        parameters = sum(p.numel() for p in model.parameters())
        print(
            f"{parameters / 1e6:.2f}M parameters on {device}, starting at step {ravex.step()} of {args.steps}",
            flush=True,
        )

        last = time.monotonic()
        while ravex.step() < args.steps:
            # Where this step starts in the data, whichever run gets here.
            generator = torch.Generator().manual_seed(args.seed + ravex.step())
            model.train()
            x, y = batch_of(train, generator)
            loss = loss_of(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()

            now = time.monotonic()
            ravex.log_metrics(
                {
                    "train/loss": loss.item(),
                    "perf/characters_per_s": args.batch * args.seq / max(now - last, 1e-9),
                }
            )
            last = now

            done = ravex.step()
            if done % args.eval_every == 0 or done == args.steps:
                model.eval()
                with torch.no_grad():
                    losses = [
                        loss_of(*batch_of(val, torch.Generator().manual_seed(10_000_000 + i))).item()
                        for i in range(args.eval_batches)
                    ]
                eval_loss = sum(losses) / len(losses)
                ravex.log_metrics({"eval/loss": eval_loss})
                print(f"step {done}  train {loss.item():.3f}  eval {eval_loss:.3f}", flush=True)
                last = time.monotonic()

        model.eval()
        start = torch.tensor([[alphabet.index("\n")]], device=device)
        sample = model.write(start, 300, torch.Generator(device=device).manual_seed(args.seed))
        print("".join(alphabet[i] for i in sample[0].tolist()), flush=True)

    run()


if __name__ == "__main__":
    main()
