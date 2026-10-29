# GPU Zero quickstart

A small GPT that learns to write like Shakespeare, one character at a time.
It trains on a laptop's CPU in a few minutes, and it is a complete repository
for [GPU Zero](https://gpuzero.dev): fork it, connect it in the console, and
launch it on a GPU in your own cloud account.

What it shows is the part that matters on rented GPUs: **stop it at any point
and it carries on from the same step, with the same losses**. That is
[Ravex](https://github.com/gpuzero-dev/ravex), one decorator in `train.py`.

## On your machine

Python 3.11 or later, and PyTorch for your machine
([pytorch.org](https://pytorch.org/get-started/locally/)); then:

```bash
pip install "ravex[moonclip]"
python train.py
```

The first run downloads Tiny Shakespeare, about a megabyte, into `data/`. Every
100 steps it prints the losses:

```
0.82M parameters on cpu, starting at step 0 of 2000
step 100  train 2.597  eval 2.590
step 200  train 2.417  eval 2.435
step 300  train 2.284  eval 2.292
```

**Stop it** with Ctrl-C, or `kill` it from another terminal, and run the same
command again. The first line says where it starts:

```
0.82M parameters on cpu, starting at step 300 of 2000
step 400  train 2.139  eval 2.188
```

It finishes the same 2000 steps the first run would have, with the same
losses, down to an eval loss around 1.83, and then writes 300 characters of
its own, which by then look like this:

```
They nothimsed sepale and in thy more shallock'l that soul
I never of I groporvery not; thy saffers there and;
And with pleak I soul right shil me, not blood
``` The checkpoints are in `checkpoints/`; delete the folder
to start over.

## On a GPU in your cloud, with GPU Zero

1. Fork this repository on GitHub.
2. In the [console](https://console.gpuzero.dev):
   - **Projects**: add your fork.
   - **Providers**: connect RunPod or vast.ai with an API key from that account.
   - **Compute**: pick the GPUs you allow and a monthly budget.
   - **Launch**: choose the newest commit and the `train` script, and queue it.
3. Watch `train/loss` and `eval/loss` on the run's page. From there you can
   resume the run, or fork it from a checkpoint with a different `lr`.

`gpuzero.toml` is what the console reads: the script, the parameters Launch
shows (each reaches `train.py` as a flag), the ones that change the model, and
where it may run. It asks for a 16-24 GB GPU on a spot node and stops a launch
at 2 US dollars; the provider bills your account directly. The
[docs](https://gpuzero.dev/docs#gpuzero-toml) describe every key.

## The files

| File | What it is |
|---|---|
| `train.py` | The model and the loop, with the Ravex decorator. Its docstring says how it stays resumable. |
| `prepare.py` | Downloads the text and turns it into numbers. |
| `gpuzero.toml` | The scripts GPU Zero may run from this repository, and where. |

To make it yours, change the model and the data in `train.py` and keep three
things: the decorator, `ravex.log_metrics` for what you want on the charts,
and flags for the parameters you list in `gpuzero.toml`.

## License

Apache-2.0. Tiny Shakespeare is the corpus from Andrej Karpathy's
[char-rnn](https://github.com/karpathy/char-rnn); it is not stored here.
