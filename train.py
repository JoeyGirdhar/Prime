"""
Train the tiny GPT on procedurally-generated molecule strings.

Runs on a laptop CPU in a few minutes with the default (small) config. No
downloads: the corpus is generated in-process and is valid by construction.

    python train.py                 # default: ~1500 iters
    python train.py --iters 3000    # train longer for a higher validity rate

Saves the trained model to checkpoints/model.pt and a loss curve to assets/.
"""

import argparse
import os
import time

import torch

from tinygpt import GPT, GPTConfig, CharTokenizer, generate_corpus

HERE = os.path.dirname(os.path.abspath(__file__))


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])   # targets = inputs shifted by 1
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, splits, block_size, batch_size, device, iters=50):
    out = {}
    model.eval()
    for name, data in splits.items():
        losses = torch.zeros(iters)
        for k in range(iters):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--n_layer", type=int, default=3)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_embd", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--corpus", type=int, default=4000)
    args = ap.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # --- Data ---
    molecules = generate_corpus(n=args.corpus, seed=0)
    tokenizer, text = CharTokenizer.from_corpus(molecules)
    print(f"corpus: {len(molecules)} molecules, {len(text)} chars, vocab={tokenizer.vocab_size}")
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    splits = {"train": data[:n], "val": data[n:]}

    # --- Model ---
    cfg = GPTConfig(vocab_size=tokenizer.vocab_size, block_size=args.block,
                    n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)
    model = GPT(cfg).to(device)
    print(f"model parameters: {model.num_params():,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # --- Training loop ---
    history = []
    start = time.time()
    for it in range(1, args.iters + 1):
        x, y = get_batch(splits["train"], args.block, args.batch, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if it % 250 == 0 or it == 1:
            ev = estimate_loss(model, splits, args.block, args.batch, device)
            history.append((it, ev["train"], ev["val"]))
            print(f"  iter {it:5d}  train {ev['train']:.3f}  val {ev['val']:.3f}  "
                  f"({time.time() - start:.0f}s)")

    # --- Save checkpoint ---
    os.makedirs(os.path.join(HERE, "checkpoints"), exist_ok=True)
    ckpt_path = os.path.join(HERE, "checkpoints", "model.pt")
    torch.save({"model": model.state_dict(), "config": vars(cfg), "chars": tokenizer.chars}, ckpt_path)
    print(f"\nsaved checkpoint -> {ckpt_path}")

    # --- Loss curve ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        its = [h[0] for h in history]
        plt.figure(figsize=(7, 4))
        plt.plot(its, [h[1] for h in history], label="train")
        plt.plot(its, [h[2] for h in history], label="val")
        plt.xlabel("iteration"); plt.ylabel("cross-entropy loss")
        plt.title("Tiny GPT training on molecule strings"); plt.legend()
        out = os.path.join(HERE, "assets", "loss_curve.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.savefig(out, dpi=120, bbox_inches="tight")
        print(f"saved loss curve -> {out}")
    except ImportError:
        print("matplotlib not installed; skipping loss curve.")


if __name__ == "__main__":
    main()
