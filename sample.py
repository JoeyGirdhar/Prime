"""
Generate molecules from the trained model and measure what it learned.

Reports three numbers:
  Validity  -- % of generated strings that are structurally valid (balanced
               parens, paired ring digits). This is the emergent-structure metric.
  Novelty   -- % of valid generations NOT present in the training corpus
               (proof it's generalizing, not memorizing).
  Unique    -- % distinct among valid generations.

    python sample.py            # generate 200 molecules and score them
    python sample.py -n 50      # generate fewer, print them all
"""

import argparse
import os

import torch

from tinygpt import GPT, GPTConfig, CharTokenizer, generate_corpus, is_valid

HERE = os.path.dirname(os.path.abspath(__file__))


def load_model(device):
    ckpt = torch.load(os.path.join(HERE, "checkpoints", "model.pt"), map_location=device)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tokenizer = CharTokenizer(ckpt["chars"])
    return model, tokenizer


def sample_molecules(model, tokenizer, n, device, temperature=0.9):
    nl = tokenizer.stoi["\n"]
    mols = []
    for _ in range(n):
        start = torch.tensor([[nl]], dtype=torch.long, device=device)   # begin at newline
        out = model.generate(start, max_new_tokens=48, temperature=temperature,
                             top_k=len(tokenizer.chars), stop_token=nl)
        s = tokenizer.decode(out[0].tolist()).strip("\n")
        if s:
            mols.append(s)
    return mols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(device)

    training_set = set(generate_corpus(n=4000, seed=0))   # same corpus the model trained on
    mols = sample_molecules(model, tokenizer, args.n, device, args.temperature)

    valid = [m for m in mols if is_valid(m)]
    novel = [m for m in valid if m not in training_set]
    unique = set(valid)

    print(f"\nGenerated {len(mols)} molecules (temperature {args.temperature}):")
    print(f"  Valid   : {len(valid)}/{len(mols)} ({len(valid)/len(mols)*100:.0f}%)")
    print(f"  Unique  : {len(unique)}/{len(valid)} of the valid ones"
          if valid else "  Unique  : n/a")
    print(f"  Novel   : {len(novel)}/{len(valid)} not seen in training"
          if valid else "  Novel   : n/a")

    print("\nSample of valid, novel molecules the model invented:")
    for m in novel[:15]:
        print(f"  {m}")
    if not novel:
        for m in valid[:15]:
            print(f"  {m}")


if __name__ == "__main__":
    main()
