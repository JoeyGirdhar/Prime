"""Shape, validator and overfit tests."""

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tinygpt import GPT, GPTConfig, is_valid, generate_corpus


def test_forward_shapes():
    cfg = GPTConfig(vocab_size=16, block_size=32, n_layer=2, n_head=2, n_embd=32)
    model = GPT(cfg)
    x = torch.randint(0, 16, (4, 32))
    logits, loss = model(x, x)
    assert logits.shape == (4, 32, 16), logits.shape
    assert loss.dim() == 0 and loss.item() > 0
    print("[shapes] OK  logits", tuple(logits.shape))


def test_validator():
    assert is_valid("CCO")
    assert is_valid("C1CCCCC1")
    assert is_valid("CC(C)CO")
    assert not is_valid("CC(C")
    assert not is_valid("C1CC")
    assert not is_valid("C==C")
    assert not is_valid("=CO")
    assert not is_valid("CCX")
    print("[validator] OK")


def test_corpus_all_valid():
    mols = generate_corpus(n=200, seed=1)
    assert len(mols) == 200
    assert all(is_valid(m) for m in mols), "generated corpus must be valid by construction"
    print("[corpus] OK  200 valid molecules")


def test_can_overfit():
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=16, block_size=16, n_layer=2, n_head=2, n_embd=64)
    model = GPT(cfg)
    x = torch.randint(0, 16, (1, 16))
    y = torch.randint(0, 16, (1, 16))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    first = None
    for _ in range(200):
        _, loss = model(x, y)
        if first is None:
            first = loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < first * 0.1, f"failed to overfit: {first:.2f} -> {loss.item():.2f}"
    print(f"[overfit] OK  loss {first:.2f} -> {loss.item():.3f}")


if __name__ == "__main__":
    test_forward_shapes()
    test_validator()
    test_corpus_all_valid()
    test_can_overfit()
    print("\nAll tinygpt tests passed.")
