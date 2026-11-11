"""SMILES-style molecule strings: a random generator and a validity check.

Validity is structural only (allowed characters, balanced branches, paired
ring digits), not chemical."""

import random

ATOMS = ["C", "C", "C", "N", "O", "F", "S"]
BONDS = ["", "", "", "=", "#"]
ALLOWED = set("CNOFS()=#123456789")


def _random_chain(rng, length, open_rings):
    """Emit a short chain of atoms with optional bonds, branches, and ring digits."""
    out = []
    for i in range(length):
        if i > 0:
            out.append(rng.choice(BONDS))
        out.append(rng.choice(ATOMS))
        if rng.random() < 0.18:
            if open_rings and rng.random() < 0.55:
                out.append(str(open_rings.pop()))
            elif len(open_rings) < 3:
                digit = rng.randint(1, 6)
                out.append(str(digit))
                open_rings.append(digit)
        if rng.random() < 0.15 and length - i > 1:
            out.append("(")
            out.extend(_random_chain(rng, rng.randint(1, 2), open_rings))
            out.append(")")
    return out


def generate_molecule(rng) -> str:
    open_rings: list[int] = []
    tokens = _random_chain(rng, rng.randint(4, 12), open_rings)
    # close any rings left open
    for digit in open_rings:
        tokens.append(str(digit))
    return "".join(tokens)


def generate_corpus(n=4000, seed=0) -> list[str]:
    rng = random.Random(seed)
    seen, mols = set(), []
    while len(mols) < n:
        m = generate_molecule(rng)
        if 3 <= len(m) <= 40 and m not in seen and is_valid(m):
            seen.add(m)
            mols.append(m)
    return mols


def is_valid(s: str) -> bool:
    """Structural validity: allowed chars, balanced parens, paired ring digits, no dangling bonds."""
    if not s or any(ch not in ALLOWED for ch in s):
        return False
    if s[0] in "()=#" or s[-1] in "(=#":
        return False
    depth = 0
    ring_counts: dict[str, int] = {}
    prev_bond = False
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return False
        elif ch in "=#":
            if prev_bond:
                return False
            prev_bond = True
            continue
        elif ch.isdigit():
            ring_counts[ch] = ring_counts.get(ch, 0) + 1
        prev_bond = False
    if depth != 0:
        return False
    # each ring digit has to be opened and closed
    return all(count % 2 == 0 for count in ring_counts.values())
