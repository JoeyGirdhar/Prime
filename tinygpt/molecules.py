"""
The "language" the transformer learns: SMILES-style molecule strings.

SMILES writes a molecule as text -- e.g. ethanol is "CCO", benzene is "c1ccccc1".
We don't need real chemistry here; we need a language with crisp, checkable
structure so we can *measure* whether the model learned it:

  - allowed atoms/bonds/branch/ring characters only
  - parentheses balanced (branches open and close)
  - ring-bond digits paired (a ring you open with "1" you must close with "1")
    -- this is the LONG-RANGE dependency that shows off attention

`generate_corpus` produces strings that are valid by construction (the training
data). `is_valid` scores arbitrary strings (used to grade the model's output).
Real chemical validity would use RDKit; this is deliberately a structural proxy.
"""

import random

ATOMS = ["C", "C", "C", "N", "O", "F", "S"]   # weighted toward carbon, like real molecules
BONDS = ["", "", "", "=", "#"]                 # mostly single bonds
ALLOWED = set("CNOFS()=#123456789")


def _random_chain(rng, length, open_rings):
    """Emit a short chain of atoms with optional bonds, branches, and ring digits."""
    out = []
    for i in range(length):
        if i > 0:
            out.append(rng.choice(BONDS))      # a bond between atoms (often empty = single)
        out.append(rng.choice(ATOMS))
        # Occasionally open OR close a ring bond using a matched digit.
        if rng.random() < 0.18:
            if open_rings and rng.random() < 0.55:
                out.append(str(open_rings.pop()))         # close a ring we opened earlier
            elif len(open_rings) < 3:
                digit = rng.randint(1, 6)
                out.append(str(digit))
                open_rings.append(digit)                  # remember to close it later
        # Occasionally open a short branch (balanced parens).
        if rng.random() < 0.15 and length - i > 1:
            out.append("(")
            out.extend(_random_chain(rng, rng.randint(1, 2), open_rings))
            out.append(")")
    return out


def generate_molecule(rng) -> str:
    open_rings: list[int] = []
    tokens = _random_chain(rng, rng.randint(4, 12), open_rings)
    # Close any rings still open, so the training string is always valid.
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
        return False               # can't start with a bond/close-paren or end mid-bond
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
                return False       # two bond symbols in a row
            prev_bond = True
            continue
        elif ch.isdigit():
            ring_counts[ch] = ring_counts.get(ch, 0) + 1
        prev_bond = False
    if depth != 0:
        return False               # unbalanced parentheses
    # Every ring-bond digit must appear an even number of times (opened and closed).
    return all(count % 2 == 0 for count in ring_counts.values())
