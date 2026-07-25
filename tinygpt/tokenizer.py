"""
Character-level tokenizer.

The "vocabulary" is just the set of characters that appear in the corpus plus a
newline that separates molecules (and doubles as the start/stop token). Tiny
vocab (~15 symbols) is part of why this trains fast on a CPU.
"""


class CharTokenizer:
    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}

    @classmethod
    def from_corpus(cls, molecules, sep="\n"):
        text = sep.join(molecules) + sep
        return cls(text), text

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)
