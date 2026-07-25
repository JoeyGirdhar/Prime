"""
Entry point. Trains the tiny GPT, then samples molecules and scores them.

    python main.py

Or run the steps separately:
    python train.py --iters 3000     # train (saves checkpoints/model.pt)
    python sample.py                 # generate + measure validity
    python tests/test_tinygpt.py     # fast tests, no training needed
"""

import subprocess
import sys

if __name__ == "__main__":
    subprocess.run([sys.executable, "train.py"], check=True)
    subprocess.run([sys.executable, "sample.py"], check=True)
