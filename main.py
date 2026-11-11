"""Train, then sample."""

import subprocess
import sys

if __name__ == "__main__":
    subprocess.run([sys.executable, "train.py"], check=True)
    subprocess.run([sys.executable, "sample.py"], check=True)
