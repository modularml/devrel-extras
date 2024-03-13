import numpy as np
import torch


def main():
    logits = torch.from_numpy(np.fromfile("outputs.bin", dtype=np.float32))
    scores = logits.softmax(dim=-1)
    print(f"Scores: {scores.numpy()}")


if __name__ == "__main__":
    main()
