import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.model import IterativeTransformerBlock
from src.utils import setup_experiment, compute_jacobian_spectral_norm


def main():
    save_dir = setup_experiment(seed=42)
    print("Running Experiment 2: Asymptotic Behavior (O(1))...")

    S, D, H = 16, 32, 4
    ffn_hidden = 64
    nu_values = np.logspace(-1, 5, 20)

    norms = []
    stds = []

    for nu in tqdm(nu_values, desc="Computing over nu values"):
        model = IterativeTransformerBlock(D, H, ffn_hidden, nu=nu)

        batch_norms = []
        for _ in range(10):
            Y = torch.randn(1, S, D) * 0.5
            C = torch.randn(1, S, D) * 0.1
            batch_norms.append(compute_jacobian_spectral_norm(model, Y, C))

        valid_norms = [n for n in batch_norms if not np.isnan(n)]

        if valid_norms:
            norms.append(np.mean(valid_norms))
            stds.append(np.std(valid_norms))
        else:
            norms.append(float("nan"))
            stds.append(float("nan"))

    norms = np.array(norms)
    stds = np.array(stds)

    plt.figure()
    plt.plot(nu_values, norms, "o-", color="#1f77b4")
    plt.fill_between(nu_values, norms - stds, norms + stds, alpha=0.3, color="#1f77b4")
    plt.xscale("log")
    plt.xlabel(r"Step size $\nu$ (log scale)")
    plt.ylabel(r"Spectral Norm $\|J_F\|_2$")
    plt.title(r"Asymptotic Stability ($O(1)$) as $\nu \to \infty$")

    if not np.isnan(norms[-1]):
        plt.axhline(
            y=norms[-1], color="r", linestyle="--", label="Asymptote", alpha=0.5
        )
        plt.legend()

    plt.grid(True, which="both", ls="-")
    plt.tight_layout()

    save_path = f"{save_dir}/fig2_asymptotic.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved Figure 2 to {save_path}")


if __name__ == "__main__":
    main()
