import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.model import IterativeTransformerBlock
from src.utils import (
    setup_experiment,
    compute_jacobian_spectral_norm,
    compute_theoretical_bound,
)


def main():
    save_dir = setup_experiment(seed=42)
    print("Running Experiment 1: Theoretical Bounds Validation...")

    S, D, H = 16, 32, 4
    ffn_hidden = 64
    nu_values = np.linspace(0.1, 2.0, 10)

    emp_norms = []
    theo_bounds = []

    model = IterativeTransformerBlock(D, H, ffn_hidden)

    for nu in tqdm(nu_values):
        model.nu = nu
        Y = torch.randn(1, S, D) * 0.5
        C = torch.randn(1, S, D) * 0.1

        emp = compute_jacobian_spectral_norm(model, Y, C)
        theo = compute_theoretical_bound(model, Y, C)

        emp_norms.append(emp)
        theo_bounds.append(theo)

    plt.figure()
    plt.plot(nu_values, emp_norms, "o-", label="Empirical ||J||₂", color="#1f77b4")
    plt.plot(nu_values, theo_bounds, "--", label="Theoretical Bound", color="#ff7f0e")
    plt.xlabel(r"Step size $\nu$")
    plt.ylabel("Spectral Norm")
    plt.title("Validation of Theorem 3.1")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = f"{save_dir}/fig1_bounds_validation.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved Figure 1 to {save_path}")


if __name__ == "__main__":
    main()
