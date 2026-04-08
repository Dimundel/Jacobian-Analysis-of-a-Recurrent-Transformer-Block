import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.model import IterativeTransformerBlock
from src.utils import setup_experiment
from src.data import load_cifar10_sample


def run_dynamics_on_input(model, x_init, nu_values, steps=1000):
    """Helper function to run recurrent dynamics and track contraction."""
    results = {}
    for nu in nu_values:
        model.nu = nu
        Y = x_init.clone()
        C = x_init.clone() * 0.1

        diffs = []
        with torch.no_grad():
            for _ in range(steps):
                Y_next = model.forward_step(Y, C)
                diffs.append(torch.norm(Y_next - Y).item())
                Y = Y_next
        results[nu] = diffs
    return results


def main():
    save_dir = setup_experiment(seed=42)
    print("Running Experiment 3: Contraction Dynamics (Synthetic vs CIFAR-10)...")

    S, D, H = 32, 64, 4
    ffn_hidden = 128
    steps = 1000
    nu_values = [0.1, 0.5, 1.0]

    torch.manual_seed(228)
    x_synth = torch.randn(1, S, D)

    model_synth = IterativeTransformerBlock(D, H, ffn_hidden)
    model_synth.eval()

    print("Computing dynamics for Synthetic data...")
    res_synth = run_dynamics_on_input(model_synth, x_synth, nu_values, steps)

    print("Loading CIFAR-10 data...")
    x_cifar_raw = load_cifar10_sample(batch_size=1)
    x_cifar_raw = x_cifar_raw / x_cifar_raw.std()

    D_input = x_cifar_raw.shape[-1]

    torch.manual_seed(100)
    projector = nn.Linear(D_input, D)

    with torch.no_grad():
        x_cifar = projector(x_cifar_raw)

    model_cifar = IterativeTransformerBlock(D, H, ffn_hidden)
    model_cifar.eval()

    print("Computing dynamics for CIFAR-10 data...")
    res_cifar = run_dynamics_on_input(model_cifar, x_cifar, nu_values, steps)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for i, nu in enumerate(nu_values):
        ax1.plot(res_synth[nu], label=f"$\\nu={nu}$", color=colors[i])
    ax1.set_yscale("log")
    ax1.set_title("Synthetic Data")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel(r"Contraction $\|Y_{t+1} - Y_t\|$")
    ax1.grid(True)
    ax1.legend()

    for i, nu in enumerate(nu_values):
        ax2.plot(res_cifar[nu], label=f"$\\nu={nu}$", color=colors[i])
    ax2.set_yscale("log")
    ax2.set_title("Structured Data (CIFAR-10)")
    ax2.set_xlabel("Iteration")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()

    save_path = f"{save_dir}/fig3_contraction.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved Figure 3 to {save_path}")


if __name__ == "__main__":
    main()
