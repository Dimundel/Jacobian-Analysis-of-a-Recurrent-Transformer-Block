import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd.functional import jacobian
from tqdm import tqdm
import os

# 1. SETUP

torch.manual_seed(42)
np.random.seed(42)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.figsize": (10, 6),
        "lines.linewidth": 2.5,
        "lines.markersize": 8,
        "grid.alpha": 0.3,
    }
)

SAVE_DIR = "paper_results"
os.makedirs(SAVE_DIR, exist_ok=True)

# 2. MODEL DEFINITION


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8, gamma: float = 1.0):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim) * gamma)

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return self.gamma * x / rms

    def get_gamma_max(self):
        return torch.max(torch.abs(self.gamma)).item()


class IterativeTransformerBlock(nn.Module):
    """
    Iterative Transformer Block (Post-Norm architecture).
    Matches Eqs (3.11)-(3.15) and Theorem 3.1 in the paper.
    """

    def __init__(self, dim: int, num_heads: int, ffn_hidden_dim: int, nu: float = 0.5):
        super().__init__()
        self.nu = nu
        self.dim = dim

        self.msa = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_hidden_dim), nn.GELU(), nn.Linear(ffn_hidden_dim, dim)
        )

        self.rms1 = RMSNorm(dim)
        self.rms2 = RMSNorm(dim)
        self.rms_final = RMSNorm(dim)

    def forward_subblocks(self, Y):
        """
        Executes the TB(Y) logic:
        1. Y' = RMS(Y + MSA(Y))
        2. TB(Y) = RMS(Y' + FFN(Y'))
        """
        attn_out, _ = self.msa(Y, Y, Y)
        Y_prime = self.rms1(Y + attn_out)

        ffn_out = self.ffn(Y_prime)
        TB_Y = self.rms2(Y_prime + ffn_out)

        return TB_Y

    def forward_step(self, Y, C):
        """
        Single recurrent step:
        Y(t+1) = RMS(Y(t) + nu * (C + TB(Y(t))))
        """
        if C.dim() == 2:
            C = C.unsqueeze(0).expand(Y.shape[0], -1, -1)

        TB_Y = self.forward_subblocks(Y)
        U = Y + self.nu * (C + TB_Y)
        Y_next = self.rms_final(U)  # Eq 3.15
        return Y_next


# 3. UTILITY FUNCTIONS


def compute_jacobian_spectral_norm(model, Y, C):
    """Computes ||J_F(Y)||_2 via autograd"""
    Y_flat = Y.detach().clone().requires_grad_(True)

    def func(y_flat):
        y_reshaped = y_flat.view(Y.shape)
        out = model.forward_step(y_reshaped, C)
        return out.view(-1)

    J = jacobian(func, Y_flat.view(-1))

    try:
        norm = torch.linalg.matrix_norm(J, ord=2).item()
    except:
        norm = torch.linalg.norm(J).item()

    return norm


def compute_theoretical_bound(model, Y, C, M=1.0, F=1.0):
    """
    Computes the bound from Theorem 3.1.
    Note: We use M=1, F=1 as conservative Lipschitz estimates for MSA/FFN layers
    initialized with standard scaling.
    """
    with torch.no_grad():
        TB_Y = model.forward_subblocks(Y)

        attn_out, _ = model.msa(Y, Y, Y)
        input_rms1 = Y + attn_out
        r_in = torch.sqrt(torch.mean(input_rms1**2, dim=-1)).mean().item()

        Y_prime = model.rms1(input_rms1)
        ffn_out = model.ffn(Y_prime)
        input_rms2 = Y_prime + ffn_out
        r_out = torch.sqrt(torch.mean(input_rms2**2, dim=-1)).mean().item()

        U = Y + model.nu * (C + TB_Y)
        r_U = torch.sqrt(torch.mean(U**2, dim=-1)).mean().item()

        gamma_max = model.rms_final.get_gamma_max()

        eps = 1e-6
        bound = (gamma_max / (r_U + eps)) * (
            1
            + model.nu
            * (gamma_max**2 / ((r_out + eps) * (r_in + eps)))
            * (1 + F)
            * (1 + M)
        )
        return bound


# 4. EXPERIMENTS


def run_experiment_1_bounds():
    """Figure 1: Validation of Theoretical Bounds"""

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
    plt.savefig(f"{SAVE_DIR}/fig1_bounds_validation.png", dpi=300)
    plt.close()


def run_experiment_2_asymptotic():
    """Figure 2: Asymptotic Behavior (O(1))"""

    S, D, H = 16, 32, 4
    ffn_hidden = 64
    nu_values = np.logspace(-1, 5, 20)

    norms = []
    stds = []

    for nu in tqdm(nu_values):
        model = IterativeTransformerBlock(D, H, ffn_hidden, nu=nu)

        batch_norms = []
        for _ in range(10):
            Y = torch.randn(1, S, D) * 0.5
            C = torch.randn(1, S, D) * 0.1
            batch_norms.append(compute_jacobian_spectral_norm(model, Y, C))

        norms.append(np.mean(batch_norms))
        stds.append(np.std(batch_norms))

    norms = np.array(norms)
    stds = np.array(stds)

    plt.figure()
    plt.plot(nu_values, norms, "o-", color="#1f77b4")
    plt.fill_between(nu_values, norms - stds, norms + stds, alpha=0.3, color="#1f77b4")
    plt.xscale("log")
    plt.xlabel(r"Step size $\nu$ (log scale)")
    plt.ylabel(r"Spectral Norm $\|J_F\|_2$")
    plt.title(r"Asymptotic Stability ($O(1)$) as $\nu \to \infty$")
    plt.axhline(y=norms[-1], color="r", linestyle="--", label="Asymptote", alpha=0.5)
    plt.grid(True, which="both", ls="-")
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/fig2_asymptotic.png", dpi=300)
    plt.close()


def load_cifar10_sample(batch_size=16):
    """Load CIFAR-10 images and preprocess them into (S, D) sequences."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    images, _ = next(iter(loader))
    B, C, H, W = images.shape
    S = H
    D = W * C
    sequences = images.permute(0, 2, 3, 1).reshape(B, S, D)  # (B, 32, 96)
    return sequences


def run_experiment_3_contraction():
    """Figure 3: Contraction Dynamics (Synthetic vs CIFAR)"""

    S, D, H = 32, 64, 4
    ffn_hidden = 128
    steps = 1000
    nu_values = [0.1, 0.5, 1.0]

    def run_dynamics_on_input(model, x_init):
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

    torch.manual_seed(228)
    x_synth = torch.randn(1, S, D)

    model_synth = IterativeTransformerBlock(D, H, ffn_hidden)
    model_synth.eval()

    res_synth = run_dynamics_on_input(model_synth, x_synth)

    x_cifar_raw = load_cifar10_sample(batch_size=1)
    x_cifar_raw = x_cifar_raw / x_cifar_raw.std()

    D_input = x_cifar_raw.shape[-1]
    torch.manual_seed(100)
    projector = nn.Linear(D_input, D)

    with torch.no_grad():
        x_cifar = projector(x_cifar_raw)

    model_cifar = IterativeTransformerBlock(D, H, ffn_hidden)
    model_cifar.eval()
    res_cifar = run_dynamics_on_input(model_cifar, x_cifar)

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
    plt.savefig(f"{SAVE_DIR}/fig3_contraction.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    run_experiment_1_bounds()
    run_experiment_2_asymptotic()
    run_experiment_3_contraction()
    print(f"Done! Results saved to {SAVE_DIR}/")
