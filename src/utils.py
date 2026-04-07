import warnings
from torch.autograd.functional import jacobian
import torch


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
    except RuntimeError as e:
        warnings.warn(
            f"SVD failed to converge during Jacobian norm computation. Returning NaN. Details: {e}"
        )
        norm = float("nan")

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
