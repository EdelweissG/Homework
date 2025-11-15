import numpy as np

def bass_model(t, p, q, M):
    """Bass diffusion model cumulative adoption function."""
    return M * (1 - np.exp(-(p + q) * t)) / (1 + (q / p) * np.exp(-(p + q) * t))

def bass_cumulative(t, p, q, M):
    """Cumulative adoption using Bass model."""
    return M * (1 - np.exp(-(p + q) * t)) / (1 + (q / p) * np.exp(-(p + q) * t))

def bass_yearly(t, p, q, M):
    """Yearly adoption from cumulative adoption."""
    return bass_cumulative(t, p, q, M) - bass_cumulative(t-1, p, q, M)

