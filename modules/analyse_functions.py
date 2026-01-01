"""
Fourier- and energy-based analysis functions for 1D density simulations.

This module provides tools to quantify and classify spatial density 
distributions ρ(x) defined on a 1D periodic domain (torus) of length L:

- `density_modulation_amplitude`: Computes the amplitude (order parameter) 
  of the dominant nonzero Fourier mode, characterizing the strength of 
  density modulations.
- `classify_behavior_energy`: Classifies the long-time behavior of a 
  simulation based on the evolution of entropy and interaction energies, 
  returning a discrete code describing diffusion, clustering, or monotone 
  evolution.

Functions
---------
density_modulation_amplitude(rho, L=L, dx=dx, N=x_nodes)
    Compute the amplitude and index of the dominant Fourier mode in a 1D density.

classify_behavior_energy(entropy, interaction)
    Classify simulation behavior using the temporal evolution of energy terms.

Notes
-----
- All functions assume a 1D periodic domain (torus) with uniform grid spacing.
- The `density_modulation_amplitude` function normalizes the input density 
  automatically to ensure unit integral.
- The `classify_behavior_energy` function uses the second-to-last time step 
  to avoid boundary artifacts and treats very small derivatives (<1e-10) 
  as zero for numerical stability.
- Typical use case: analyzing output from finite-difference simulations of 
  the McKean–Vlasov equation.
"""

import numpy as np

from modules.aux_functions import *


def classify_behavior_energy(entropy, interaction):
    """
    Classify simulation behavior using only the energy terms over time.

    Parameters
    ----------
    entropy : ndarray
        Time series of the entropy term from the free energy functional.
    interaction : ndarray
        Time series of the interaction term from the free energy functional.

    Returns
    -------
    result_code : int
        Classification code:
        0 : Clustering (interaction dominates at final time).
        1 : Clustering (interaction dominates, but sign changes observed in energies).
        2 : Diffusion (entropy dominates, but sign changes observed in energies).
        3 : Diffusion (entropy dominates, monotone evolution, no sign changes).

    Notes
    -----
    - Evaluates monotonicity and number of sign changes in the temporal derivatives
      of entropy and interaction energies to distinguish behaviors.
    - Very small derivatives (<1e-10) are treated as zero for numerical stability.
    - Uses the second-to-last time step to determine the dominant energy.
    """
    entropy_dt = df_dt_1d(entropy)[1:-1]
    interaction_dt = df_dt_1d(interaction)[1:-1]

    entropy_dt = np.where(np.abs(entropy_dt) < 1e-10, 0, entropy_dt)
    interaction_dt = np.where(np.abs(interaction_dt) < 1e-10, 0, interaction_dt)

    entropy_sgn_changes = len(find_idx_sign_changes(entropy_dt))
    interaction_sgn_changes = len(find_idx_sign_changes(interaction_dt))

    diffusion=None
    clustering=None
    monotony=None

    result_code = None

    if entropy_sgn_changes==0 and interaction_sgn_changes==0: 
        monotony=True    
        if entropy[-2] < interaction[-2]:
            diffusion=True
            clustering=False
            result_code=3
        elif entropy[-2] > interaction[-2]:
            clustering=True
            diffusion=False
            result_code=0
    else:
        monotony = False
        if entropy[-2] < interaction[-2]:
            diffusion=True
            clustering=False
            result_code=2
        elif entropy[-2] > interaction[-2]:
            clustering=True
            diffusion=False
            result_code=1

    assert result_code!= None
    return result_code

def density_modulation_amplitude(rho, L=L, dx=dx, N=x_nodes):
    """
    Compute the density modulation amplitude (order parameter) for a 1D density on the torus.

    Parameters
    ----------
    rho : array_like
        1D array of density values sampled uniformly over the torus [0, L).
        Should integrate to 1 over the domain, but the function will normalize if needed.

    Returns
    -------
    A : float
        Density modulation amplitude = |rho_hat[m_star]| for the dominant nonzero Fourier mode.
    m_star : int
        Integer index of the dominant mode (1 <= m_star <= N/2).
    """
    rho = np.asarray(rho, dtype=np.float64)

    # Ensure normalization ∫ rho dx = 1
    rho /= rho.sum() * dx

    # Compute Fourier coefficients with physical normalization
    rho_hat = np.fft.fft(rho) * dx

    # Find dominant nonzero mode
    amplitudes = np.abs(rho_hat)
    amplitudes[0] = 0.0  # ignore uniform mode
    m_star = np.argmax(amplitudes[:N//2])
    A = amplitudes[m_star]

    return A, m_star