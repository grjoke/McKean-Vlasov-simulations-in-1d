"""
Free-energy evaluation for 1D McKean–Vlasov simulations.

This module provides functions to compute the free energy and its components
(entropy and interaction energy) for trajectories of a 1D density evolving
according to a McKean–Vlasov-type equation on a periodic domain.

The free energy functional is given by

    F[ρ] = (σ² / 2) ∫ ρ(x) log ρ(x) dx
           + (1 / 2) ∫∫ W(x - y) ρ(x) ρ(y) dx dy,

where W is a periodic Morse interaction potential.

The input trajectory is assumed to be provided as a 2D array of shape
(x_nodes, t_nodes), representing the density ρ(x, t) sampled on a uniform
spatial grid over time.

Functions
---------
calc_free_energy
    Compute free, entropy, and interaction energy for a full trajectory.

calc_free_energy_neg
    Variant of `calc_free_energy` that regularizes negative density values
    inside the logarithm.

calc_free_energy_unif
    Compute the free energy of the uniform steady state.

Notes
-----
- Periodic boundary conditions are enforced via convolution with `mode='wrap'`.
- All spatial integrals are approximated using the trapezoidal rule on a
  uniform grid.
- The entropy term uses the convention (σ² / 2) ∫ ρ log ρ dx.
"""

import numpy as np
from scipy.ndimage import  convolve, convolve1d

from modules.params import *
from modules.aux_functions import norm_array
from modules.potential import int_potential


def calc_free_energy(density, sigma=sigma, c_a = c_a, l_a = l_a, c_r = c_r, l_r = l_r):
    """
    Compute the free energy and its components for a density trajectory.

    Parameters
    ----------
    density : ndarray
        Density trajectory of shape (x_nodes, t_nodes), where each column
        represents the density ρ(x, t_i) at a fixed time.
    sigma : float, optional
        Noise amplitude appearing in the entropy term.
    c_a, l_a : float, optional
        Strength and length scale of the attractive part of the interaction.
    c_r, l_r : float, optional
        Strength and length scale of the repulsive part of the interaction.

    Returns
    -------
    free_energy : ndarray
        Total free energy F(t) at each time step.
    entropy_energy : ndarray
        Entropy contribution (σ² / 2) ∫ ρ log ρ dx at each time step.
    interaction_energy : ndarray
        Interaction energy (1 / 2) ∫ ρ (W * ρ) dx at each time step.

    Notes
    -----
    - Assumes the density is nonnegative everywhere.
    - Uses periodic convolution to evaluate the interaction term.
    """

    interaction_pot = int_potential(X, c_a = c_a, l_a = l_a, c_r = c_r, l_r = l_r)
    
    t_nodes = density.shape[1]
    interaction_energy = np.zeros(t_nodes)
    entropy_energy = np.zeros(t_nodes) 

    for i in range(t_nodes):
        cur_dens = density[:,i]
        conv = convolve1d(interaction_pot, cur_dens, mode='wrap')
        conv = conv*dx
        u_star = cur_dens*conv
        
        log_dens = np.log(cur_dens)
        interaction_energy[i] = 0.5 * dx * np.sum(u_star)
        entropy_energy[i] = 0.5*sigma**2 * dx * np.sum(cur_dens*log_dens)

    free_energy = entropy_energy + interaction_energy
    return free_energy, entropy_energy, interaction_energy

# if trajectory has negative values (which should be avoided) this function can be used, which cancels these values out
def calc_free_energy_neg(density, sigma=sigma, c_a = c_a, l_a = l_a, c_r = c_r, l_r = l_r):
    """
    Compute the free energy for trajectories that may contain negative values.

    This function regularizes the entropy term by replacing negative density
    values inside the logarithm, while keeping the interaction term unchanged.

    Parameters
    ----------
    density : ndarray
        Density trajectory of shape (x_nodes, t_nodes).
    sigma : float, optional
        Noise amplitude appearing in the entropy term.
    c_a, l_a : float, optional
        Strength and length scale of the attractive interaction.
    c_r, l_r : float, optional
        Strength and length scale of the repulsive interaction.

    Returns
    -------
    free_energy : ndarray
        Total free energy at each time step.
    entropy_energy : ndarray
        Regularized entropy contribution.
    interaction_energy : ndarray
        Interaction energy contribution.

    Notes
    -----
    - Negative density values are replaced by 1 inside the logarithm only.
    - This function is intended for diagnostic purposes when numerical
      instabilities produce small negative densities.
    """

    interaction_pot = int_potential(X, c_a = c_a, l_a = l_a, c_r = c_r, l_r = l_r)

    t_nodes = density.shape[1]
    interaction_energy = np.zeros(t_nodes)
    entropy_energy = np.zeros(t_nodes) 

    for i in range(t_nodes):
        cur_dens = density[:,i]
        conv = convolve1d(interaction_pot, cur_dens, mode='wrap')
        conv = conv*dx
        u_star = cur_dens*conv
        
        cur_dens_log = np.where(cur_dens>=0, cur_dens, 1)
        log_dens = np.log(cur_dens_log)
        interaction_energy[i] = 0.5 * dx * np.sum(u_star)
        entropy_energy[i] = 0.5*sigma**2 * dx * np.sum(cur_dens*log_dens)

    free_energy = entropy_energy + interaction_energy
    return free_energy, entropy_energy, interaction_energy


# function to calculate the free energy of a uniform distribution
def calc_free_energy_unif(sigma=sigma, c_a = c_a, l_a = l_a, c_r = c_r, l_r = l_r, dx=dx):
    """
    Compute the free energy of the uniform density state.

    The uniform density is defined as ρ(x) = 1 / L on the periodic domain.

    Parameters
    ----------
    sigma : float, optional
        Noise amplitude appearing in the entropy term.
    c_a, l_a : float, optional
        Strength and length scale of the attractive interaction.
    c_r, l_r : float, optional
        Strength and length scale of the repulsive interaction.
    dx : float, optional
        Spatial grid spacing.

    Returns
    -------
    free_energy_unif : float
        Total free energy of the uniform state.
    entropy_unif : float
        Entropy contribution of the uniform state.
    interaction_unif : float
        Interaction energy of the uniform state.

    Notes
    -----
    - The uniform density is normalized explicitly using `norm_array`.
    - Useful as a reference energy level for stability and phase-transition
      analysis.
    """

    interaction_pot = int_potential(X, c_a = c_a, l_a = l_a, c_r = c_r, l_r = l_r)
    u_unif = np.ones(x_nodes, dtype=np.float64)
    u_unif = norm_array(u_unif, dx)

    log_unif = np.log(u_unif)
    entropy_unif = 0.5*sigma**2 * dx * np.sum(u_unif*log_unif)

    conv = convolve1d(interaction_pot, u_unif, mode='wrap')
    conv = conv*dx
    u_star = u_unif*conv
    interaction_unif = 0.5 * dx * np.sum(u_star)
    
    free_energy_unif = interaction_unif + entropy_unif
    return free_energy_unif, entropy_unif, interaction_unif