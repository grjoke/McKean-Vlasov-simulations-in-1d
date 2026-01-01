'''
Definition of several metrics or measures for probability densities
'''

import numpy as np
from scipy.stats import wasserstein_distance

from modules.params import *

def mean(array, domain=X, dx=dx):
    mean = np.sum(domain*array*dx)
    return mean

def variance(array, domain=X, dx=dx):
    mean_c = mean(array, domain)
    variance = np.sum((domain-mean_c)**2 * array * dx)
    return variance

def shannon_entropy(density, dx=dx):
    entropy = -np.sum(density*np.log(density)*dx)
    return entropy

def gini_coefficient(density):
    """Compute the Gini coefficient for a given density distribution."""
    N = len(density)
    if N == 0:
        return 0  # Edge case for empty array
    
    density_sorted = np.sort(density)  # Step 2: Sort values
    mean_density = np.mean(density)    # Step 3: Compute mean density
    # Step 4: Compute Gini numerator
    numerator = np.sum(np.abs(density_sorted[:, None] - density_sorted[None, :]))
    # Step 5: Normalize
    G = numerator / (2 * N**2 * mean_density)
    
    return G

def participation_ratio(density):
    density_sq = density*density
    localization = np.sum(density_sq*dx)
    pr = 1/localization
    return pr

def kl_divergence(density, ref_density, dx=dx):
    np.where(ref_density >=0, ref_density, 1)
    kl_div = np.sum(density * np.log(density/ref_density) * dx)
    return kl_div

def peak_to_trough_ratio(density):
    u_max = np.max(density)
    u_min = np.min(density)
    ptt_ratio = u_min/u_max
    return ptt_ratio

    
def wasserstein_1(density, ref_density):
    return wasserstein_distance(X, X, density, ref_density)

def wasserstein_periodic(density, ref_density):
    step = 1
    distances = []
    for k in range(0, len(X), step):
        dist = wasserstein_distance(X, X, u_weights=np.roll(density, k), v_weights=np.roll(ref_density, k))
        distances.append(dist)
    best_shift_index = np.argmin(distances)
    dist = distances[best_shift_index]

    return dist

def energy_dist(density, ref_density):
    return energy_distance(X, X, density, ref_density)