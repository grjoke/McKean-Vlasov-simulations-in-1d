'''
Core fuction to simulate the evolution of a propability density, which behaves accordingly to the McKean-Vlasov equation 
'''


import numpy as np
from scipy.ndimage import  convolve, convolve1d
from copy import deepcopy

from modules.params import *
from modules.aux_functions import *
from modules.potential import *


def model_simulation(u_init, sigma=sigma, t_end = t_end, c_a = c_a, l_a = l_a, c_r = c_r, l_r = l_r):
    """
    Run a numerical simulation of the 1D McKean–Vlasov (Fokker–Planck) equation
    with a Morse interaction potential.

    The evolution is computed via an explicit Euler time-stepping scheme with
    a convolution-based evaluation of the nonlocal interaction term. The PDE
    solved is of aggregation–diffusion type:

        ∂_t u = D ∂_xx u + ∂_x( u * (W' * u) ),

    where ``W`` is a Morse potential and ``W'`` its derivative (the interaction force).

    Parameters
    ----------
    u_init : ndarray of shape (Nx,)
        Initial density distribution at time ``t=0``. Must be defined on the
        same spatial grid as the global variable ``X``.
    sigma : float, optional
        Noise strength (diffusion parameter). The diffusion coefficient in the
        PDE is ``D = sigma**2 / 2``. Default is the global ``sigma``.
    t_end : float, optional
        Final simulation time. Determines the number of time steps. 
        Default is global ``t_end``.
    c_a, l_a : float, optional
        Morse attraction strength ``c_a`` and attraction length scale ``l_a``.
    c_r, l_r : float, optional
        Morse repulsion strength ``c_r`` and repulsion length scale ``l_r``.

    Notes
    -----
    - Uses global variables ``X``, ``x_nodes``, ``dx``, ``dt`` as well as the
      functions ``Morse_force``, ``int_potential``, ``df_dx`` and ``df_dx2``.
    - Nonlocal term computed via a periodic convolution using ``scipy.ndimage.convolve1d``.
    - Time stepping uses explicit Euler; choose ``dt`` small enough for stability.
    - Mass is not explicitly renormalized; instabilities may occur for large ``sigma`` 
      or inappropriate ``dt``.

    Returns
    -------
    u_store_fp : ndarray of shape (Nx, Nt+1)
        Time evolution of the density. Column ``u_store_fp[:, j]`` corresponds
        to time step ``j``.
    vmax_time : float
        Approximate physical time at which the global maximum of ``u(x,t)`` occurs.
    vmax : float
        Maximum value of the density over the entire simulation.
    vmax_x : int
        Time-index at which the maximum density value occurs.
    u_max : ndarray of shape (Nx,)
        Density profile at the time step where the global maximum ``vmax`` occurs.
    u_end : ndarray of shape (Nx,)
        Density profile near the final simulation time (third-to-last stored column).

    """
    
    interaction_force = int_force(X, c_a = c_a, l_a = l_a, c_r = c_r, l_r = l_r)
    interaction_pot = int_potential(X, c_a = c_a, l_a = l_a, c_r = c_r, l_r = l_r)

    t_nodes = int(t_end/dt)
    t = np.linspace(0, t_end, t_nodes)

    u_store_fp = np.zeros((x_nodes, t_nodes+1))
    checkpoint = t_nodes/10

    u = u_init

    counter_t = 0
    counter_int = 0

    while counter_t < t_end:
        u_store_fp[:,counter_int] = u

        w = deepcopy(u)

        conv = convolve1d(w, interaction_force, mode='wrap')
        u_star = w * conv

        # Mathematically convolving and differenciating can be switched, so it does not matter 
        # if the potential is first derived (results in interaction force) 
        # and then convolved or the other way around
     
        #conv = convolve1d(interaction_pot, w, mode='wrap')
        #conv_dx = df_dx(conv) / dx
        #u_star = (w * conv_dx)

        u_heat = dt * (0.5*sigma**2.) * df_dx2(w, dx)
        u_conv = dt * df_dx(u_star) 
        u = u_heat + w + u_conv  
                
        counter_t += dt
        counter_int += 1


        ### Progress bar, can be commented in, if needed
        #if counter_int % int(checkpoint) == 0:
        #    print(np.round(counter_int/t_nodes, 2)*100, '%') 

    ## Retrieve some data out of the stored trajectory
    vmax_beginning = np.max(u_store_fp[:,0])
    vmax = np.max(u_store_fp)
    vmin = np.min(u_store_fp)

    vmax_x = np.where(u_store_fp == vmax)[1][0]
    if vmax_x == 0: 
        vmax_time = np.round(t[vmax_x], 2)
    else: 
        vmax_time = np.round(t[vmax_x-1], 2)

    u_max = u_store_fp[:, vmax_x]
    u_end = u_store_fp[:,-3]

    return u_store_fp, vmax_time, vmax, vmax_x, u_max, u_end