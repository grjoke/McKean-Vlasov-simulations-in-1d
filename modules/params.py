"""
Simulation parameters for the finite-difference solution of the 1D 
McKean–Vlasov equation.

This module defines all key parameters for spatial and temporal 
discretization, interaction forces, noise amplitude, and plotting 
for numerical solutions of 1D-McKean-Vlasov equation for approximation 
of interacting particle systems.

Parameters
----------
L : float
    Length of the spatial domain.
c_a, c_r : float
    Strength of the attractive and repulsive interactions, respectively.
l_a, l_r : float
    Characteristic length scales for attraction and repulsion.
x_nodes : int
    Number of spatial grid points.
dx : float
    Spatial grid spacing.
X : ndarray
    Array of spatial coordinates, symmetric around zero.
t_end : float
    Total simulation time.
dt : float
    Time step for the simulation.
t_nodes : int
    Number of temporal nodes.
checkpoint : float
    Interval used for progress display during simulations.
sigma : float
    Standard deviation of the noise (amplitude) in the Langevin dynamics.
plot_dir : str
    Directory path for saving plots.

Notes
-----
- The spatial grid X is defined using a uniform discretization with x_nodes points.
- Interaction parameters (c_a, c_r, l_a, l_r) correspond to a Morse-type potential.
- The module is intended to be imported in simulation scripts to ensure
  consistent use of parameters across functions and modules.
"""

import numpy as np

# potential parameters
L = 5 
c_a = 4
c_r = 1
l_a = 0.025*L
l_r = 0.01*L


## parameters for discretization
x_nodes = pow(2,8) #90
dx = L / x_nodes
X = np.linspace(-L/2, L/2, x_nodes, endpoint=False)

# temporal parameters
t_end = 10
dt = 1*1e-4 
t_nodes = int(t_end/dt)

# value needed for progress display
checkpoint = t_nodes/10


sigma = 1.0
plot_dir = 'Plots_McKeanVlasov_1d/'
