"""
Potential and force functions for 1D particle interaction simulations.

This module provides the Morse interaction potential and its corresponding
force for 1D aggregation–diffusion or McKean–Vlasov models. The potential 
is periodic, with both attractive and repulsive contributions, and the force 
is its spatial derivative. These functions are typically used to construct 
convolution kernels in numerical simulations.

Functions
---------
int_potential(dist, c_a=c_a, l_a=l_a, c_r=c_r, l_r=l_r)
    Compute the periodic Morse interaction potential at given distances.

int_force(dist, c_a=c_a, l_a=l_a, c_r=c_r, l_r=l_r)
    Compute the derivative of the periodic Morse potential (interaction force),
    with regularization at dist = 0.

Notes
-----
- Both functions assume a 1D periodic domain of length L (from params.py).
- The force uses a regularization at dist = 0, setting W'(0) = 0 for numerical stability.
- Typically, ``dist`` corresponds to the difference between spatial grid points
  when building convolution kernels for interaction terms.
- The Morse potential is normalized using hyperbolic functions to ensure
  periodicity without discontinuities.
"""

import numpy as np

from modules.params import *

def int_potential(dist, c_a = c_a, l_a = l_a, c_r = c_r, l_r = l_r):
    """
    Compute the periodic Morse interaction potential evaluated at distances ``dist``.

    The potential is normalized using hyperbolic functions to impose periodic
    boundary conditions on an interval of length ``L``. The potential has an
    attractive part (parameters ``c_a``, ``l_a``) and a repulsive part 
    (``c_r``, ``l_r``):

        W(x) = -c_a * cosh((|x| - L/2)/l_a) / sinh(L/(2 l_a))
               + c_r * cosh((|x| - L/2)/l_r) / sinh(L/(2 l_r))

    Parameters
    ----------
    dist : ndarray or float
        Distance(s) at which the Morse potential is evaluated. Typically 
        ``dist = X - X[i]`` when constructing convolution kernels.
    c_a : float, optional
        Attraction strength.
    l_a : float, optional
        Attraction length scale.
    c_r : float, optional
        Repulsion strength.
    l_r : float, optional
        Repulsion length scale.

    Returns
    -------
    ndarray or float
        The value(s) of the interaction potential evaluated at ``dist``.

    Notes
    -----
    The formula implements a *periodic* Morse potential via a hyperbolic
    normalization, which avoids discontinuities at the domain boundaries.
    """
    pot = -c_a*np.cosh((np.abs(dist)-L/2)/l_a)/np.sinh(L/(2*l_a)) + c_r*np.cosh((np.abs(dist)-L/2)/l_r)/np.sinh(L/(2*l_r))
    return pot

def int_potential_attr(dist, c_a = c_a, l_a = l_a):
    """
    Compute the attractive part of the periodic Morse interaction potential 
    evaluated at distances ``dist``. The periodicity is again imposed by hyperbolic functions.

        W_a(x) = -c_a * cosh((|x| - L/2)/l_a) / sinh(L/(2 l_a))

    Parameters
    ----------
    dist : ndarray or float
        Distance(s) at which the Morse potential is evaluated. Typically 
        ``dist = X - X[i]`` when constructing convolution kernels.
    c_a : float, optional
        Attraction strength.
    l_a : float, optional
        Attraction length scale.

    Returns
    -------
    ndarray or float
        The value(s) of the attractive part of the interaction potential evaluated at ``dist``.

    Notes
    -----
    The formula implements a *periodic* Morse potential via a hyperbolic
    normalization, which avoids discontinuities at the domain boundaries.
    """

    pot_attr = (-c_a*np.cosh((np.abs(dist)-L/2)/l_a))/np.sinh(L/(2*l_a))
    return pot_attr

def int_potential_rep(dist, c_r = c_r, l_r = l_r):
    """
    Compute the repulsive part of the  periodic Morse interaction potential evaluated
    at distances ``dist`` using hyperbolic functions to impose perdiodicity.


        W_r(x) = c_r * cosh((|x| - L/2)/l_r) / sinh(L/(2 l_r))

    Parameters
    ----------
    dist : ndarray or float
        Distance(s) at which the Morse potential is evaluated. Typically 
        ``dist = X - X[i]`` when constructing convolution kernels.
    c_r : float, optional
        Repulsion strength.
    l_r : float, optional
        Repulsion length scale.

    Returns
    -------
    ndarray or float
        The value(s) of the interaction potentials repulsive part evaluated at ``dist``.

    Notes
    -----
    The formula implements a *periodic* Morse potential via a hyperbolic
    normalization, which avoids discontinuities at the domain boundaries.
    """
    pot_rep = c_r*np.cosh((np.abs(dist)-L/2)/l_r)/np.sinh(L/(2*l_r))
    return pot_rep
    
def int_force(dist, c_a = c_a, l_a = l_a, c_r = c_r, l_r = l_r):
    """
    Compute the derivative of the periodic Morse interaction potential, i.e.
    the interaction force W'(x), evaluated at distances ``dist``.

    The expression corresponds to:

        W'(x) = d/dx W(x)
              = -c_a * sinh((|x| - L/2)/l_a) * x / (l_a * |x| * sinh(L/(2 l_a)))
                + c_r * sinh((|x| - L/2)/l_r) * x / (l_r * |x| * sinh(L/(2 l_r)))

    Parameters
    ----------
    dist : ndarray or float
        Distance(s) at which the force kernel is evaluated.
        Must satisfy dist != 0 for the analytical expression (division by |x|).
        For numerical use, the ``dist=0`` point is handled implicitly when used
        as a convolution kernel.
    c_a : float, optional
        Attraction strength.
    l_a : float, optional
        Attraction length scale.
    c_r : float, optional
        Repulsion strength.
    l_r : float, optional
        Repulsion length scale.

    Returns
    -------
    ndarray or float
        Values of the interaction force W'(x) evaluated at ``dist``.

    Notes
    -----
    - The formula includes division by |dist|. In numerical applications,
      the ``dist = 0`` entry of the convolution kernel should be set to 0
      manually if needed, since W'(0) = 0 by symmetry.
    - The force inherits periodicity via the same hyperbolic normalization
      as the potential.
    """
    
    force = ((-c_a*np.sinh((np.abs(dist)-L/2)/l_a)*dist)/(l_a*np.sinh(L/(2*l_a))*np.abs(dist))) + ((c_r*np.sinh((np.abs(dist)-L/2)/l_r)*dist)/(l_r*np.sinh(L/(2*l_r))*np.abs(dist)))
    force = np.where(dist == 0, 0.0, force)
    return force