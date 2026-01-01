"""
Auxiliary functions for 1D McKean–Vlasov and Fokker–Planck simulations.

This module provides helper functions for numerical simulation of 1D
aggregation–diffusion models with nonlocal interactions, including:

- Normalization of discretized probability densities.
- Finite-difference operators for first- and second-order spatial derivatives
  and time derivatives (periodic boundary conditions).
- Array analysis utilities for local minima/maxima, sign changes, and
  consecutive sequences.

All functions operate on 1D NumPy arrays and assume uniform grid spacing 
where applicable.
"""

import numpy as np

from modules.params import *

def norm_array(array, dx): 
    """
    Normalize a 1D array so that its discrete integral equals 1.

    Parameters
    ----------
    array : ndarray
        Values to normalize (e.g., a discretized probability density).
    dx : float
        Grid spacing used for the discrete integral.

    Returns
    -------
    ndarray
        Normalized array with sum(array * dx) == 1.
    """
    integral = np.sum(array*dx)
    out = (1/integral)*array
    return out


def df_dx(array): 
    """
    Compute the first-order spatial derivative of a 1D array using central differences
    with periodic boundary conditions.

    Parameters
    ----------
    array : ndarray
        1D array of values defined on a uniform spatial grid.

    Returns
    -------
    ndarray
        Approximate first spatial derivative. Assumes dx is applied externally if needed.
    """

    df_dx = 0.5 * (np.roll(array, -1) - np.roll(array, 1)) #/ (dx)
    return df_dx

def df_dx2(array, dx):
    """
    Compute the second-order spatial derivative of a 1D array using central differences
    with periodic boundary conditions.

    Parameters
    ----------
    array : ndarray
        1D array of values defined on a uniform spatial grid.
    dx : float
        Spatial grid spacing.

    Returns
    -------
    ndarray
        Approximate second spatial derivative.
    """

    df_dx2 = (np.roll(array, -1) - 2*array + np.roll(array, 1)) / (dx**2)
    return df_dx2

def df_dt_1d(array, dt=dt):
    """
    Compute the first-order time derivative of a 1D array using central differences.

    Parameters
    ----------
    array : ndarray
        1D array of values defined on a uniform time grid.
    dt : float, optional
        Time step. Default is global dt.

    Returns
    -------
    ndarray
        Approximate first time derivative.
    """
    df_dt = 0.5 * (np.roll(array, -1) - np.roll(array, 1)) / dt
    return df_dt

def find_local_min_in_arr(array):
    """
    Find the index of the first local minimum in a 1D array using binary search.

    Parameters
    ----------
    array : ndarray
        1D array to search for a local minimum.

    Returns
    -------
    int
        Index of the first local minimum found. Returns -1 if none exists.
    """
    l, r = 0, len(array) - 1
    ans = -1 
    while l <= r:
        mid = l + (r - l) // 2
        # Check if mid is a local minimum
        if (mid == 0 or array[mid] < array[mid - 1]) and (mid == len(array) - 1 or array[mid] < array[mid + 1]):
            # Store the local minimum index
            ans = mid 
            # Continue searching for a smaller index
            r = mid - 1
        # If left neighbor is smaller, move left
        elif mid > 0 and array[mid - 1] < array[mid]:
            r = mid - 1
        # Otherwise, move right
        else:
            l = mid + 1
    return ans 

def findLocalMaxMin(arr): 
    """
    Find the first local maximum and the first local minimum in a 1D array.

    Parameters
    ----------
    arr : ndarray
        1D array to analyze.

    Returns
    -------
    tuple of int
        Indices of the first local maximum and the first local minimum.
    """
    n = len(arr)
    mx = [] 
    mn = [] 
 
    # Iterating over all points to check 
    # local maxima and local minima 
    for i in range(2, n-2): 
        # Condition for local minima 
        if(arr[i-1] > arr[i] < arr[i + 1]): 
            mn.append(i) 
        # Condition for local maxima 
        elif(arr[i-1] < arr[i] > arr[i + 1]): 
            mx.append(i) 
    return mx[0], mn[0]


def find_idx_sign_changes(array): 
    """
    Find indices where the sign of a 1D array changes.

    Parameters
    ----------
    array : ndarray
        1D array of numbers.

    Returns
    -------
    ndarray
        Indices where the product of consecutive elements is negative (sign change occurs).
    """
    idx_sc = np.where(array[:-1] * array[1:] < 0 )[0] +1
    return idx_sc

def consecutive(data, stepsize=1):
    """
    Split an array of consecutive numbers into subarrays where the sequence is continuous.

    Parameters
    ----------
    data : ndarray or list
        1D array of numbers.
    stepsize : int, optional
        Expected difference between consecutive elements (default is 1).

    Returns
    -------
    list of ndarray
        List of subarrays of consecutive numbers.
    """
    return np.split(data, np.where(np.diff(data) != stepsize)[0])