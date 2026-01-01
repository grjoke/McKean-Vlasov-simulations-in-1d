'''
Functions for plotting the derived data in a suitable way
'''


import numpy as np
import matplotlib.pyplot as plt

from modules.params import *
from modules.aux_functions import *



def plot_density_evolution(u_store_fp, sigma=sigma, t_end=t_end,  num_curves=4, saving=False):
    """
    Plot the time evolution of the probability density obtained from the
    McKean–Vlasov (Fokker–Planck) simulation.

    This function visualizes several density snapshots as well as the final density 
    of a density time evolution stored in ``u_store_fp`` at equidistant times. The function 
    additionally checks mass conservation and prints a warning if the integral 
    of the density deviates significantly from 1.

    Parameters
    ----------
    u_store_fp : ndarray of shape (Nx, Nt)
        Array containing the density values for all spatial grid points ``Nx`` 
        and all time steps ``Nt``. The entry ``u_store_fp[i, j]`` corresponds 
        to the density at spatial grid point ``X[i]`` and time ``j * dt``.
    sigma : float, optional
        Noise strength (diffusion parameter) used in the simulation. 
        Only used for labeling the plot. Default is the global variable ``sigma``.
    t_end : float, optional
        End time of the simulation. Determines the distribution of plotted 
        snapshots. Default is the global variable ``t_end``.
    num_curves : int, optional
        Number of intermediate time snapshots to plot in addition to the final 
        density. Default is 4.
    saving : bool, optional
        If ``True``, the plot is saved as a PDF in ``plot_dir``. 
        Default is ``False``.

    Notes
    -----
    - Uses global variables ``X``, ``dx``, ``dt`` and ``plot_dir``.
    - Checks whether the density remains normalized by verifying that 
      ``sum(u * dx)`` stays within the interval ``[0.998, 1.02]``.

    Returns
    -------
    int
        Always returns ``0``. The function is intended for plotting side effects.

    """

    fname = 'PDFs_sigma=' + str(sigma) + '_t_end=' + str(t_end) + '.pdf'
    t_nodes = int(t_end/dt)
    
    for i in range(num_curves):
        row = int((i*t_nodes)/(num_curves))
        loc = u_store_fp[:, row]
        plt.plot(X, loc, label = r't= ' + str(np.round(i*t_nodes*dt/num_curves, 2)) + ' s', alpha = 1)
        check = np.sum(loc*dx)
        if check < 0.998 or check > 1.02:
            print('WARNING: THE SIMULATION GET UNSTABLE AND THE DENSITY IS NOT PRESERVED')
    loc = u_store_fp[:,-2]
    plt.plot(X, loc, label = r't= ' + str(np.round(t_nodes*dt, 2)) + ' s', alpha = 1)
    check = np.sum(loc*dx)
    if check < 0.998 or check > 1.02:
        print('WARNING: THE SIMULATION GET UNSTABLE AND THE DENSITY IS NOT PRESERVED')
    plt.title(r'Solution of McKean-Vlasov equation at different time steps' + '\n' + r'for $\sigma = $' + str(sigma) + r' and $t_{end} = $' + str(t_end))
    plt.grid()
    plt.legend()
    plt.ylabel(r'$\rho(x,t)$')
    plt.xlabel(r'$x$')
    plt.axis(ymin = 0)
    #plt.ylim(0,0.5)
    if saving==True: 
        plt.savefig(plot_dir + fname, dpi=300, bbox_inches='tight', transparent=True)
    plt.show()
    return 0 


def plot_free_energy_without_derivative(free_energy, entropy_energy, interaction_energy, vmax_time, sigma=sigma, t_end=t_end, plot=False, saving=False):
    """
    Plot the time evolution of the free energy and its components.This function 
    visualizes the total free energy as well as the entropy and
    interaction contributions for a given density trajectory. It assumes that
    all energy terms have been computed beforehand (e.g. via a dedicated
    free-energy module).

    Parameters
    ----------
    free_energy : array_like
        Total free energy F[ρ(t)] evaluated at each time step.
    entropy_energy : array_like
        Entropic contribution to the free energy at each time step.
    interaction_energy : array_like
        Interaction contribution to the free energy at each time step.
    vmax_time : float
        Time at which the maximum density value is attained.
    sigma : float, optional
        Noise amplitude used in the simulation.
    t_end : float, optional
        Final simulation time.
    plot : bool, optional
        If True, generate and display the plot.
    saving : bool, optional
        If True, save the plot to disk.

    Returns
    -------
    int
        Always returns 0 (for legacy compatibility).
    """
    t = np.linspace(0, t_end, free_energy.size, endpoint=True)
    
    if plot == True:
        fname_fe = 'FreeEnergyAlone_sigma=' + str(sigma) + '_t_end=' + str(t_end) + '.pdf'
        fig, ax1 = plt.subplots(1,1)

        fr_energ = ax1.plot(t[1:-1], free_energy[1:-1], color='b', label = r'$F[\rho]$')
        ent_energ = ax1.plot(t[1:-1], entropy_energy[1:-1], color = 'g', linestyle='dotted' ,label = r'$F_{ent}[\rho]$')
        int_energ = ax1.plot(t[1:-1], interaction_energy[1:-1], color = 'r', linestyle='dotted', label = r'$F_{int}[\rho]$')
        vmax_t = ax1.axvline(x=vmax_time, c='m', linestyle='--', label=r'$t(\rho_{max})$')
        zero_1 = ax1.axhline(y=0, color = 'tab:orange', linestyle='-.')
        ax1.set_xlabel(r'$t$ $[s]$')
        ax1.set_ylabel(r'$F[\rho]$ $[J]$')
        ax1.set_title(r'Free energy evolution of McKean-Vlasov equation' + '\n' +  r'$\sigma$ = ' + str(sigma))
        ax1.legend()

        if saving==True: 
            plt.savefig(plot_dir + fname_fe,dpi=300, bbox_inches='tight', transparent=True)
        plt.show()
    return 0


def plot_free_energy_without_tp(free_energy, entropy_energy, interaction_energy, vmax_time, sigma=sigma, t_end=t_end, plot=False, saving=False):
    """
    Plot the free energy and its time derivative without identifying turning points.

    This function displays both the free energy evolution and its temporal
    derivative, but does not attempt to extract or annotate characteristic
    turning points. It is mainly intended for diagnostic or exploratory analysis.

    Parameters
    ----------
    free_energy : array_like
        Total free energy F[ρ(t)] evaluated at each time step.
    entropy_energy : array_like
        Entropic contribution to the free energy.
    interaction_energy : array_like
        Interaction contribution to the free energy.
    vmax_time : float
        Time at which the maximum density value is attained.
    vmax_x : float
        Spatial position of the maximum density (not used for plotting, kept for
        interface consistency).
    sigma : float, optional
        Noise amplitude used in the simulation.
    t_end : float, optional
        Final simulation time.
    plot : bool, optional
        If True, generate and display the plot.
    saving : bool, optional
        If True, save the plot to disk.

    Returns
    -------
    int
        Always returns 0 (for legacy compatibility).
    """
    free_energy_dt = df_dt_1d(free_energy)
    entropy_dt = df_dt_1d(entropy_energy)
    interaction_dt = df_dt_1d(interaction_energy)

    t = np.linspace(0, t_end, free_energy.size, endpoint=True)

    if plot == True:
        fname_fe = 'FreeEnergy_sigma=' + str(sigma) + '_t_end=' + str(t_end) + '.pdf'
        fig, (ax1, ax2) = plt.subplots(1,2)#, sharex=True)

        fig.set_size_inches(12,5.5)
        fr_energ = ax1.plot(t[2:-2], free_energy[2:-2], color='b', label = r'$F[\rho]$')
        ent_energ = ax1.plot(t[2:-2], entropy_energy[2:-2], color = 'g', linestyle='dotted' ,label = r'$F_{ent}[\rho]$')
        int_energ = ax1.plot(t[2:-2], interaction_energy[2:-2], color = 'r', linestyle='dotted', label = r'$F_{int}[\rho]$')
        vmax_t = ax1.axvline(x=vmax_time, c='m', linestyle='--', label=r'$t(\rho_{max})$')
        zero_1 = ax1.axhline(y=0, color = 'tab:orange', linestyle='-.')

        ax1.set_xlabel(r'$t$ $[s]$')
        ax1.set_ylabel(r'$F[\rho]$ $[J]$')
        ax1.set_title(r'Free energy evolution of McKean-Vlasov equation' + '\n' +  r'$\sigma$ = ' + str(sigma))
        ax1.legend()

        fr_energ_dt = ax2.plot(t[2:-2], free_energy_dt[2:-2], color='b', label = r'$\partial_t$ $F[\rho]$')
        ent_energ_dt = ax2.plot(t[2:-2], entropy_dt[2:-2], color='g', linestyle='dotted', label = r'$\partial_t$ $F_{ent}[\rho]$')
        int_energ_dt = ax2.plot(t[2:-2], interaction_dt[2:-2], color='r', linestyle='dotted', label = r'$\partial_t$ $F_{int}[\rho]$')
        vmax_t = ax2.axvline(x=vmax_time, c='m', linestyle='--', label=r'$t(\rho_{max})$')
        zero_2 = ax2.axhline(y=0, color = 'tab:orange', linestyle='-.')

        ax2.set_xlabel(r'$t$ $[s]$')
        ax2.set_ylabel(r'$\partial t F[\rho]$ $[J/s]$')
        ax2.set_title(r'Derivative of free energy evolution for $\sigma$ = ' + str(sigma))
        ax2.legend()

        if saving==True: 
            plt.savefig(plot_dir + fname_fe,dpi=300, bbox_inches='tight', transparent=True)
        plt.show()
    return 0


def plot_free_energy_with_tp_clustering(free_energy, entropy_energy, interaction_energy, vmax_time, sigma=sigma, t_end=t_end, plot=False, saving=False): 
    """
    Plot free energy evolution and identify a turning point associated with clustering.

    In the clustering regime, a characteristic turning point is identified as a
    local minimum of the time derivative of the interaction energy. This time
    is interpreted as the onset or strengthening of aggregation.

    Parameters
    ----------
    free_energy : array_like
        Total free energy F[ρ(t)].
    entropy_energy : array_like
        Entropic contribution to the free energy.
    interaction_energy : array_like
        Interaction contribution to the free energy.
    vmax_time : float
        Time at which the maximum density value is attained.
    sigma : float, optional
        Noise amplitude used in the simulation.
    t_end : float, optional
        Final simulation time.
    plot : bool, optional
        If True, generate and display the plot.
    saving : bool, optional
        If True, save the plot to disk.

    Returns
    -------
    float
        Time corresponding to the local minimum of ∂ₜF_int[ρ], interpreted as a
        clustering-related turning point.
    """
    free_energy_dt = df_dt_1d(free_energy)
    entropy_dt = df_dt_1d(entropy_energy)
    interaction_dt = df_dt_1d(interaction_energy)

    t = np.linspace(0, t_end, free_energy.size, endpoint=True)

    min_dt_idx = find_local_min_in_arr(interaction_dt)
    min_dt_time = np.round(t[min_dt_idx], 2)

    if plot == True: 
        fname_fe = 'FreeEnergy_sigma=' + str(sigma) + '_t_end=' + str(t_end) + '_with_tp.pdf'
        fig, (ax1, ax2) = plt.subplots(1,2)#, sharex=True)

        fig.set_size_inches(12,5.5)
        fr_energ = ax1.plot(t[1:-1], free_energy[1:-1], color='b', label = r'$F[\rho]$')
        ent_energ = ax1.plot(t[1:-1], entropy_energy[1:-1], color = 'g', linestyle='dotted' , label = r'$F_{ent}[\rho]$')
        int_energ = ax1.plot(t[1:-1], interaction_energy[1:-1], color = 'r', linestyle='dotted', label = r'$F_{int}[\rho]$')
        #zero_1 = ax1.axhline(y=0, color = 'tab:orange', linestyle='-.')
        ax1.axvline(x=vmax_time, color='m', linestyle = '-.', label=r'$t(\rho_{max})$')
        tp_1 = ax1.axvline(x=min_dt_time, color = 'k', linestyle = '--', label='turning point')
        ax1.set_xlabel(r'$t$ $[s]$')
        ax1.set_ylabel(r'$F[\rho]$ $[J]$')
        ax1.set_title(r'Free energy evolution of McKean-Vlasov equation' + '\n' +  r'$\sigma$ = ' + str(sigma))
        ax1.legend()

        fr_energ_dt = ax2.plot(t[2:-2], free_energy_dt[2:-2], color='b', label = r'$\partial_t$ $F[\rho]$')
        ent_energ_dt = ax2.plot(t[2:-2], entropy_dt[2:-2], color='g', linestyle='dotted', label = r'$\partial_t$ $F_{ent}[\rho]$')
        int_energ_dt = ax2.plot(t[2:-2], interaction_dt[2:-2], color='r', linestyle='dotted', label = r'$\partial_t$ $F_{ent}[\rho]$')
        zero_2 = ax2.axhline(y=0, color = 'tab:orange', linestyle='-.')
        ax2.axvline(x=vmax_time, color='m', linestyle = '-.', label=r'$t(\rho_{max})$')
        tp_2 = ax2.axvline(x=min_dt_time, color = 'k', linestyle = '--', label=r'local min in $\partial_t$ $F[\rho]$')
        ax2.set_xlabel(r'$t$ $[s]$')
        ax2.set_ylabel(r'$\partial t F[\rho]$ $[J/s]$')
        ax2.set_title(r'Derivative of free energy evolution for $\sigma$ = ' + str(sigma))
        ax2.legend()
        
        if saving==True:
            plt.savefig(plot_dir + fname_fe,dpi=300, bbox_inches='tight', transparent=True)
        plt.show()

    return min_dt_time

def plot_free_energy_with_tp_diffusion(free_energy, entropy_energy, interaction_energy, vmax_time, sigma=sigma, t_end=t_end, plot=False, saving=False):
    """
    Plot free energy evolution and analyze metastability in the diffusion regime.

    This function identifies turning points in the total free-energy derivative
    and detects possible metastable plateaus characterized by near-zero
    ∂ₜF[ρ] over an extended time interval.

    Parameters
    ----------
    free_energy : array_like
        Total free energy F[ρ(t)].
    entropy_energy : array_like
        Entropic contribution to the free energy.
    interaction_energy : array_like
        Interaction contribution to the free energy.
    vmax_time : float
        Time at which the maximum density value is attained.
    vmax_x : float
        Spatial position of the maximum density (not directly used for plotting).
    sigma : float, optional
        Noise amplitude used in the simulation.
    t_end : float, optional
        Final simulation time.
    plot : bool, optional
        If True, generate and display the plot.
    saving : bool, optional
        If True, save the plot to disk.

    Returns
    -------
    min_dt_time : float
        Time corresponding to a local minimum of ∂ₜF[ρ].
    lifetime_duration : float
        Estimated duration of a metastable plateau in physical time units.
        Returns 0 if no metastable region is detected.
    """

    free_energy_dt = df_dt_1d(free_energy)
    entropy_dt = df_dt_1d(entropy_energy)
    interaction_dt = df_dt_1d(interaction_energy)

    lifetime_duration = 0

    t = np.linspace(0, t_end, free_energy.size, endpoint=True)

    zero_derivative = np.where(np.abs(free_energy_dt)<0.002)[0]
    areas = consecutive(zero_derivative)
    if len(areas) > 1:
        lifetime_idx = areas[0]
        if len(lifetime_idx) > 0: 
            metastable = True
            idx_begin = lifetime_idx[0]
            idx_end = lifetime_idx[-1]
            lifetime = idx_end - idx_begin
            begin_time = t[idx_begin]
            end_time = t[idx_end]
            lifetime_duration = (lifetime/int(t_end/dt)) * t_end
    else: 
        metastable = False

    min_dt_idx = find_local_min_in_arr(free_energy_dt)
    min_dt_time = np.round(t[min_dt_idx], 2)

    if plot == True:  
        fname_fe = 'FreeEnergy_sigma=' + str(sigma) + '_t_end=' + str(t_end) + '_with_tp.pdf'
        fig, (ax1, ax2) = plt.subplots(1,2)#, sharex=True)

        fig.set_size_inches(12,5.5)
        fr_energ = ax1.plot(t[1:-1], free_energy[1:-1], color='b', label = r'$F[\rho]$')
        ent_energ = ax1.plot(t[1:-1], entropy_energy[1:-1], color = 'g', linestyle='dotted' , label = r'$F_{ent}[\rho]$')
        int_energ = ax1.plot(t[1:-1], interaction_energy[1:-1], color = 'r', linestyle='dotted', label = r'$F_{int}[\rho]$')
        #zero_1 = ax1.axhline(y=0, color = 'tab:orange', linestyle='-.')
        ax1.axvline(x=vmax_time, color='m', linestyle = '-.', label=r'$t(\rho_{max})$')
        tp_1 = ax1.axvline(x=min_dt_time, color = 'k', linestyle = '--', label='turning point')
        if metastable == True: 
            begin_ms = ax1.axvline(x=begin_time, linestyle = 'dotted', label = 'lifetime')
            end_ms = ax1.axvline(x=end_time, linestyle='dotted')
        ax1.set_xlabel(r'$t$ $[s]$')
        ax1.set_ylabel(r'$F[\rho]$ $[J]$')
        ax1.set_title(r'Free energy evolution of McKean-Vlasov equation' + '\n' +  r'$\sigma$ = ' + str(sigma))
        ax1.legend()

        fr_energ_dt = ax2.plot(t[1:-1], free_energy_dt[1:-1], color='b', label = r'$\partial_t$ $F[\rho]$')
        ent_energ_dt = ax2.plot(t[1:-1], entropy_dt[1:-1], color='g', linestyle='dotted', label = r'$\partial_t$ $F_{ent}[\rho]$')
        int_energ_dt = ax2.plot(t[1:-1], interaction_dt[1:-1], color='r', linestyle='dotted', label = r'$\partial_t$ $F_{int}[\rho]$')
        zero_2 = ax2.axhline(y=0, color = 'tab:orange', linestyle='-.')
        ax2.axvline(x=vmax_time, color='m', linestyle = '-.', label=r'$t(\rho_{max})$')
        tp_2 = ax2.axvline(x=min_dt_time, color = 'k', linestyle = '--', label=r'local min in $\partial_t$ $F[\rho]$')
        if metastable == True: 
            begin_ms = ax2.axvline(x=begin_time, linestyle = 'dotted', label = 'lifetime')
            end_ms = ax2.axvline(x=end_time, linestyle='dotted')
        ax2.set_xlabel(r'$t$ $[s]$')
        ax2.set_ylabel(r'$\partial t F[\rho]$ $[J/s]$')
        ax2.set_title(r'Derivative of Free Energy evolution for $\sigma$ = ' + str(sigma))
        ax2.legend()

        if saving==True:
            plt.savefig(plot_dir + fname_fe,dpi=300, bbox_inches='tight', transparent=True)
        plt.show()
        
    return min_dt_time, lifetime_duration, 
    


def colored_line_between_pts(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified between (x, y) points by a third value.

    It does this by creating a collection of line segments between each pair of
    neighboring points. The color of each segment is determined by the
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should have a size one less than that of x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.*

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Check color array size (LineCollection still works, but values are unused)
    if len(c) != len(x) - 1:
        warnings.warn(
            "The c argument should have a length one less than the length of x and y. "
            "If it has the same length, use the colored_line function instead."
        )

    # Create a set of line segments so that we can color them individually
    # This creates the points as an N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, **lc_kwargs)

    # Set the values used for colormapping
    lc.set_array(c)

    return ax.add_collection(lc)