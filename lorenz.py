from __future__ import division
import numpy as np
from scipy.integrate import odeint
from scipy.signal import argrelextrema

#: The sigma value to use when `simulate_trajectory` is called without the sigma argument
default_sigma = 10.0

#: The rho value to use when `simulate_trajectory` is called without the rho argument
default_rho = 28.0

#: The beta value to use when `simulate_trajectory` is called without the beta argument
default_beta = 8.0 / 3.0

def dx(x, y, sigma):
    return sigma * (y - x)

def dy(x, y, z, rho):
    return (rho - z) * x - y

def dz(x, y, z, beta):
    return x * y - beta * z

def dxyz(xyz, t, sigma, rho, beta):
    x, y, z = xyz
    return [
        dx(x, y, sigma),
        dy(x, y, z, rho),
        dz(x, y, z, beta)
    ]

def jacobian(xyz, t, sigma, rho, beta):
    x, y, z = xyz
    return [
        [-sigma, sigma, 0.0],
        [rho - z, -1.0, -x],
        [y, x, -beta]
    ]

def simulate_trajectory(xyz0, t, sigma=None, rho=None, beta=None,
                        full_output=0, **kwargs):
    """
    Generate coordinate data for the trajectory of a phase point
    with initial position `xyz0`, in the lorenz system defined by
    parameters `sigma`, `rho` and `beta`.

    Uses `scipy.integrate.odeint` to integrate over the conventional
    lorenz equations using intial conditions `xyz0` and time array `t`.
    The resulting coordinate arrays are returned separately and have
    length len(t).

    :param xyz0: Initial position of phase point (at `t = 0`), of
      form `[x0, y0, z0]`.
    :type xyz0: array of floats

    :param t: A sequence of points in time for which to calculate
      coordinates.
    :type t: array of floats

    :param sigma: The sigma-value (aka **Prandtl number**) of the
      Lorenz system, used to calculate the time-derivative of `x`.
      Defaults to `simlorenz.default_sigma`, which can be set before
      calling `simulate_trajectory`.
    :type sigma: float, optional

    :param rho: The rho-value (aka the **Rayleigh number**) of the \
    Lorenz system, used to calculate the time-derivative of `y`. \
    Defaults to `simlorenz.default_rho`, which can be set before \
    calling `simulate_trajectory`.
    :type rho: float, optional

    :param beta: The beta-value (no special name) of the Lorenz \
    system, used to calculate the time-derivative of `z`. \
    Defaults to `simlorenz.default_beta`, which can be set before \
    calling `simulate_trajectory`.
    :type beta: float, optional

    :param full_output: True if to return a dict of optional outputs \
    as the fourth return value. See the `scipy.integreate.odeint` \
    documentation for more details.
    :type full_output: bool, optional

    :param kwargs: All other keyword args are sent to odeint.

    :returns:
      - **x** *(array of floats)*: `x`-coordinates of trajectory
      - **y** *(array of floats)*: `y`-coordinates of trajectory
      - **z** *(array of floats)*: `z`-coordinates of trajectory
      - **infodict** *(dict, only if full_output==True)*: additional
        output

    """
    if sigma is None: sigma = default_sigma
    if rho is None: rho = default_rho
    if beta is None: beta = default_beta
    
    xyz, info = odeint(dxyz, xyz0, t, args=(sigma, rho, beta),
        Dfun=jacobian, full_output=1, **kwargs)
    x, y, z = zip(*xyz)

    if full_output:
        return x, y, z, info
    else:
        return x, y, z


def lorenz_map(z):
    """Calculates Lorenz map for given z-values (as returned by
    `simulate_trajectory`).
    
    :param z: array of z-coordinates of a trajectory in a Lorenz system
    :type z: array of floats
    :returns:
      - **p_n** *(array of floats)*: The independent values of the
        Lorenz map.
      - **p_np1** *(array of floats)*: The dependent values of the
        Lorenz map.

    """
    z = np.array(z)
    peak_indexes = argrelextrema(z, np.greater)
    peaks = z[peak_indexes]
    p0 = peaks[:-1]
    p1 = peaks[1:]
    return p0, p1
