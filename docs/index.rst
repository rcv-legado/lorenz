.. Lorenz documentation master file, created by
   sphinx-quickstart on Wed May 21 05:29:51 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Lorenz
======
This is just a simple wrapper around `scipy.integrate.odeint` to integrate the Lorenz equations given initial conditions. I created this little module while working on an undergrad non-linear dynamics project in 2014.

The Lorenz equations are:

.. math::
    \dot{x}&=\sigma(y-x)\\
    \dot{y}&=(\rho-z)x-y\\
    \dot{z}&=xy-\beta z

Example
=======
::

    import lorenz
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.axes3d import Axes3D
    
    # The initial x, y and z values
    initial_position = [0.1, 0.2, 0.3]

    # The time domain
    t = np.arange(0.0, 500.0, 0.001)

    # Get coordinates along trajectory
    x, y, z = lorenz.simulate_trajectory(initial_position, t,
        sigma=10.0, rho=28.0, beta=8.0/3.0)

    # Plot trajectory in 3D
    ax = Axes3D(plt.gcf())
    ax.plot(x, y, z)
    plt.show()

    # Get Lorenz map
    z_n, z_np1 = lorenz.lorenz_map(z)

    # Plot Lorenz map
    plt.plot(z_n, z_np1, 'bo')
    plt.show()

lorenz.py
=========
.. automodule:: lorenz
    :members: