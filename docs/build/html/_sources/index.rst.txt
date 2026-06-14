discrete1
=========

1D Discrete Ordinates Multigroup Neutron Transport in Python.

- Spatial Discretization: Diamond Difference
- Temporal Discretization: Backward Euler, BDF2
- Multigroup Convergence: Source Iteration, DMD
- K-eigenvalue Convergence: Power Iteration

.. note::
   Experimental features: DJINN, SVD-DJINN, and a collision-based HYBRID method
   for accelerated and data-efficient neutron transport.

Get started
-----------

- Install: ``pip install discrete1``
- New here? See :doc:`getting-started`.
- Browse runnable scripts in :doc:`examples`.
- Dive into the :doc:`api` generated from the source docstrings.

Repository
----------

- Repo: https://github.com/bwhewe-13/discrete1
- Issues: https://github.com/bwhewe-13/discrete1/issues

.. toctree::
    :maxdepth: 2
    :caption: Contents:

    getting-started
    examples
    discrete1
    api
