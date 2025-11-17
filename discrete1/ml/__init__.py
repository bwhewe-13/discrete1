"""Machine learning utilities for discrete1.

This optional subpackage provides tools that accelerate the 1D neutron
transport solver via data-driven models. It includes helpers for data
preparation, training and inference wrappers for DJINN-based models, and
common error metrics.

Notes
-----
This subpackage relies on optional dependencies. Install them with::

    pip install discrete1[ml]

Dependencies include scikit-learn for data utilities, TensorFlow/Keras for
autoencoders used in reduced-order models, and ``djinn`` for DJINN models.
Only import these when needed to keep the base installation light.
"""
