"""One-dimensional neutron transport equation solver with ML acceleration.

Features:
- Fixed source and criticality problems
- Discrete ordinates (SN) angular discretization
- Diamond difference spatial discretization
- BDF1/BDF2 temporal integration
- Slab and spherical geometries
- DJINN-based ML acceleration
- Numba-optimized implementations
"""

from setuptools import find_packages, setup

setup(
    name="discrete1",
    description="""1D neutron transport equation solver with ML acceleration.
        Features:
        - Fixed source and criticality problems
        - Discrete ordinates (SN) angular discretization
        - Diamond difference spatial discretization
        - BDF1/BDF2 temporal integration
        - Slab and spherical geometries
        - DJINN-based ML acceleration
        - Numba-optimized implementations""",
    version="0.1.0",
    python_requires=">=3.10",
    author="Ben Whewell",
    author_email="ben.whewell@pm.me",
    url="https://github.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "numba",
        "scipy",
        "pytest",
        "tqdm",
    ],
    extras_require={
        "ml": [
            "optuna",
            "scikit-learn>=1.4.0",
            "tensorboard",
            "djinnml",
        ],
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "isort",
            "flake8",
            "pre-commit",
        ],
        "all": [
            "optuna",
            "scikit-learn>=1.4.0",
            "tensorboard",
            "djinnml",
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "isort",
            "flake8",
            "pre-commit",
        ],
    },
)
