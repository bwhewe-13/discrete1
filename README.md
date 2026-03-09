# discrete1

[![Documentation Status](https://github.com/bwhewe-13/discrete1/actions/workflows/docs.yml/badge.svg)](https://bwhewe-13.github.io/discrete1/)
[![Tests](https://github.com/bwhewe-13/discrete1/actions/workflows/tests.yml/badge.svg)](https://github.com/bwhewe-13/discrete1/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/bwhewe-13/discrete1/branch/master/graph/badge.svg)](https://codecov.io/gh/bwhewe-13/discrete1)
![License](https://img.shields.io/badge/license-BSD%203--Clause-blue.svg)

⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠟⡋⢅⣂⣐⡨⠙⡻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠟⣩⣥⣬⢕⢰⣾⣿⣿⣿⣿⣷⣄⠊⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⢸⣳⣳⣻⣿⡆⣿⣿⣿⣿⣿⣿⣿⣮⠠⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡘⢜⢞⢮⢗⢡⣿⣿⣿⣿⣿⣿⣿⣿⣧⠂⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠐⣰⣴⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣆⠊⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⡿⠟⡛⠫⠉⠍⠛⡛⢛⠻⠿⢿⣿⣿⡇⢂⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡐⢸⣿⣿⠿⠿⠛⡛⢋⠩⠉⠍⠩⢙⠛⢿⣿⣿
⡿⠃⣬⣴⣾⣿⣿⣿⣿⣷⣶⣾⣤⣥⣠⠩⠠⠹⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⢃⠡⢁⣂⣬⣴⣷⣾⣿⣿⣿⣿⣿⣷⣾⣄⠌⢿
⠅⢱⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⢌⣶⣴⣠⢉⠛⠿⣿⣿⣿⡿⠟⠫⢑⣠⣬⣶⡐⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡆⢊
⠄⣹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢁⢺⣿⣿⣿⣿⣾⣴⠀⠍⠩⢠⣵⣿⣿⣿⣿⣿⡇⠊⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⢂
⣇⠌⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢀⣾⣿⣿⣿⠿⠋⣡⣨⣾⣶⣄⡌⡙⠿⣿⣿⣿⣷⠁⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠠⣹
⣿⡄⠜⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠄⣿⠿⢋⢡⣤⣿⣿⣿⣿⣿⣿⣿⣶⣴⡈⡙⠿⣿⠐⣸⣿⣿⣿⣿⣿⣿⣿⡿⢋⢅⢆⡬⡐⣿
⣿⡿⢢⢴⣲⣶⣌⢻⣿⣿⣿⣿⣿⣿⡇⠌⢡⣨⣶⣿⣿⠟⡛⡩⡩⡍⡍⡛⡻⣿⣿⣶⣅⡌⢂⢸⣿⣿⣿⣿⣿⣿⣿⠡⡱⡱⡍⣿⣖⢸
⣿⡇⢯⡻⡮⣟⣿⠀⣿⣿⣿⣿⠟⡋⡐⢸⣿⣿⣿⠟⡡⡪⡪⡪⡪⡪⡪⣪⣖⢌⠻⣿⣿⣿⡆⢂⠙⠻⣿⣿⣿⣿⣿⡆⠪⢪⢪⠪⢂⣾
⣿⣷⡘⠎⢟⠵⠃⠼⣿⣿⠟⡁⣢⣾⢁⢺⣿⣿⠏⢔⢱⢱⢱⢱⢱⢱⢱⢙⣿⣷⡕⠹⣿⣿⡇⠌⣷⣅⠌⠻⣿⣿⠟⠡⣵⣶⣶⣶⣿⣿
⣿⣿⣿⣷⣶⣾⣷⣔⠈⠅⣢⣾⣿⣿⠠⢸⣿⣿⠨⡢⠣⡣⡣⡣⡣⡣⡣⡣⡪⡻⡫⠅⣿⣿⡇⠂⣿⣿⣿⣮⠈⠅⢬⣾⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⠟⠡⣨⣆⠌⠻⣿⣿⠐⢸⣿⣿⡈⡪⡸⢸⢸⢸⢸⢸⢸⢸⢸⢸⢸⠁⣿⣿⡇⡁⣿⣿⠟⠡⣨⣦⡉⠻⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⡿⠡⣨⣾⣿⣿⣷⣅⡌⢛⠎⢸⣿⣿⣧⠨⡸⡐⢕⢱⢱⢱⢱⢱⢱⢱⠁⣾⣿⣿⡇⠰⢋⢑⣬⣾⣿⣿⣿⣔⢈⠻⣿⣿⣿⣿
⣿⣿⣿⠏⢄⣵⣿⣿⣿⣿⣿⣿⣿⣦⡂⢙⠻⣿⣿⣷⣔⠘⢌⠆⡕⠜⡌⡪⢘⣠⣾⣿⣿⠟⡃⢅⣶⣿⣿⣿⣿⣿⣿⣿⣷⡈⡙⣿⣿⣿
⣿⣿⠃⢬⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠂⣆⡌⡙⢿⣿⣿⣶⣶⣤⣥⣶⣶⣿⣿⠿⠋⢅⣢⠐⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣆⠨⢿⣿
⣿⠃⢬⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣧⠁⣿⣿⣶⣄⡌⠛⠿⣿⣿⣿⣿⠿⠋⢅⣬⣾⣿⣿⠈⣼⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣆⠊⢿
⡇⢌⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠐⢹⣿⣿⣿⣿⣾⣤⡊⢙⠋⢅⣬⣾⣿⣿⣿⣿⡏⠌⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡎⢘
⡐⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡎⢸⣿⣿⣿⠿⢛⢉⣐⣴⣮⣄⡊⡙⠿⢿⣿⣿⠇⢢⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠂
⡆⠙⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠇⡂⠋⢍⣠⣬⣶⣿⣿⣿⣿⣿⣿⣶⣦⣆⡨⠙⠨⠸⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⢃⢱
⣿⣦⡂⡙⢛⠿⠿⠿⠿⡛⢛⠛⡉⢅⣂⣥⡠⢹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⣁⣮⣄⣂⡩⢉⠛⡛⢛⠻⢛⠛⡛⠍⣐⣴⣿
⣿⣿⣿⣷⣶⣶⣵⣬⣶⣶⣶⣿⣿⣿⣿⣿⣧⠂⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠠⣼⣿⣿⣿⣿⣿⣷⣶⣶⣶⣶⣶⣶⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡆⡙⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠃⢲⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡄⠜⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠃⢥⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡔⠘⢿⣿⣿⣿⣿⣿⣿⡿⠃⢬⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣮⡀⠛⠿⣿⣿⠿⠋⣂⣵⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⠿⠻⠿⠻⠿⠻⠿⠻⠿⠻⠿⠻⠟⠿⠻⠟⠿⠻⠟⠿⠻⠿⠺⠔⠄⠂⠦⠳⠟⠿⠻⠟⠿⠻⠟⠿⠻⠟⠿⠻⠟⠿⠻⠟⠿⠻⠿⠻⠿⠻

This solves the neutron transport equation for one-dimensional problems
in both slab and sphere geometry. The discrete ordinates method is used
to solve the equation and numba is used to accelerate the transport
sweeps. This code can be used to solve fixed source, time dependent, and
criticality problems.


- **Spatial Discretization**: Diamond Difference
- **Temporal Discretization**: Backward Euler, BDF2
- **Multigroup Convergence**: Source Iteration, DMD
- **K-eigenvalue Convergence**: Power Iteration


## Current Research Techniques
This is an experimental code that explores different acceleration and data
saving techniques related to the neutron transport equation.


1. DJINN incorporates Deep Jointly-Informed Neural Networks
    into the S<sub>N</sub> code for &Sigma;<sub>s</sub> &Phi;
    and &Sigma;<sub>f</sub> &Phi; calculations<sup>1</sup>.

2. Collision-Based Hybrid Method separates the collided and uncollided terms to
    be used with different numbers of ordinates (N) and energy groups (G) for
    time-dependent problems<sup>2</sup>.


## Documentation

The documentation is built automatically using Sphinx and deployed to GitHub Pages. You can find the latest documentation at:

[https://bwhewe-13.github.io/discrete1/](https://bwhewe-13.github.io/discrete1/)

To build the documentation locally:

```bash
# Install dependencies
python -m pip install -r docs/requirements.txt

# Build HTML docs
cd docs
make html
# Output will be in docs/build/html
```

## ML Installation Options

Choose one optional ML extra depending on which DJINN backend you want:

- `discrete1[ml]`: PyTorch-oriented DJINN fork (`bwhewe-13/DJINN`)
- `discrete1[tf-ml]`: TensorFlow-oriented DJINN fork (`llnl/DJINN`, `djinn` subdirectory)

Install examples:

```bash
python -m pip install -e ".[ml]"
python -m pip install -e ".[tf-ml]"
```

AutoDJINN backend selection examples:

```python
from discrete1.ml.predict import AutoDJINN

# PyTorch backend
model_torch = AutoDJINN(
    "encoder.pt",
    "djinn_model",
    "decoder.pt",
    transformer=lambda x: x,
    detransformer=lambda x: x,
    backend="torch",
)

# TensorFlow backend
model_tf = AutoDJINN(
    "encoder.h5",
    "djinn_model",
    "decoder.h5",
    transformer=lambda x: x,
    detransformer=lambda x: x,
    backend="tensorflow",
)
```

**Note:** The PyTorch ML path (`discrete1[ml]`) is expected to be the preferred option going forward, since it is currently receiving active updates and development.

## Development

To set up the development environment:

```bash
# Clone the repository
git clone https://github.com/bwhewe-13/discrete1.git
cd discrete1

# Install package with development dependencies
python -m pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

To run the tests with coverage reporting:

```bash
pytest
```

Coverage reports will be generated in `coverage_html/index.html` and `coverage.xml`.


<sup>1</sup> Ben Whewell and Ryan G. McClarren, (2022). Data Reduction in
    Deterministic Neutron Transport Calculations Using Machine Learning.
    Annals of Nuclear Energy, 176, p .109276. DOI: 10.1016/j.anucene.2022.109276.

<sup>2</sup> Ben Whewell, Ryan G. McClarren, Cory D. Hauck, and Minwoo Shin,
    (2023). Multigroup Neutron Transport Using a Collision-Based Hybrid Method,
    Nuclear Science and Engineering, 197:7, 1386-1405, DOI: 10.1080/00295639.2022.2154119.
