---
layout: default
title: Getting started
nav_order: 2
---

# Getting started

## Installation

```bash
pip install discrete1
```

Optional extras:

- Machine learning features: `pip install "discrete1[ml]"`
- Development tools: `pip install "discrete1[dev]"`

## Quick usage

The package includes solvers for fixed source, time-dependent, and criticality problems in slab and spherical geometry.

Try running one of the included examples after cloning the repo:

```bash
git clone https://github.com/bwhewe-13/discrete1.git
cd discrete1
python examples/fixed_source_slab_reeds.py
```

For more, see the [Examples](examples.html) and the [API (Sphinx)]({{ site.baseurl }}/sphinx/index.html).
