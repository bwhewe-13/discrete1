"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

# Make sure the project package is importable. Adjust the path to point at the
# repository root (two levels up from docs/source). This allows autodoc to
# import the `discrete1` package.
sys.path.insert(0, os.path.abspath("../.."))


# -- Project information -----------------------------------------------------

project = "discrete1"
copyright = "2025, Ben Whewell"
author = "Ben Whewell"

# The full version, including alpha/beta/rc tags
release = "0.1.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones. We enable commonly useful extensions for API documentation.
extensions = [
    "sphinx.ext.napoleon",  # support for NumPy/Google style docstrings
    "sphinx.ext.autodoc",  # core autodoc support
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",  # add links to highlighted source
    "sphinx.ext.coverage",
]

# Generate autosummary pages automatically.
autosummary_generate = True

# Napoleon settings (tweak as desired)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_preprocess_types = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
# You can switch to 'sphinx_rtd_theme' for ReadTheDocs-style layout. If you do
# that, add `sphinx_rtd_theme` to `docs/requirements.txt`.

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Optionally show source links for the Python objects
html_show_sourcelink = Truehtml_show_sourcelink = True
