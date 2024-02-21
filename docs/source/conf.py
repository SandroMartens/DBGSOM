# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import pathlib
import sys

sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("dbgsom"))
sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())


project = "DBGSOM"
copyright = "2023, Sandro Martens"
author = "Sandro Martens"
release = "0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "numpydoc",
    "sphinx.ext.autosummary",
    # "sphinx_autodoc_typehints",
]
numpydoc_class_members_toctree = False
templates_path = ["_templates"]
exclude_patterns = []

root_doc = "index"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]

sys.path.append(os.path.abspath("./dbgsom"))
sys.path.insert(0, os.path.abspath("."))
