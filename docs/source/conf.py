# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import tomllib
from pathlib import Path

# Fügt das Projekt-Root-Verzeichnis hinzu (wo der Ordner 'dbgsom' liegt)
sys.path.insert(0, os.path.abspath("../.."))

project = "DBGSOM"
copyright = "2023, Sandro Martens"
author = "Sandro Martens"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx_gallery.gen_gallery",
]
numpydoc_class_members_toctree = False

sphinx_gallery_conf = {
    "examples_dirs": "../../examples",
    "gallery_dirs": "auto_examples",
}
templates_path = ["_templates"]
exclude_patterns = []
root_doc = "index"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_theme_options = {
    # ... falls hier schon Einträge stehen, lass sie drin ...
    "pygments_light_style": "tango",
    "pygments_dark_style": "monokai",
}
pygments_style = "tango"
pygments_dark_style = "monokai"
html_static_path = ["_static"]


pyproject_path = Path(__file__).parents[2] / "pyproject.toml"

with open(pyproject_path, "rb") as f:
    pyproject_data = tomllib.load(f)
# 3. Version extrahieren
release = pyproject_data["project"]["version"]  # z. B. "0.1.3"
version = ".".join(release.split(".")[:2])  # Macht daraus "0.1" (für kurze Darstellung)
