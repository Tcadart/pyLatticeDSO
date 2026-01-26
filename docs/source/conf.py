# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
print("Added to sys.path:", os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))



# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pyLatticeDesign'
copyright = '2025, Thomas Cadart'
author = 'Thomas Cadart'
release = '1.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',      # support Google & NumPy docstring
    'sphinx.ext.viewcode',      # ajoute lien vers code source
    'sphinx_autodoc_typehints', # affiche les types Python
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# GitHub Pages configuration
html_extra_path = []

def setup(app):
    """Add custom setup for Sphinx build."""
    # Create .nojekyll file to disable Jekyll processing on GitHub Pages
    import os
    def create_nojekyll(app, env):
        if app.builder.name == 'html':
            nojekyll_path = os.path.join(app.outdir, '.nojekyll')
            with open(nojekyll_path, 'w') as f:
                f.write('')
    
    app.connect('env-updated', create_nojekyll)
