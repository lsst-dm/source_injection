"""Sphinx configuration file for an LSST stack package.

This configuration only affects single-package Sphinx documentation builds.
"""

from documenteer.sphinxconfig.stackconf import build_package_configs
import lsst.source_injection


_g = globals()
_g.update(build_package_configs(
    project_name='source_injection',
    version=lsst.source_injection.version.__version__))
