"""
Lazy import registry with namespace support.

A lightweight library for managing lazy-loading registries with type safety
and built-in support for pretrained model patterns.
"""

from .registry import NAMESPACE, ImportString, LazyImportDict, Namespace, Registry

__all__ = ["ImportString", "LazyImportDict", "Registry", "Namespace", "NAMESPACE"]

# Version is managed by hatch-vcs from Git tags
try:
    from ._version import __version__
except ImportError:
    # Fallback for editable installs without build
    try:
        from importlib.metadata import version

        __version__ = version("mediaref")
    except Exception:
        __version__ = "0.0.0.dev0"
