"""
Lazy import registry with namespace support.

A lightweight library for managing lazy-loading registries with type safety
and built-in support for pretrained model patterns.
"""

from .registry import LazyImportDict, NAMESPACE, Namespace, Registry

__all__ = ["LazyImportDict", "Registry", "Namespace", "NAMESPACE"]
