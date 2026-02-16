"""Custom exceptions for lazyregistry."""

__all__ = ["LazyRegistryError", "ImportFailedError"]


class LazyRegistryError(Exception):
    """Base exception for all lazyregistry errors."""


class ImportFailedError(ImportError, LazyRegistryError):
    """Raised when a lazy import string fails to resolve.

    Inherits from both ``ImportError`` and ``LazyRegistryError`` so that
    callers can catch either:

    * ``ImportError`` — standard Python convention for failed imports.
    * ``LazyRegistryError`` — catch-all for any lazyregistry error.
    * ``ImportFailedError`` — the most specific type.

    The original ``pydantic.ValidationError`` is chained as ``__cause__``
    for full diagnostic details.
    """
