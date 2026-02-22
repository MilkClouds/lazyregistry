"""
Core registry classes for lazy import management.

References:
- Namespace concept from Python official tutorial:
  https://docs.python.org/3/tutorial/classes.html
- Entry point design (group/name/object reference pattern):
  https://packaging.python.org/en/latest/specifications/entry-points/
  Note: Adopts nothing from entry point implementation, but refers to the group/name/object reference design.
- Registry pattern with parent/scope/location from mmengine:
  https://mmengine.readthedocs.io/en/latest/advanced_tutorials/registry.html
  https://github.com/open-mmlab/mmdetection/blob/main/mmdet/registry.py
"""

from __future__ import annotations

import sys
from collections import UserDict
from typing import TYPE_CHECKING, Generic, TypeVar

from pydantic import ImportString as PydanticImportString
from pydantic import TypeAdapter, ValidationError

from .exceptions import ImportFailedError

__all__ = ["ImportString", "LazyImportDict", "Registry", "Namespace", "NAMESPACE"]

_import_adapter = TypeAdapter(PydanticImportString)

K = TypeVar("K")
V = TypeVar("V")


class ImportString(str):
    """String that represents an import path.

    Examples:
        >>> import_str = ImportString("json:dumps")
        >>> func = import_str.load()
        >>> func({"key": "value"})
        '{"key": "value"}'
    """

    def load(self):
        """Import and return the object referenced by this import string.

        Raises:
            ImportFailedError: If the import string cannot be resolved.
                The original ``pydantic.ValidationError`` is chained as
                ``__cause__`` for full diagnostic details.
        """
        try:
            return _import_adapter.validate_python(self)
        except ValidationError as e:
            raise ImportFailedError(f"Failed to import '{self}'") from e


class LazyImportDict(UserDict, Generic[K, V]):
    """Dictionary that lazily imports values as needed.

    Attributes:
        auto_import_strings: If True, string values are automatically converted to ImportString.
        eager_load: If True, values are immediately loaded upon assignment.

    Examples:
        >>> registry = LazyImportDict()
        >>> registry["json"] = "json:dumps"  # Auto-converted to ImportString
        >>> registry.update({"pickle": "pickle:dumps"})
    """

    data: dict[K, V]  # on python>=3.9, it's enough to remove this line and use `UserDict[K, V]` as base class
    auto_import_strings: bool = True
    eager_load: bool = False

    def __setitem__(self, key: K, item: V) -> None:
        if self.auto_import_strings and isinstance(item, str):
            self.data[key] = ImportString(item)  # type: ignore[assignment]
        else:
            self.data[key] = item

        if self.eager_load:
            _ = self[key]

    def __getitem__(self, key: K) -> V:
        value = self.data[key]
        if isinstance(value, ImportString):
            self.data[key] = value.load()
        return self.data[key]


class Registry(LazyImportDict[K, V], Generic[K, V]):
    """A named registry with lazy import support.

    Examples:
        >>> registry = Registry(name="plugins")
        >>> registry["my_plugin"] = "mypackage.plugins:MyPlugin"
        >>> plugin = registry["my_plugin"]  # Lazily imported on first access
    """

    def __init__(self, *args, name: str, **kwargs):
        self.name = name
        super().__init__(*args, **kwargs)


if TYPE_CHECKING:
    _NamespaceBase = UserDict[str, Registry]
else:  # pragma: no cover
    if sys.version_info >= (3, 9):
        _NamespaceBase = UserDict[str, Registry]
    else:
        # UserDict in Python 3.8 does not support subscription
        _NamespaceBase = UserDict


class Namespace(_NamespaceBase):
    """Container for multiple named registries.

    Each registry is completely isolated from others.

    Examples:
        >>> namespace = Namespace()
        >>> namespace["plugins"]["my_plugin"] = "mypackage:MyPlugin"
        >>> namespace["handlers"]["my_handler"] = "mypackage:MyHandler"
        >>> plugin = namespace["plugins"]["my_plugin"]
    """

    def __missing__(self, key: str) -> Registry:
        self.data[key] = Registry(name=key)
        return self.data[key]


# Global namespace instance
NAMESPACE = Namespace()
