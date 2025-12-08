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

from collections import UserDict
from typing import Generic, TypeVar

from pydantic import ImportString as PydanticImportString
from pydantic import TypeAdapter

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
        """Import and return the object referenced by this import string."""
        return _import_adapter.validate_python(self)


class LazyImportDict(UserDict[K, V], Generic[K, V]):
    """Dictionary that lazily imports values as needed.

    Attributes:
        auto_import_strings: If True, string values are automatically converted to ImportString.
        eager_load: If True, values are immediately loaded upon assignment.

    Examples:
        >>> registry = LazyImportDict()
        >>> registry["json"] = "json:dumps"  # Auto-converted to ImportString
        >>> registry.update({"pickle": "pickle:dumps"})
    """

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


class Namespace(UserDict[str, Registry]):
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
