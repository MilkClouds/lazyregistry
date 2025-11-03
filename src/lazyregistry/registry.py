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
from typing import Generic, TypeVar, Union, overload

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
        """Import and return the object referenced by this import string.

        Returns:
            The imported object.

        Raises:
            pydantic.ValidationError: If the import string is invalid or the module/attribute cannot be imported.

        Examples:
            >>> import_str = ImportString("json:dumps")
            >>> func = import_str.load()
            >>> callable(func)
            True
        """
        return _import_adapter.validate_python(self)


class LazyImportDict(UserDict[K, V], Generic[K, V]):
    """Dictionary that lazily imports values as needed."""

    @overload
    def register(self, key: K, value: V, *, is_instance: bool = True, eager_load: bool = False) -> None: ...

    @overload
    def register(self, key: K, value: str, *, is_instance: bool = False, eager_load: bool = False) -> None: ...

    def register(self, key: K, value: Union[V, str], *, is_instance: bool = False, eager_load: bool = False) -> None:
        """Register a value in the dictionary.

        Args:
            key: The key to register the value under.
            value: Either an instance of V or an import string.
            is_instance: If True, value is already an instance of V.
                        If False, value is an import string.
            eager_load: If True, immediately load the value.
        """
        if is_instance:
            self.data[key] = value  # type: ignore[assignment]
        else:
            self.data[key] = ImportString(value)  # type: ignore[assignment]
        if eager_load:
            self[key]

    def __getitem__(self, key: K) -> V:
        value = self.data[key]
        if isinstance(value, ImportString):
            self.data[key] = _import_adapter.validate_python(value)  # type: ignore[assignment]
        return self.data[key]


class Registry(LazyImportDict[K, V], Generic[K, V]):
    """A named registry with lazy import support.

    Examples:
        >>> registry = Registry(name="plugins")
        >>> registry.register("my_plugin", "mypackage.plugins:MyPlugin")
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
        >>> namespace["plugins"].register("my_plugin", "mypackage:MyPlugin")
        >>> namespace["handlers"].register("my_handler", "mypackage:MyHandler")
        >>> plugin = namespace["plugins"]["my_plugin"]
    """

    def __missing__(self, key: str) -> Registry:
        self.data[key] = Registry(name=key)
        return self.data[key]


# Global namespace instance
NAMESPACE = Namespace()
