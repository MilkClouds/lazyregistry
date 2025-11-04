"""Tests for core registry functionality."""

import pytest

from lazyregistry import NAMESPACE
from lazyregistry.registry import ImportString, LazyImportDict, Namespace, Registry


class TestImportString:
    """Test ImportString class."""

    def test_import_string_is_str(self):
        """ImportString should be a string subclass."""
        s = ImportString("json:dumps")
        assert isinstance(s, str)
        assert s == "json:dumps"

    def test_load_method(self):
        """Test load() method imports the object."""
        import_str = ImportString("json:dumps")
        func = import_str.load()

        # Should return the actual function
        assert callable(func)
        import json

        assert func is json.dumps

    def test_load_method_with_module(self):
        """Test load() method with module import."""
        import_str = ImportString("json")
        module = import_str.load()

        # Should return the module
        import json

        assert module is json

    def test_load_method_invalid_import(self):
        """Test load() method with invalid import string."""
        import_str = ImportString("nonexistent_module:function")

        with pytest.raises(Exception):  # Pydantic will raise an error
            import_str.load()

    def test_load_method_multiple_calls(self):
        """Test that load() can be called multiple times."""
        import_str = ImportString("json:loads")
        func1 = import_str.load()
        func2 = import_str.load()

        # Should return the same object
        assert func1 is func2


class TestLazyImportDict:
    """Test LazyImportDict class."""

    def test_setitem_auto_import_strings(self):
        """Test __setitem__ with auto_import_strings=True (default)."""
        registry = LazyImportDict()
        registry["json"] = "json:dumps"

        # Should be ImportString before access
        assert isinstance(registry.data["json"], ImportString)

        # Should be loaded after access
        func = registry["json"]
        assert callable(func)
        assert not isinstance(registry.data["json"], ImportString)

    def test_setitem_with_instance(self):
        """Test __setitem__ with actual instance."""
        registry = LazyImportDict()
        import json

        registry["json"] = json.dumps

        # Should be the actual object
        assert registry.data["json"] is json.dumps
        assert registry["json"] is json.dumps

    def test_update_method(self):
        """Test update() method works with auto_import_strings."""
        registry = LazyImportDict()
        registry.update(
            {
                "json": "json:dumps",
                "pickle": "pickle:dumps",
            }
        )

        # Both should be ImportStrings
        assert isinstance(registry.data["json"], ImportString)
        assert isinstance(registry.data["pickle"], ImportString)

        # Access should load them
        import json

        assert registry["json"] is json.dumps

    def test_auto_import_strings_toggle(self):
        """Test toggling auto_import_strings attribute."""
        registry = LazyImportDict()

        # Default: auto_import_strings=True
        registry["auto"] = "json:dumps"
        assert isinstance(registry.data["auto"], ImportString)

        # Disable auto conversion
        registry.auto_import_strings = False
        registry["plain"] = "just a string"
        assert registry.data["plain"] == "just a string"
        assert not isinstance(registry.data["plain"], ImportString)

        # Re-enable
        registry.auto_import_strings = True
        registry["auto2"] = "json:loads"
        assert isinstance(registry.data["auto2"], ImportString)

    def test_eager_load_attribute(self):
        """Test eager_load attribute."""
        registry = LazyImportDict()
        registry.eager_load = True

        registry["json"] = "json:dumps"

        # Should be loaded immediately
        assert not isinstance(registry.data["json"], ImportString)
        assert callable(registry.data["json"])

    def test_key_error(self):
        """Test KeyError for missing keys."""
        registry = LazyImportDict()
        with pytest.raises(KeyError):
            _ = registry["nonexistent"]

    def test_attribute_not_in_dict(self):
        """Test that class attributes don't appear as dict items."""
        registry = LazyImportDict()

        # Set attributes
        registry.auto_import_strings = False
        registry.eager_load = True

        # Attributes should NOT be in the dictionary
        assert "auto_import_strings" not in registry
        assert "eager_load" not in registry
        assert "auto_import_strings" not in registry.data
        assert "eager_load" not in registry.data

        # Add a real item
        registry["real_item"] = "value"
        assert "real_item" in registry
        assert len(registry) == 1

    def test_multiple_instances_independent(self):
        """Test that different instances have independent attributes."""
        registry1 = LazyImportDict()
        registry2 = LazyImportDict()

        # Change registry1 settings
        registry1.auto_import_strings = False
        registry1.eager_load = True

        # registry2 should have default settings
        assert registry2.auto_import_strings is True
        assert registry2.eager_load is False

        # Add items
        registry1["item"] = "string"
        registry2["item"] = "json:dumps"

        # registry1: plain string (auto_import_strings=False)
        assert registry1.data["item"] == "string"

        # registry2: ImportString (auto_import_strings=True, eager_load=False)
        assert isinstance(registry2.data["item"], ImportString)

    def test_existing_items_unaffected_by_attribute_change(self):
        """Test that changing attributes doesn't affect existing items."""
        registry = LazyImportDict()

        # Add item with auto_import_strings=True
        registry["item1"] = "json:dumps"
        assert isinstance(registry.data["item1"], ImportString)

        # Change attribute
        registry.auto_import_strings = False

        # Existing item should still be ImportString
        assert isinstance(registry.data["item1"], ImportString)

        # New item should be plain string
        registry["item2"] = "just a string"
        assert registry.data["item2"] == "just a string"

    def test_update_respects_current_settings(self):
        """Test that update() method respects current attribute settings."""
        registry = LazyImportDict()

        # Default settings: auto_import_strings=True
        registry.update({"a": "json:dumps", "b": "json:loads"})
        assert isinstance(registry.data["a"], ImportString)
        assert isinstance(registry.data["b"], ImportString)

        # Change settings
        registry.auto_import_strings = False
        registry.update({"c": "plain", "d": "string"})
        assert registry.data["c"] == "plain"
        assert registry.data["d"] == "string"

    def test_attribute_name_as_dict_key(self):
        """Test that attribute names can be used as dict keys without conflict."""
        registry = LazyImportDict()

        # Set attributes
        registry.auto_import_strings = False
        registry.eager_load = True

        # Use same names as dict keys
        registry["auto_import_strings"] = "value1"
        registry["eager_load"] = "value2"

        # Attributes should be unchanged
        assert registry.auto_import_strings is False
        assert registry.eager_load is True

        # Dict items should exist
        assert registry["auto_import_strings"] == "value1"
        assert registry["eager_load"] == "value2"

    def test_eager_load_with_auto_import_disabled(self):
        """Test that eager_load has no effect when auto_import_strings is False."""
        registry = LazyImportDict()
        registry.auto_import_strings = False
        registry.eager_load = True

        # Should store as plain string (auto_import takes precedence)
        registry["item"] = "plain string"
        assert registry.data["item"] == "plain string"
        assert not isinstance(registry.data["item"], ImportString)


class TestRegistry:
    """Test Registry class."""

    def test_registry_has_name(self):
        """Registry should have a name attribute."""
        registry = Registry(name="test")
        assert registry.name == "test"

    def test_registry_basic_usage(self):
        """Test basic registry usage with dict-style assignment."""
        registry = Registry(name="serializers")
        registry["json"] = "json:dumps"

        func = registry["json"]
        assert callable(func)
        result = func({"key": "value"})
        assert isinstance(result, str)


class TestNamespace:
    """Test Namespace class."""

    def test_namespace_auto_creates_registry(self):
        """Namespace should auto-create registries."""
        ns = Namespace()
        assert "models" not in ns.data

        # Access should create registry
        registry = ns["models"]
        assert isinstance(registry, Registry)
        assert registry.name == "models"
        assert "models" in ns.data

    def test_namespace_isolation(self):
        """Registries in namespace should be isolated."""
        ns = Namespace()
        ns["models"]["bert"] = "json:dumps"
        ns["tokenizers"]["bert"] = "json:loads"

        # Different registries should have different values
        model_func = ns["models"]["bert"]
        tokenizer_func = ns["tokenizers"]["bert"]
        assert model_func is not tokenizer_func


class TestGlobalNamespace:
    """Test global NAMESPACE instance."""

    def test_global_namespace_exists(self):
        """Global NAMESPACE should exist."""
        assert isinstance(NAMESPACE, Namespace)

    def test_global_namespace_usage(self):
        """Test using global NAMESPACE."""
        NAMESPACE["test_registry"]["test_key"] = "json:dumps"
        func = NAMESPACE["test_registry"]["test_key"]
        assert callable(func)


class TestIntegration:
    """Integration tests."""

    def test_mixed_registration(self):
        """Test mixing import strings and instances."""
        registry = Registry(name="mixed")

        # Register import string (auto-converted)
        registry["lazy"] = "json:dumps"

        # Register instance
        import json

        registry["eager"] = json.loads

        # Both should work
        assert callable(registry["lazy"])
        assert callable(registry["eager"])
        assert registry["eager"] is json.loads

    def test_overwrite_registration(self):
        """Test overwriting a registration."""
        registry = Registry(name="test")
        registry["key"] = "json:dumps"
        registry["key"] = "json:loads"

        # Should have the new value
        func = registry["key"]
        assert func.__name__ == "loads"

    def test_dict_methods(self):
        """Test that dict methods work."""
        registry = Registry(name="test")
        registry["a"] = "json:dumps"
        registry["b"] = "json:loads"

        # Test keys()
        assert set(registry.keys()) == {"a", "b"}

        # Test len()
        assert len(registry) == 2

        # Test in
        assert "a" in registry
        assert "c" not in registry
