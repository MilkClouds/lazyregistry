"""Tests for core registry functionality."""

import pytest

from lazyregistry import NAMESPACE
from lazyregistry.registry import ImportString, LazyImportDict, Namespace, Registry


class TestImportString:
    """Test ImportString class."""

    def test_load_method(self):
        """Test load() method imports the object."""
        import_str = ImportString("json:dumps")
        func = import_str.load()

        # Should return the actual function
        assert callable(func)
        import json

        assert func is json.dumps

    def test_load_method_invalid_import(self):
        """Test load() method with invalid import string."""
        import_str = ImportString("nonexistent_module:function")

        with pytest.raises(Exception):  # Pydantic will raise an error
            import_str.load()


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

    def test_auto_import_strings_behavior(self):
        """Test auto_import_strings attribute behavior."""
        registry = LazyImportDict()

        # Default: auto_import_strings=True
        registry["auto"] = "json:dumps"
        assert isinstance(registry.data["auto"], ImportString)

        # Disable auto conversion
        registry.auto_import_strings = False
        registry["plain"] = "just a string"
        assert registry.data["plain"] == "just a string"
        assert not isinstance(registry.data["plain"], ImportString)

        # Existing items unaffected by attribute change
        assert isinstance(registry.data["auto"], ImportString)

        # Re-enable
        registry.auto_import_strings = True
        registry["auto2"] = "json:loads"
        assert isinstance(registry.data["auto2"], ImportString)

        # update() respects current settings
        registry.auto_import_strings = False
        registry.update({"c": "plain", "d": "string"})
        assert registry.data["c"] == "plain"
        assert registry.data["d"] == "string"

    def test_eager_load_behavior(self):
        """Test eager_load attribute behavior."""
        registry = LazyImportDict()
        registry.eager_load = True

        # Should be loaded immediately
        registry["json"] = "json:dumps"
        assert not isinstance(registry.data["json"], ImportString)
        assert callable(registry.data["json"])

        # eager_load has no effect when auto_import_strings is False
        registry.auto_import_strings = False
        registry["plain"] = "plain string"
        assert registry.data["plain"] == "plain string"
        assert not isinstance(registry.data["plain"], ImportString)

    def test_attributes_isolated_from_dict(self):
        """Test that attributes are isolated from dict items."""
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

        # Different instances have independent attributes
        registry2 = LazyImportDict()
        assert registry2.auto_import_strings is True
        assert registry2.eager_load is False

        # Attribute names can be used as dict keys without conflict
        registry["auto_import_strings"] = "value1"
        registry["eager_load"] = "value2"
        assert registry.auto_import_strings is False  # Attribute unchanged
        assert registry.eager_load is True  # Attribute unchanged
        assert registry["auto_import_strings"] == "value1"  # Dict item exists
        assert registry["eager_load"] == "value2"  # Dict item exists

    def test_key_error(self):
        """Test KeyError for missing keys."""
        registry = LazyImportDict()
        with pytest.raises(KeyError):
            _ = registry["nonexistent"]


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

    def test_overwrite_registration(self):
        """Test overwriting a registration."""
        registry = Registry(name="test")
        registry["key"] = "json:dumps"
        registry["key"] = "json:loads"

        # Should have the new value
        func = registry["key"]
        assert func.__name__ == "loads"
