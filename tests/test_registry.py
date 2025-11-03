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

    def test_register_import_string(self):
        """Test registering an import string."""
        registry = LazyImportDict()
        registry.register("json", "json:dumps")

        # Should be ImportString before access
        assert isinstance(registry.data["json"], ImportString)

        # Should be loaded after access
        func = registry["json"]
        assert callable(func)
        assert not isinstance(registry.data["json"], ImportString)

    def test_register_instance(self):
        """Test registering an instance directly."""
        registry = LazyImportDict()
        import json

        registry.register("json", json.dumps, is_instance=True)

        # Should be the actual object
        assert registry.data["json"] is json.dumps
        assert registry["json"] is json.dumps

    def test_eager_load(self):
        """Test eager loading."""
        registry = LazyImportDict()
        registry.register("json", "json:dumps", eager_load=True)

        # Should be loaded immediately
        assert not isinstance(registry.data["json"], ImportString)
        assert callable(registry.data["json"])

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
        """Test basic registry usage."""
        registry = Registry(name="serializers")
        registry.register("json", "json:dumps")

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
        ns["models"].register("bert", "json:dumps")
        ns["tokenizers"].register("bert", "json:loads")

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
        NAMESPACE["test_registry"].register("test_key", "json:dumps")
        func = NAMESPACE["test_registry"]["test_key"]
        assert callable(func)


class TestIntegration:
    """Integration tests."""

    def test_mixed_registration(self):
        """Test mixing import strings and instances."""
        registry = Registry(name="mixed")

        # Register import string
        registry.register("lazy", "json:dumps")

        # Register instance
        import json

        registry.register("eager", json.loads, is_instance=True)

        # Both should work
        assert callable(registry["lazy"])
        assert callable(registry["eager"])
        assert registry["eager"] is json.loads

    def test_overwrite_registration(self):
        """Test overwriting a registration."""
        registry = Registry(name="test")
        registry.register("key", "json:dumps")
        registry.register("key", "json:loads")

        # Should have the new value
        func = registry["key"]
        assert func.__name__ == "loads"

    def test_dict_methods(self):
        """Test that dict methods work."""
        registry = Registry(name="test")
        registry.register("a", "json:dumps")
        registry.register("b", "json:loads")

        # Test keys()
        assert set(registry.keys()) == {"a", "b"}

        # Test len()
        assert len(registry) == 2

        # Test in
        assert "a" in registry
        assert "c" not in registry
