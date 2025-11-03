"""Basic usage examples for lazyregistry."""

from lazyregistry import Registry, NAMESPACE


def example_1_basic_registry():
    """Example 1: Basic registry with lazy imports."""
    print("=== Example 1: Basic Registry ===")
    
    registry = Registry(name="plugins")
    
    # Register by import string (lazy)
    registry.register("json", "json:dumps")
    registry.register("pickle", "pickle:dumps")
    
    # Import happens only when accessed
    json_dumps = registry["json"]
    print(f"Loaded: {json_dumps}")
    # pickle is never imported if not used


def example_2_instance_registration():
    """Example 2: Register direct instances."""
    print("\n=== Example 2: Instance Registration ===")
    
    import json
    import pickle
    
    registry = Registry(name="serializers")
    
    # Register already-imported objects
    registry.register("json", json.dumps, is_instance=True)
    registry.register("pickle", pickle.dumps, is_instance=True)
    
    # No import needed - already available
    serializer = registry["json"]
    print(f"Serializer: {serializer}")


def example_3_namespace():
    """Example 3: Using namespaces to organize registries."""
    print("\n=== Example 3: Namespaces ===")
    
    # Auto-creates registries on access
    NAMESPACE["parsers"].register("json", "json:loads")
    NAMESPACE["parsers"].register("yaml", "yaml:safe_load")
    
    NAMESPACE["writers"].register("json", "json:dumps")
    NAMESPACE["writers"].register("yaml", "yaml:dump")
    
    # Access from anywhere
    json_parser = NAMESPACE["parsers"]["json"]
    json_writer = NAMESPACE["writers"]["json"]
    
    print(f"Parser: {json_parser}")
    print(f"Writer: {json_writer}")


def example_4_eager_loading():
    """Example 4: Eager loading for critical components."""
    print("\n=== Example 4: Eager Loading ===")
    
    registry = Registry(name="critical")
    
    # Import immediately
    registry.register("logging", "logging:getLogger", eager_load=True)
    print("Logger imported immediately!")
    
    # Import lazily (default)
    registry.register("optional", "json:dumps")
    print("Optional not imported yet")


if __name__ == "__main__":
    example_1_basic_registry()
    example_2_instance_registration()
    example_3_namespace()
    example_4_eager_loading()

