"""
Plugin system example using lazyregistry.

Shows how to build an extensible plugin architecture where plugins
are registered and loaded on-demand.
"""

from lazyregistry import Registry

# Create a plugin registry
PLUGINS: Registry[str, type] = Registry(name="plugins")


class PluginBase:
    """Base class for all plugins."""
    
    def execute(self, data: str) -> str:
        raise NotImplementedError


# Register plugins using decorator pattern
def plugin(name: str):
    """Decorator to register a plugin."""
    def decorator(cls: type) -> type:
        PLUGINS.register(name, cls, is_instance=True)
        return cls
    return decorator


# Define plugins
@plugin("uppercase")
class UppercasePlugin(PluginBase):
    """Convert text to uppercase."""
    
    def execute(self, data: str) -> str:
        return data.upper()


@plugin("reverse")
class ReversePlugin(PluginBase):
    """Reverse text."""
    
    def execute(self, data: str) -> str:
        return data[::-1]


@plugin("repeat")
class RepeatPlugin(PluginBase):
    """Repeat text twice."""
    
    def execute(self, data: str) -> str:
        return data * 2


# Plugin manager
class PluginManager:
    """Manage and execute plugins."""
    
    @staticmethod
    def run(plugin_name: str, data: str) -> str:
        """Run a plugin by name."""
        plugin_class = PLUGINS[plugin_name]
        plugin = plugin_class()
        return plugin.execute(data)
    
    @staticmethod
    def list_plugins() -> list[str]:
        """List all registered plugins."""
        return list(PLUGINS.keys())


# Example usage
if __name__ == "__main__":
    print("Available plugins:", PluginManager.list_plugins())
    
    text = "hello"
    
    # Run different plugins
    print(f"\nOriginal: {text}")
    print(f"Uppercase: {PluginManager.run('uppercase', text)}")
    print(f"Reverse: {PluginManager.run('reverse', text)}")
    print(f"Repeat: {PluginManager.run('repeat', text)}")

