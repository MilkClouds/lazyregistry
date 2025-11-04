"""
Plugin system using lazyregistry.

Demonstrates:
1. Decorator-based plugin registration
2. Lazy loading of plugin classes
3. Extensible architecture for third-party plugins
"""

from lazyregistry import Registry

# Global plugin registry
PLUGINS: Registry[str, type] = Registry(name="plugins")


def plugin(name: str):
    """Decorator to register a plugin by name."""

    def decorator(cls: type) -> type:
        PLUGINS[name] = cls
        return cls

    return decorator


# ============================================================================
# Built-in plugins
# ============================================================================


@plugin("uppercase")
class UppercasePlugin:
    """Convert text to uppercase."""

    def process(self, text: str) -> str:
        return text.upper()


@plugin("lowercase")
class LowercasePlugin:
    """Convert text to lowercase."""

    def process(self, text: str) -> str:
        return text.lower()


@plugin("reverse")
class ReversePlugin:
    """Reverse text."""

    def process(self, text: str) -> str:
        return text[::-1]


@plugin("title")
class TitleCasePlugin:
    """Convert to title case."""

    def process(self, text: str) -> str:
        return text.title()


# ============================================================================
# Plugin manager
# ============================================================================


class PluginManager:
    """Centralized plugin management."""

    @staticmethod
    def execute(plugin_name: str, text: str) -> str:
        """Execute a plugin by name."""
        plugin_class = PLUGINS[plugin_name]  # Lazy load here
        return plugin_class().process(text)

    @staticmethod
    def available() -> list[str]:
        """List all available plugins."""
        return sorted(PLUGINS.keys())

    @staticmethod
    def pipeline(text: str, *plugin_names: str) -> str:
        """Execute multiple plugins in sequence."""
        result = text
        for name in plugin_names:
            result = PluginManager.execute(name, result)
        return result


# ============================================================================
# Demo
# ============================================================================

if __name__ == "__main__":
    print("Available plugins:", PluginManager.available())

    text = "hello world"
    print(f"\nOriginal: '{text}'")

    # Single plugins
    print(f"Uppercase: '{PluginManager.execute('uppercase', text)}'")
    print(f"Title: '{PluginManager.execute('title', text)}'")
    print(f"Reverse: '{PluginManager.execute('reverse', text)}'")

    # Pipeline
    result = PluginManager.pipeline(text, "uppercase", "reverse")
    print(f"\nPipeline (uppercase -> reverse): '{result}'")
