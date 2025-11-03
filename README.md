# lazyregistry

A lightweight Python library for lazy import registries with namespace support. Defer expensive imports until they're actually needed, while maintaining clean and organized code.

## Features

- **Lazy Imports**: Import modules only when accessed, not when registered
- **Type-Safe**: Full generic type support with proper type hints
- **Namespace Support**: Organize registries into isolated namespaces
- **Flexible Registration**: Register by import string or direct instance
- **Eager Loading**: Optional immediate loading for critical components
- **Pydantic Integration**: Built on Pydantic's robust import validation

## Installation

```bash
pip install lazyregistry
```

## Quick Start

### Basic Registry Usage

```python
from lazyregistry import Registry

# Create a registry
plugins = Registry(name="plugins")

# Register a plugin by import string (lazy)
plugins.register("my_plugin", "mypackage.plugins:MyPlugin")

# Plugin is imported only when accessed
plugin_class = plugins["my_plugin"]  # Import happens here
```

### Using Namespaces

Organize multiple registries under a single namespace:

```python
from lazyregistry import Namespace

namespace = Namespace()

# Auto-creates registries on first access
namespace["models"].register("bert", "transformers:BertModel")
namespace["tokenizers"].register("bert", "transformers:BertTokenizer")
namespace["processors"].register("image", "PIL.Image:Image")

# Access registered items
model = namespace["models"]["bert"]
tokenizer = namespace["tokenizers"]["bert"]
```

### Global Namespace

Use the pre-configured global namespace:

```python
from lazyregistry import NAMESPACE

# Register across your application
NAMESPACE["handlers"].register("http", "myapp.handlers:HTTPHandler")
NAMESPACE["handlers"].register("websocket", "myapp.handlers:WebSocketHandler")

# Access anywhere in your code
handler = NAMESPACE["handlers"]["http"]
```

### Direct Instance Registration

Register already-imported objects:

```python
from lazyregistry import Registry
from myapp import MyClass

registry = Registry(name="components")

# Register an instance directly
registry.register("my_component", MyClass, is_instance=True)

# Or register an import string (default)
registry.register("other_component", "myapp:OtherClass")
```

### Eager Loading

Force immediate import for critical components:

```python
registry = Registry(name="critical")

# Import immediately, not lazily
registry.register(
    "database",
    "myapp.db:Database",
    eager_load=True  # Imported right away
)
```

### LazyImportDict

Use the base dictionary class for custom implementations:

```python
from lazyregistry import LazyImportDict

# Generic lazy import dictionary
cache: LazyImportDict[str, type] = LazyImportDict()

cache.register("numpy", "numpy:ndarray")
cache.register("pandas", "pandas:DataFrame")

# Imports happen on access
np_array = cache["numpy"]
df = cache["pandas"]
```

## Advanced Example: Pretrained Model Pattern

Build a HuggingFace-style auto-loader with registries:

```python
from pathlib import Path
from typing import ClassVar, Type
from pydantic import BaseModel
from lazyregistry import NAMESPACE, Registry

# Create a registry for models
MODEL_REGISTRY: Registry[str, Type["ModelMixin"]] = NAMESPACE["models"]

class ModelConfig(BaseModel):
    name: str
    hidden_size: int = 768

class ModelMixin:
    config_cls: ClassVar[Type[ModelConfig]]

    def __init__(self, config: ModelConfig):
        self.config = config

    def save_pretrained(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "config.json").write_text(
            self.config.model_dump_json()
        )

    @classmethod
    def from_pretrained(cls, path: str):
        config_file = Path(path) / "config.json"
        config = cls.config_cls.model_validate_json(
            config_file.read_text()
        )
        return cls(config)

class AutoModel:
    @classmethod
    def register(cls, name: str):
        def decorator(model_cls: Type[ModelMixin]):
            MODEL_REGISTRY.register(name, model_cls, is_instance=True)
            return model_cls
        return decorator

    @classmethod
    def from_pretrained(cls, path: str):
        config = ModelConfig.model_validate_json(
            (Path(path) / "config.json").read_text()
        )
        model_cls = MODEL_REGISTRY[config.name]
        return model_cls.from_pretrained(path)

# Register models
@AutoModel.register("bert")
class BertModel(ModelMixin):
    config_cls = ModelConfig

@AutoModel.register("gpt")
class GPTModel(ModelMixin):
    config_cls = ModelConfig

# Use the auto-loader
model = AutoModel.from_pretrained("./saved_model")
```

## API Reference

### `LazyImportDict[K, V]`

Base dictionary class with lazy import support.

**Methods:**
- `register(key: K, value: V | str, *, is_instance: bool = False, eager_load: bool = False) -> None`
  - Register a value by key
  - `is_instance=True`: value is already imported
  - `is_instance=False`: value is an import string
  - `eager_load=True`: import immediately

### `Registry[K, V]`

Named registry extending `LazyImportDict`.

**Constructor:**
- `Registry(name: str)`: Create a named registry

### `Namespace`

Container for multiple named registries.

**Usage:**
- `namespace[registry_name]`: Auto-creates registry if not exists
- Each registry is completely isolated

### `NAMESPACE`

Pre-configured global `Namespace` instance for application-wide use.

## Why lazyregistry?

### Problem

```python
# Traditional approach - all imports happen upfront
from heavy_module_1 import ClassA
from heavy_module_2 import ClassB
from heavy_module_3 import ClassC

REGISTRY = {
    "a": ClassA,  # Imported even if never used
    "b": ClassB,  # Imported even if never used
    "c": ClassC,  # Imported even if never used
}
```

### Solution

```python
# lazyregistry - imports only what you use
from lazyregistry import Registry

registry = Registry(name="components")
registry.register("a", "heavy_module_1:ClassA")
registry.register("b", "heavy_module_2:ClassB")
registry.register("c", "heavy_module_3:ClassC")

# Only ClassA is imported
component = registry["a"]
```

## Inspiration

This library draws inspiration from:

- **Python Namespaces**: Official Python namespace concepts
- **Entry Points**: The group/name/object reference pattern from Python packaging
- **MMEngine Registry**: Parent/scope/location patterns from OpenMMLab
- **HuggingFace Transformers**: Auto-classes and pretrained model patterns

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
