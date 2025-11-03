# lazyregistry

A lightweight Python library for lazy-loading registries with namespace support and type safety.

## Installation

```bash
pip install lazyregistry
```

## Quick Start

```python
from lazyregistry import Registry

registry = Registry(name="plugins")

# Register by import string (lazy - imported on access)
registry.register("json", "json:dumps")

# Register by instance (immediate - already imported)
import pickle
registry.register("pickle", pickle.dumps, is_instance=True)

# Import happens here
serializer = registry["json"]
```

## Features

- **Lazy imports** - Defer expensive imports until first access
- **Instance registration** - Register both import strings and direct objects
- **Namespaces** - Organize multiple registries
- **Type-safe** - Full generic type support
- **Eager loading** - Optional immediate import for critical components

## Examples

Run examples with: `uv run python examples/<example>.py`

### Basic Usage

See [`examples/basic_usage.py`](examples/basic_usage.py):

```python
from lazyregistry import Registry, NAMESPACE

# Simple registry
registry = Registry(name="plugins")
registry.register("json", "json:dumps")
registry.register("pickle", "pickle:dumps")

# Using global namespace
NAMESPACE["parsers"].register("json", "json:loads")
NAMESPACE["writers"].register("json", "json:dumps")

# Access from anywhere
parser = NAMESPACE["parsers"]["json"]
```

### Plugin System

See [`examples/plugin_system.py`](examples/plugin_system.py):

```python
from lazyregistry import Registry

PLUGINS = Registry(name="plugins")

def plugin(name: str):
    def decorator(cls):
        PLUGINS.register(name, cls, is_instance=True)
        return cls
    return decorator

@plugin("uppercase")
class UppercasePlugin:
    def execute(self, data: str) -> str:
        return data.upper()

# Use plugin
plugin_class = PLUGINS["uppercase"]
result = plugin_class().execute("hello")  # "HELLO"
```

### Pretrained Models (HuggingFace-style)

See [`examples/pretrained.py`](examples/pretrained.py) for a complete implementation:

```python
from lazyregistry import NAMESPACE

MODEL_REGISTRY = NAMESPACE["models"]

class AutoModel:
    @classmethod
    def register(cls, model_type: str):
        def decorator(model_class):
            MODEL_REGISTRY.register(model_type, model_class, is_instance=True)
            return model_class
        return decorator

@AutoModel.register("bert")
class BertModel:
    def save_pretrained(self, path): ...

    @classmethod
    def from_pretrained(cls, path): ...

# Auto-detect model type from config and load
model = AutoModel.from_pretrained("./saved_model")
```

## API

### `Registry[K, V]`

```python
registry = Registry(name="my_registry")

# Register import string (lazy)
registry.register("key", "module:object")

# Register instance (immediate)
registry.register("key", obj, is_instance=True)

# Eager load (import immediately)
registry.register("key", "module:object", eager_load=True)

# Access
value = registry["key"]
```

### `Namespace`

```python
from lazyregistry import NAMESPACE

# Auto-creates registries
NAMESPACE["models"].register("bert", "transformers:BertModel")
NAMESPACE["tokenizers"].register("bert", "transformers:BertTokenizer")

# Access
model = NAMESPACE["models"]["bert"]
```

### `LazyImportDict[K, V]`

Base class for custom implementations. Same API as `Registry` but without `name` parameter.

## Why?

**Before:**
```python
# All imports happen upfront
from heavy_module_1 import ClassA
from heavy_module_2 import ClassB
from heavy_module_3 import ClassC

REGISTRY = {"a": ClassA, "b": ClassB, "c": ClassC}
```

**After:**
```python
# Import only what you use
from lazyregistry import Registry

registry = Registry(name="components")
registry.register("a", "heavy_module_1:ClassA")
registry.register("b", "heavy_module_2:ClassB")
registry.register("c", "heavy_module_3:ClassC")

# Only ClassA is imported
component = registry["a"]
```

## License

MIT
