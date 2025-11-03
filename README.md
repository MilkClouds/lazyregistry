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
- **Pretrained models** - Built-in support for save_pretrained/from_pretrained pattern

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
from pydantic import BaseModel
from lazyregistry import NAMESPACE, AutoRegistry, PretrainedMixin

class ModelConfig(BaseModel):
    model_type: str
    hidden_size: int = 768

class AutoModel(AutoRegistry):
    registry = NAMESPACE["models"]
    config_class = ModelConfig
    type_key = "model_type"

@AutoModel.register("bert")
class BertModel(PretrainedMixin[ModelConfig]):
    config_class = ModelConfig

# Save and load
config = ModelConfig(model_type="bert", hidden_size=1024)
model = BertModel(config)
model.save_pretrained("./my_model")

# Auto-detect model type from config and load
loaded = AutoModel.from_pretrained("./my_model")
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

### `PretrainedMixin[ConfigT]`

Mixin class providing save_pretrained/from_pretrained functionality.

```python
from pydantic import BaseModel
from lazyregistry import PretrainedMixin

class MyConfig(BaseModel):
    name: str
    hidden_size: int = 768

class MyModel(PretrainedMixin[MyConfig]):
    config_class = MyConfig

# Save and load
config = MyConfig(name="my_model")
model = MyModel(config)
model.save_pretrained("./path")
loaded = MyModel.from_pretrained("./path")
```

### `AutoRegistry`

Auto-loader registry for pretrained models with type detection.

```python
from lazyregistry import AutoRegistry, PretrainedMixin, NAMESPACE

class AutoModel(AutoRegistry):
    registry = NAMESPACE["models"]
    config_class = ModelConfig
    type_key = "model_type"  # Field in config to identify model type

@AutoModel.register("bert")
class BertModel(PretrainedMixin[ModelConfig]):
    config_class = ModelConfig

# Automatically detects model type and loads
model = AutoModel.from_pretrained("./path")
```

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
