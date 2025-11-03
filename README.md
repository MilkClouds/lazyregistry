# lazyregistry

[![CI](https://github.com/milkclouds/lazyregistry/actions/workflows/ci.yml/badge.svg)](https://github.com/milkclouds/lazyregistry/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/milkclouds/lazyregistry/branch/main/graph/badge.svg)](https://codecov.io/gh/milkclouds/lazyregistry)
[![pypi](https://img.shields.io/pypi/v/lazyregistry.svg)](https://pypi.python.org/pypi/lazyregistry)
[![Python Versions](https://img.shields.io/pypi/pyversions/lazyregistry.svg)](https://pypi.org/project/lazyregistry/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A lightweight Python library for lazy-loading registries with namespace support and type safety

## Installation

```bash
# Install with pip
$ pip install lazyregistry

# Add to your project with uv
$ uv add "lazyregistry"
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

Run examples: `uv run python examples/<example>.py`

### 1. Plugin System

[`examples/plugin_system.py`](examples/plugin_system.py) - Extensible plugin architecture with decorator-based registration:

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
    def process(self, text: str) -> str:
        return text.upper()

# Execute plugins
PluginManager.execute("uppercase", "hello")  # "HELLO"
PluginManager.pipeline("hello", "uppercase", "reverse")  # "OLLEH"
```

### 2. Pretrained Models

[`examples/pretrained.py`](examples/pretrained.py) - HuggingFace-style save/load with two patterns:

**Basic (config only):**
```python
from pydantic import BaseModel
from lazyregistry import NAMESPACE
from lazyregistry.pretrained import AutoRegistry, PretrainedMixin

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

# Save and auto-load
model = BertModel(ModelConfig(model_type="bert"))
model.save_pretrained("./model")
loaded = AutoModel.from_pretrained("./model")  # Auto-detects type
```

**Advanced (config + custom state):**
```python
class Tokenizer(PretrainedMixin[TokenizerConfig]):
    def __init__(self, config, vocab: dict[str, int] | None = None):
        super().__init__(config)
        self.vocab = vocab or {}

    def save_pretrained(self, path):
        super().save_pretrained(path)
        # Save additional state (vocabulary)
        Path(path).joinpath("vocab.txt").write_text(...)

    @classmethod
    def from_pretrained(cls, path):
        config = cls.config_class.model_validate_json(...)
        vocab = ...  # Load vocabulary
        return cls(config, vocab=vocab)
```

## API Reference

### Core Classes

**`ImportString`** - String that represents an import path with lazy loading capability
```python
from lazyregistry import ImportString

# Create an import string
import_str = ImportString("json:dumps")

# Load the object when needed
func = import_str.load()
func({"key": "value"})  # '{"key": "value"}'
```

**`Registry[K, V]`** - Named registry with lazy import support
```python
registry = Registry(name="plugins")
registry.register("key", "module:object")           # Lazy
registry.register("key", obj, is_instance=True)     # Immediate
registry.register("key", "module:object", eager_load=True)  # Load now
value = registry["key"]
```

**`Namespace`** - Container for multiple registries
```python
from lazyregistry import NAMESPACE

NAMESPACE["models"].register("bert", "transformers:BertModel")
model = NAMESPACE["models"]["bert"]
```

**`LazyImportDict[K, V]`** - Base class for custom implementations (same API as `Registry` without `name`)

### Pretrained Pattern

**`PretrainedMixin[ConfigT]`** - Save/load with Pydantic config
```python
class MyModel(PretrainedMixin[MyConfig]):
    config_class = MyConfig

model.save_pretrained("./path")
loaded = MyModel.from_pretrained("./path")
```

**`AutoRegistry`** - Auto-detect model type from config
```python
class AutoModel(AutoRegistry):
    registry = NAMESPACE["models"]
    config_class = ModelConfig
    type_key = "model_type"

@AutoModel.register("bert")
class BertModel(PretrainedMixin[ModelConfig]):
    config_class = ModelConfig

loaded = AutoModel.from_pretrained("./path")  # Auto-detects type
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

## Testing

Run tests with coverage:

```bash
uv run pytest tests/ --cov=lazyregistry --cov-report=term-missing
```

The test suite includes:
- **Core registry tests** - LazyImportDict, Registry, Namespace functionality
- **Pretrained tests** - save/load patterns, AutoRegistry, custom state
- **Example tests** - Verify all examples run correctly

## License

MIT
