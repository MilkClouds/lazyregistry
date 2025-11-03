"""
HuggingFace-style pretrained model pattern using lazyregistry.

This example shows how to build an auto-loader system similar to
transformers.AutoModel with save_pretrained/from_pretrained support.
"""

from pathlib import Path
from typing import ClassVar, Type
from pydantic import BaseModel
from lazyregistry import NAMESPACE, Registry

# Create a registry for models
MODEL_REGISTRY: Registry[str, Type["ModelBase"]] = NAMESPACE["models"]


class ModelConfig(BaseModel):
    """Model configuration."""
    model_type: str
    hidden_size: int = 768
    num_layers: int = 12


class ModelBase:
    """Base class with save/load functionality."""

    config_class: ClassVar[Type[ModelConfig]] = ModelConfig

    def __init__(self, config: ModelConfig):
        self.config = config

    def save_pretrained(self, path: str) -> None:
        """Save model configuration."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        (save_path / "config.json").write_text(
            self.config.model_dump_json(indent=2)
        )

    @classmethod
    def from_pretrained(cls, path: str) -> "ModelBase":
        """Load model from saved configuration."""
        config_file = Path(path) / "config.json"
        config = cls.config_class.model_validate_json(config_file.read_text())
        return cls(config)


class AutoModel:
    """Auto-loader for registered models."""

    @classmethod
    def register(cls, model_type: str):
        """Decorator to register a model class."""
        def decorator(model_class: Type[ModelBase]) -> Type[ModelBase]:
            MODEL_REGISTRY.register(model_type, model_class, is_instance=True)
            return model_class
        return decorator

    @classmethod
    def from_pretrained(cls, path: str) -> ModelBase:
        """Load any registered model from path."""
        config_file = Path(path) / "config.json"
        config = ModelConfig.model_validate_json(config_file.read_text())
        model_class = MODEL_REGISTRY[config.model_type]
        return model_class.from_pretrained(path)


# Register model implementations
@AutoModel.register("bert")
class BertModel(ModelBase):
    """BERT model implementation."""
    pass


@AutoModel.register("gpt2")
class GPT2Model(ModelBase):
    """GPT-2 model implementation."""
    pass


@AutoModel.register("t5")
class T5Model(ModelBase):
    """T5 model implementation."""
    pass


# Example usage
if __name__ == "__main__":
    import tempfile

    # Create and save a model
    config = ModelConfig(model_type="bert", hidden_size=768)
    model = BertModel(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save
        model.save_pretrained(tmpdir)
        print(f"Saved model to {tmpdir}")

        # Load using AutoModel (automatically detects model_type)
        loaded_model = AutoModel.from_pretrained(tmpdir)
        print(f"Loaded: {type(loaded_model).__name__}")
        print(f"Config: {loaded_model.config}")
