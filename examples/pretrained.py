"""
HuggingFace-style pretrained model pattern using lazyregistry.

This example shows how to build an auto-loader system similar to
transformers.AutoModel with save_pretrained/from_pretrained support.
"""

from pydantic import BaseModel
from lazyregistry import NAMESPACE, AutoRegistry, PretrainedMixin


class ModelConfig(BaseModel):
    """Model configuration."""
    model_type: str
    hidden_size: int = 768
    num_layers: int = 12


class AutoModel(AutoRegistry):
    """Auto-loader for registered models."""
    registry = NAMESPACE["models"]
    config_class = ModelConfig
    type_key = "model_type"


# Register model implementations
@AutoModel.register("bert")
class BertModel(PretrainedMixin[ModelConfig]):
    """BERT model implementation."""
    config_class = ModelConfig


@AutoModel.register("gpt2")
class GPT2Model(PretrainedMixin[ModelConfig]):
    """GPT-2 model implementation."""
    config_class = ModelConfig


@AutoModel.register("t5")
class T5Model(PretrainedMixin[ModelConfig]):
    """T5 model implementation."""
    config_class = ModelConfig


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
