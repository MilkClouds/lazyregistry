"""
Pretrained model pattern support for lazyregistry.

Provides base classes and utilities for implementing HuggingFace-style
save_pretrained/from_pretrained patterns with automatic model registration.
"""

import os
from pathlib import Path
from typing import Any, ClassVar, Type, Union

from pydantic import BaseModel

from lazyregistry.registry import Registry

__all__ = ["PathLike", "PretrainedConfig", "PretrainedMixin", "AutoRegistry"]

PathLike = Union[str, os.PathLike]


class PretrainedConfig(BaseModel):
    """
    Base configuration class for pretrained models.

    All model-specific configs should inherit from this class and set
    a hardcoded model_type value.

    Example:
        >>> class BertConfig(PretrainedConfig):
        ...     model_type: str = "bert"
        ...     hidden_size: int = 768
        ...     num_layers: int = 12
    """

    model_type: str


class PretrainedMixin:
    """
    Mixin class providing save_pretrained/from_pretrained functionality.

    Classes inheriting from this mixin can be saved to and loaded from disk
    with their configuration automatically serialized/deserialized.

    Recommended pattern: Create a base model class and have each specific model
    inherit from it. Each model should have its own config class (inheriting from
    PretrainedConfig) with a hardcoded model_type field for use with AutoRegistry.

    Example:
        >>> # Model-specific config with hardcoded type
        >>> class BertConfig(PretrainedConfig):
        ...     model_type: str = "bert"
        ...     hidden_size: int = 768

        >>> # Base model class
        >>> class BaseModel(PretrainedMixin):
        ...     config_class = PretrainedConfig

        >>> # Specific model inherits from base
        >>> class BertModel(BaseModel):
        ...     config_class = BertConfig

        >>> # No need to specify model_type when creating config
        >>> config = BertConfig(hidden_size=1024)
        >>> model = BertModel(config)
        >>> model.save_pretrained("./bert_model")
        >>> loaded = BertModel.from_pretrained("./bert_model")
    """

    config_class: ClassVar[Type[PretrainedConfig]]
    config_filename: ClassVar[str] = "config.json"

    def __init__(self, config: PretrainedConfig):
        self.config = config

    def save_pretrained(self, save_directory: PathLike) -> None:
        """Save the model configuration to a directory."""
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        config_file = save_path / self.config_filename
        config_file.write_text(self.config.model_dump_json(indent=2))

    @classmethod
    def from_pretrained(cls, pretrained_path: PathLike, **kwargs: Any) -> "PretrainedMixin":
        """Load a model from a saved configuration."""
        config_file = Path(pretrained_path) / cls.config_filename
        config = cls.config_class.model_validate_json(config_file.read_text())
        return cls(config, **kwargs)  # type: ignore[arg-type]


class AutoRegistry:
    """
    Auto-loader registry for pretrained models.

    Provides decorator-based registration and automatic model loading
    based on configuration type detection.

    Recommended pattern: Create a base model class and base config class. Each
    specific model inherits from the base model and has its own config class
    (inheriting from PretrainedConfig) with a hardcoded model_type value.

    Registration methods:
        1. Decorator: @AutoModel.register_module("model_type")
        2. Direct: AutoModel.registry["model_type"] = ModelClass
        3. Bulk: AutoModel.registry.update({...})

    Example:
        >>> from lazyregistry import NAMESPACE

        >>> # Model-specific configs with hardcoded model_type
        >>> class BertConfig(PretrainedConfig):
        ...     model_type: str = "bert"
        ...     hidden_size: int = 768

        >>> class GPT2Config(PretrainedConfig):
        ...     model_type: str = "gpt2"
        ...     hidden_size: int = 768

        >>> # Base model class
        >>> class BaseModel(PretrainedMixin):
        ...     config_class = PretrainedConfig

        >>> class AutoModel(AutoRegistry):
        ...     registry = NAMESPACE["models"]
        ...     config_class = PretrainedConfig
        ...     type_key = "model_type"

        >>> # Decorator registration - models inherit from BaseModel
        >>> @AutoModel.register_module("bert")
        ... class BertModel(BaseModel):
        ...     config_class = BertConfig

        >>> # Direct registration via .registry
        >>> AutoModel.registry["gpt2"] = GPT2Model
        >>> AutoModel.registry["t5"] = "transformers:T5Model"

        >>> # Bulk registration via .registry.update() - useful for many models
        >>> AutoModel.registry.update({
        ...     "roberta": RobertaModel,
        ...     "albert": "transformers:AlbertModel",
        ...     "electra": "transformers:ElectraModel",
        ... })

        >>> # Auto-detect and load based on config.model_type
        >>> config = BertConfig(hidden_size=1024)
        >>> model = BertModel(config)
        >>> model.save_pretrained("./saved_model")
        >>> loaded = AutoModel.from_pretrained("./saved_model")  # Auto-detects as BertModel
    """

    registry: ClassVar[Registry]
    config_class: ClassVar[Type[PretrainedConfig]]
    type_key: ClassVar[str] = "model_type"
    config_filename: ClassVar[str] = "config.json"

    def __init__(self) -> None:
        """AutoRegistry should not be instantiated."""
        raise TypeError(
            f"{self.__class__.__name__} is designed to be used as a static class and should not be instantiated."
        )

    @classmethod
    def register_module(cls, model_type: str):
        """
        Decorator to register a model class.

        For registering external classes or bulk registration of multiple models,
        use direct .registry access (e.g., AutoModel.registry["key"] = value or
        AutoModel.registry.update({...})) instead.

        Args:
            model_type: Model type identifier (must match the hardcoded model_type
                       in the model's config class).

        Example:
            >>> # Config with hardcoded model_type
            >>> class BertConfig(PretrainedConfig):
            ...     model_type: str = "bert"

            >>> # Base model class
            >>> class BaseModel(PretrainedMixin):
            ...     config_class = PretrainedConfig

            >>> @AutoModel.register_module("bert")
            ... class BertModel(BaseModel):
            ...     config_class = BertConfig
        """

        def decorator(model_class: Type[PretrainedMixin]) -> Type[PretrainedMixin]:
            cls.registry[model_type] = model_class
            return model_class

        return decorator

    @classmethod
    def from_pretrained(cls, pretrained_path: PathLike, **kwargs: Any) -> PretrainedMixin:
        """Load a model by auto-detecting type from config."""
        config_file = Path(pretrained_path) / cls.config_filename
        config = cls.config_class.model_validate_json(config_file.read_text())
        model_type = getattr(config, cls.type_key)
        model_class = cls.registry[model_type]
        return model_class.from_pretrained(pretrained_path, **kwargs)
