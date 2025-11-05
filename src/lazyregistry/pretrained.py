"""
Pretrained model pattern support for lazyregistry.

Provides base classes and utilities for implementing HuggingFace-style
save_pretrained/from_pretrained patterns with automatic model registration.
"""

import os
from pathlib import Path
from typing import Any, ClassVar, Generic, Type, TypeVar, Union

from pydantic import BaseModel

from lazyregistry.registry import Registry

__all__ = ["PathLike", "PretrainedMixin", "AutoRegistry"]

PathLike = Union[str, os.PathLike]
ConfigT = TypeVar("ConfigT", bound=BaseModel)
ModelT = TypeVar("ModelT", bound="PretrainedMixin")
AutoConfigT = TypeVar("AutoConfigT", bound=BaseModel)


class PretrainedMixin(Generic[ConfigT]):
    """
    Mixin class providing save_pretrained/from_pretrained functionality.

    Classes inheriting from this mixin can be saved to and loaded from disk
    with their configuration automatically serialized/deserialized.

    Example:
        >>> class MyConfig(BaseModel):
        ...     name: str
        ...     hidden_size: int = 768

        >>> class MyModel(PretrainedMixin[MyConfig]):
        ...     config_class = MyConfig
        ...
        ...     def __init__(self, config: MyConfig):
        ...         self.config = config

        >>> config = MyConfig(name="my_model", hidden_size=1024)
        >>> model = MyModel(config)
        >>> model.save_pretrained("./my_model")
        >>> loaded = MyModel.from_pretrained("./my_model")
    """

    config_class: ClassVar[Type[ConfigT]]
    config_filename: ClassVar[str] = "config.json"

    def __init__(self, config: ConfigT):
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


class AutoRegistry(Generic[ModelT, AutoConfigT]):
    """
    Auto-loader registry for pretrained models.

    Provides decorator-based registration and automatic model loading
    based on configuration type detection.

    Registration methods:
        1. Decorator: @AutoModel.register_module("model_type")
        2. Direct: AutoModel.registry["model_type"] = ModelClass
        3. Bulk: AutoModel.registry.update({...})

    Example:
        >>> from pydantic import BaseModel
        >>> from lazyregistry import NAMESPACE

        >>> class ModelConfig(BaseModel):
        ...     model_type: str
        ...     hidden_size: int = 768

        >>> class AutoModel(AutoRegistry):
        ...     registry = NAMESPACE["models"]
        ...     config_class = ModelConfig
        ...     type_key = "model_type"

        >>> # Decorator registration
        >>> @AutoModel.register_module("bert")
        ... class BertModel(PretrainedMixin[ModelConfig]):
        ...     config_class = ModelConfig

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
        >>> model = AutoModel.from_pretrained("./saved_model")
    """

    registry: ClassVar[Registry]
    config_class: ClassVar[Type[AutoConfigT]]
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
            model_type: Model type identifier (must match config's type_key value).

        Example:
            >>> @AutoModel.register_module("bert")
            ... class BertModel(PretrainedMixin):
            ...     config_class = ModelConfig
        """

        def decorator(model_class: Type[ModelT]) -> Type[ModelT]:
            cls.registry[model_type] = model_class
            return model_class

        return decorator

    @classmethod
    def from_pretrained(cls, pretrained_path: PathLike, **kwargs: Any) -> ModelT:
        """Load a model by auto-detecting type from config."""
        config_file = Path(pretrained_path) / cls.config_filename
        config = cls.config_class.model_validate_json(config_file.read_text())
        model_type = getattr(config, cls.type_key)
        model_class = cls.registry[model_type]
        return model_class.from_pretrained(pretrained_path, **kwargs)
