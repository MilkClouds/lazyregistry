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

    config_class: ClassVar[Type[BaseModel]]
    config_filename: ClassVar[str] = "config.json"

    def __init__(self, config: ConfigT):
        """Initialize with a configuration object."""
        self.config = config

    def save_pretrained(self, save_directory: PathLike) -> None:
        """
        Save the model configuration to a directory.

        Args:
            save_directory: Directory path where the config will be saved.
        """
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        config_file = save_path / self.config_filename
        config_file.write_text(self.config.model_dump_json(indent=2))

    @classmethod
    def from_pretrained(cls, pretrained_path: PathLike, **kwargs: Any) -> "PretrainedMixin":
        """
        Load a model from a saved configuration.

        Args:
            pretrained_path: Directory path containing the saved config.
            **kwargs: Additional arguments passed to the model constructor.

        Returns:
            Instance of the model class.
        """
        config_file = Path(pretrained_path) / cls.config_filename
        config = cls.config_class.model_validate_json(config_file.read_text())
        return cls(config, **kwargs)


class AutoRegistry(Generic[ModelT]):
    """
    Auto-loader registry for pretrained models.

    Provides a decorator-based registration system and automatic model loading
    based on configuration type detection.

    Example:
        >>> from pydantic import BaseModel
        >>> from lazyregistry import NAMESPACE

        >>> class ModelConfig(BaseModel):
        ...     model_type: str
        ...     hidden_size: int = 768

        >>> MODEL_REGISTRY = NAMESPACE["models"]

        >>> class AutoModel(AutoRegistry):
        ...     registry = MODEL_REGISTRY
        ...     config_class = ModelConfig
        ...     type_key = "model_type"

        >>> @AutoModel.register("bert")
        ... class BertModel(PretrainedMixin[ModelConfig]):
        ...     config_class = ModelConfig

        >>> @AutoModel.register("gpt2")
        ... class GPT2Model(PretrainedMixin[ModelConfig]):
        ...     config_class = ModelConfig

        >>> # Auto-detect and load based on config.model_type
        >>> model = AutoModel.from_pretrained("./saved_model")
    """

    registry: ClassVar[Registry]
    config_class: ClassVar[Type[BaseModel]]
    type_key: ClassVar[str] = "model_type"
    config_filename: ClassVar[str] = "config.json"

    def __init__(self) -> None:
        """AutoRegistry should not be instantiated."""
        raise TypeError(
            f"{self.__class__.__name__} is designed to be used as a static class and should not be instantiated."
        )

    @classmethod
    def register(cls, model_type: str):
        """
        Decorator to register a model class.

        Args:
            model_type: Unique identifier for the model type.

        Returns:
            Decorator function that registers the class.

        Example:
            >>> @AutoModel.register("bert")
            ... class BertModel(PretrainedMixin):
            ...     pass
        """

        def decorator(model_class: Type[ModelT]) -> Type[ModelT]:
            cls.registry.register(model_type, model_class, is_instance=True)
            return model_class

        return decorator

    @classmethod
    def from_pretrained(cls, pretrained_path: PathLike, **kwargs: Any) -> ModelT:
        """
        Load a model from a saved configuration.

        Automatically detects the model type from the configuration file
        and loads the appropriate registered model class.

        Args:
            pretrained_path: Directory path containing the saved config.
            **kwargs: Additional arguments passed to the model constructor.

        Returns:
            Instance of the registered model class.

        Raises:
            KeyError: If the model type is not registered.
            FileNotFoundError: If the config file doesn't exist.
        """
        config_file = Path(pretrained_path) / cls.config_filename
        config = cls.config_class.model_validate_json(config_file.read_text())

        # Get model type from config
        model_type = getattr(config, cls.type_key)

        # Load the registered model class
        model_class = cls.registry[model_type]

        # Use the model class's from_pretrained method
        return model_class.from_pretrained(pretrained_path, **kwargs)
