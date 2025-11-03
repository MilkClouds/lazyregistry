"""
Template for pretrained model pattern with registry support.

This is a TEMPLATE module. Replace "Something" with your actual class name:
- SomethingConfig -> YourConfig
- SomethingMixin -> YourMixin
- AutoSomething -> AutoYour
- SOMETHING_REGISTRY -> YOUR_REGISTRY
- something_config.json -> your_config.json
"""

import os
from pathlib import Path
from typing import ClassVar, Type, Union

from pydantic import BaseModel

from sthrgs.registry import Namespace, Registry

PathLike = Union[str, os.PathLike]

# TODO: Rename "something" to your actual registry name
SOMETHING_REGISTRY: Registry[str, Type["SomethingMixin"]] = Namespace()["something"]
CONFIG_FILE = "something_config.json"


class SomethingConfig(BaseModel):
    """Configuration for Something implementations.

    Example:
        config = SomethingConfig(name="my_something")
    """

    name: str


class SomethingMixin:
    """Base class for Something implementations with save/load functionality.

    Example:
        class MyThing(SomethingMixin):
            config_cls = SomethingConfig

        config = SomethingConfig(name="my_thing")
        obj = MyThing(config)
        obj.save_pretrained("./path")
        loaded = MyThing.from_pretrained("./path")
    """

    config_cls: ClassVar[Type[SomethingConfig]]

    def __init__(self, config: SomethingConfig):
        self.config = config

    def save_pretrained(self, save_directory: PathLike) -> None:
        """Save the configuration to a directory."""
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        (save_path / CONFIG_FILE).write_text(self.config.model_dump_json())

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: PathLike, **kwargs):
        """Load a pretrained model from a directory."""
        config_file = Path(pretrained_model_name_or_path) / CONFIG_FILE
        config = cls.config_cls.model_validate_json(config_file.read_text())
        return cls(config, **kwargs)


class AutoSomething:
    """Auto-loader for registered Something implementations

    Example:
        @AutoSomething.register("my_thing")
        class MyThing(SomethingMixin):
            config_cls = SomethingConfig

        obj = AutoSomething.from_pretrained("./path")
    """

    def __init__(self) -> None:
        raise NotImplementedError("AutoSomething is a static class and should not be instantiated")

    @classmethod
    def register(cls, name: str):
        """Register a class as a Something implementation."""

        def decorator(something_cls: Type[SomethingMixin]) -> Type[SomethingMixin]:
            SOMETHING_REGISTRY[name] = something_cls
            return something_cls

        return decorator

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: PathLike, **kwargs):
        """Load a Something implementation from pretrained."""
        config_file = Path(pretrained_model_name_or_path) / CONFIG_FILE
        config = SomethingConfig.model_validate_json(config_file.read_text())
        subclass = SOMETHING_REGISTRY[config.name]
        return subclass.from_pretrained(pretrained_model_name_or_path, **kwargs)
