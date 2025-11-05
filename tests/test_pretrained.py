"""Tests for pretrained model functionality."""

import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from lazyregistry import NAMESPACE
from lazyregistry.pretrained import AutoRegistry, PathLike, PretrainedConfig, PretrainedMixin


class SimpleConfig(PretrainedConfig):
    """Simple configuration for testing."""

    model_type: str = "test"
    value: int = 42


class BertConfig(PretrainedConfig):
    """BERT configuration for testing."""

    model_type: str = "bert"
    value: int = 42


class GPTConfig(PretrainedConfig):
    """GPT configuration for testing."""

    model_type: str = "gpt"
    value: int = 42


class UnknownConfig(PretrainedConfig):
    """Unknown configuration for testing."""

    model_type: str = "unknown"
    value: int = 42


class BaseTestModel(PretrainedMixin):
    """Base model for testing."""

    config_class = PretrainedConfig


class SimpleModel(BaseTestModel):
    """Simple model for pretrained functionality testing."""

    config_class = SimpleConfig


class TestPretrainedMixin:
    """Test PretrainedMixin class."""

    def test_init_with_config(self):
        """Test initialization with config."""
        config = SimpleConfig(value=100)
        model = SimpleModel(config=config)
        assert model.config == config
        assert model.config.value == 100

    def test_save_pretrained(self):
        """Test saving pretrained model."""
        config = SimpleConfig(value=123)
        model = SimpleModel(config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)

            # Check config file exists
            config_file = Path(tmpdir) / "config.json"
            assert config_file.exists()

            # Check config content
            saved_config = SimpleConfig.model_validate_json(config_file.read_text())
            assert saved_config.model_type == "test"
            assert saved_config.value == 123

    def test_from_pretrained(self):
        """Test loading pretrained model."""
        config = SimpleConfig(value=456)
        model = SimpleModel(config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded = SimpleModel.from_pretrained(tmpdir)

            assert loaded.config.model_type == "test"
            assert loaded.config.value == 456

    def test_save_load_roundtrip(self):
        """Test save and load roundtrip."""
        config = SimpleConfig(value=789)
        model = SimpleModel(config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded = SimpleModel.from_pretrained(tmpdir)

            assert model.config == loaded.config


class CustomConfig(PretrainedConfig):
    """Config with custom state."""

    model_type: str = "custom"
    vocab_size: int = 100


class CustomModel(PretrainedMixin):
    """Model with custom state beyond config."""

    config_class = CustomConfig

    def __init__(self, *args, vocab: Optional[Dict[str, int]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab = vocab or {}

    def save_pretrained(self, save_directory: PathLike) -> None:
        """Save config and vocabulary."""
        super().save_pretrained(save_directory)

        # Save vocabulary
        save_path = Path(save_directory)
        vocab_file = save_path / "vocab.txt"
        sorted_vocab = sorted(self.vocab.items(), key=lambda x: x[1])
        vocab_file.write_text("\n".join(word for word, _ in sorted_vocab))

    @classmethod
    def from_pretrained(cls, pretrained_path: PathLike, **kwargs: Any):
        """Load config and vocabulary."""
        config_file = Path(pretrained_path) / cls.config_filename
        config = cls.config_class.model_validate_json(config_file.read_text())

        # Load vocabulary
        vocab_file = Path(pretrained_path) / "vocab.txt"
        vocab = {}
        if vocab_file.exists():
            words = vocab_file.read_text().strip().split("\n")
            vocab = {word: idx for idx, word in enumerate(words)}

        return cls(config=config, vocab=vocab, **kwargs)


class TestCustomPretrained:
    """Test custom pretrained with additional state."""

    def test_save_custom_state(self):
        """Test saving custom state."""
        config = CustomConfig(model_type="custom", vocab_size=5)
        vocab = {"<unk>": 0, "hello": 1, "world": 2}
        model = CustomModel(config=config, vocab=vocab)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)

            # Check vocab file exists
            vocab_file = Path(tmpdir) / "vocab.txt"
            assert vocab_file.exists()

            # Check vocab content
            words = vocab_file.read_text().strip().split("\n")
            assert words == ["<unk>", "hello", "world"]

    def test_load_custom_state(self):
        """Test loading custom state."""
        config = CustomConfig(model_type="custom", vocab_size=5)
        vocab = {"<unk>": 0, "hello": 1, "world": 2}
        model = CustomModel(config=config, vocab=vocab)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded = CustomModel.from_pretrained(tmpdir)

            assert loaded.vocab == vocab


class AutoTestModel(AutoRegistry):
    """Auto-loader for test models."""

    registry = NAMESPACE["test_models"]
    config_class = PretrainedConfig
    type_key = "model_type"


@AutoTestModel.register_module("bert")
class BertTestModel(BaseTestModel):
    """BERT test model."""

    config_class = BertConfig


@AutoTestModel.register_module("gpt")
class GPTTestModel(BaseTestModel):
    """GPT test model."""

    config_class = GPTConfig


class TestAutoRegistry:
    """Test AutoRegistry class."""

    def test_register_module(self):
        """Test register_module decorator and dict-style registration."""
        # Decorator registration
        assert "bert" in NAMESPACE["test_models"]
        assert "gpt" in NAMESPACE["test_models"]

        # Dict-style registration
        class T5TestModel(BaseTestModel):
            config_class = SimpleConfig

        AutoTestModel.registry["t5"] = T5TestModel
        assert "t5" in NAMESPACE["test_models"]

    def test_bulk_registration_via_update(self):
        """Test bulk registration via .registry.update()."""

        class RobertaTestModel(BaseTestModel):
            config_class = SimpleConfig

        class AlbertTestModel(BaseTestModel):
            config_class = SimpleConfig

        # Bulk registration
        AutoTestModel.registry.update(
            {
                "roberta": RobertaTestModel,
                "albert": AlbertTestModel,
            }
        )

        assert "roberta" in NAMESPACE["test_models"]
        assert "albert" in NAMESPACE["test_models"]
        assert NAMESPACE["test_models"]["roberta"] is RobertaTestModel
        assert NAMESPACE["test_models"]["albert"] is AlbertTestModel

    def test_from_pretrained_auto_detect(self):
        """Test auto-detection from config."""
        config = BertConfig(value=999)
        model = BertTestModel(config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)

            # Load using AutoRegistry (should auto-detect type)
            loaded = AutoTestModel.from_pretrained(tmpdir)

            assert isinstance(loaded, BertTestModel)
            assert loaded.config.model_type == "bert"
            assert loaded.config.value == 999

    def test_from_pretrained_different_types(self):
        """Test loading different model types."""
        # Save BERT model
        bert_config = BertConfig(value=111)
        bert_model = BertTestModel(config=bert_config)

        # Save GPT model
        gpt_config = GPTConfig(value=222)
        gpt_model = GPTTestModel(config=gpt_config)

        with tempfile.TemporaryDirectory() as bert_dir:
            with tempfile.TemporaryDirectory() as gpt_dir:
                bert_model.save_pretrained(bert_dir)
                gpt_model.save_pretrained(gpt_dir)

                # Load both
                loaded_bert = AutoTestModel.from_pretrained(bert_dir)
                loaded_gpt = AutoTestModel.from_pretrained(gpt_dir)

                assert isinstance(loaded_bert, BertTestModel)
                assert isinstance(loaded_gpt, GPTTestModel)
                assert loaded_bert.config.value == 111
                assert loaded_gpt.config.value == 222

    def test_unknown_model_type(self):
        """Test error for unknown model type."""

        # Create a model with unknown type config
        class UnknownModel(PretrainedMixin):
            config_class = UnknownConfig

        config = UnknownConfig(value=0)
        model = UnknownModel(config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)

            # Should raise KeyError for unknown type
            with pytest.raises(KeyError):
                AutoTestModel.from_pretrained(tmpdir)

    def test_missing_type_key(self):
        """Test error when config doesn't have the required type key."""

        # Create a config without the type key
        class NoTypeKeyConfig(PretrainedConfig):
            value: int = 42

        class NoTypeKeyModel(PretrainedMixin):
            config_class = NoTypeKeyConfig

        config = NoTypeKeyConfig(value=99)
        model = NoTypeKeyModel(config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)

            # Should raise ValueError for missing type key
            with pytest.raises(ValueError, match="does not contain required type key"):
                AutoTestModel.from_pretrained(tmpdir)

    def test_cannot_instantiate(self):
        """Test that AutoRegistry cannot be instantiated."""
        with pytest.raises(TypeError, match="should not be instantiated"):
            AutoTestModel()
