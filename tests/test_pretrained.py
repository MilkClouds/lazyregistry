"""Tests for pretrained model functionality."""

import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import pytest
from pydantic import BaseModel

from lazyregistry import NAMESPACE
from lazyregistry.pretrained import AutoRegistry, PathLike, PretrainedMixin


class SimpleConfig(BaseModel):
    """Simple configuration for testing."""

    model_type: str
    value: int = 42


class SimpleModel(PretrainedMixin[SimpleConfig]):
    """Simple model for pretrained functionality testing."""

    config_class = SimpleConfig


class TestPretrainedMixin:
    """Test PretrainedMixin class."""

    def test_init_with_config(self):
        """Test initialization with config."""
        config = SimpleConfig(model_type="test", value=100)
        model = SimpleModel(config)
        assert model.config == config
        assert model.config.value == 100

    def test_save_pretrained(self):
        """Test saving pretrained model."""
        config = SimpleConfig(model_type="test", value=123)
        model = SimpleModel(config)

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
        config = SimpleConfig(model_type="test", value=456)
        model = SimpleModel(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded = SimpleModel.from_pretrained(tmpdir)

            assert loaded.config.model_type == "test"
            assert loaded.config.value == 456

    def test_save_load_roundtrip(self):
        """Test save and load roundtrip."""
        config = SimpleConfig(model_type="test", value=789)
        model = SimpleModel(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded = SimpleModel.from_pretrained(tmpdir)

            assert model.config == loaded.config


class CustomConfig(BaseModel):
    """Config with custom state."""

    model_type: str
    vocab_size: int = 100


class CustomModel(PretrainedMixin[CustomConfig]):
    """Model with custom state beyond config."""

    config_class = CustomConfig

    def __init__(self, config: CustomConfig, vocab: Optional[Dict[str, int]] = None):
        super().__init__(config)
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
    def from_pretrained(cls, pretrained_path: PathLike, **kwargs: Any) -> "CustomModel":
        """Load config and vocabulary."""
        config_file = Path(pretrained_path) / cls.config_filename
        config = cls.config_class.model_validate_json(config_file.read_text())

        # Load vocabulary
        vocab_file = Path(pretrained_path) / "vocab.txt"
        vocab = {}
        if vocab_file.exists():
            words = vocab_file.read_text().strip().split("\n")
            vocab = {word: idx for idx, word in enumerate(words)}

        return cls(config, vocab=vocab, **kwargs)


class TestCustomPretrained:
    """Test custom pretrained with additional state."""

    def test_save_custom_state(self):
        """Test saving custom state."""
        config = CustomConfig(model_type="custom", vocab_size=5)
        vocab = {"<unk>": 0, "hello": 1, "world": 2}
        model = CustomModel(config, vocab=vocab)

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
        model = CustomModel(config, vocab=vocab)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded = CustomModel.from_pretrained(tmpdir)

            assert loaded.vocab == vocab


class AutoTestModel(AutoRegistry):
    """Auto-loader for test models."""

    registry = NAMESPACE["test_models"]
    config_class = SimpleConfig
    type_key = "model_type"


@AutoTestModel.register("bert")
class BertTestModel(PretrainedMixin[SimpleConfig]):
    """BERT test model."""

    config_class = SimpleConfig


@AutoTestModel.register("gpt")
class GPTTestModel(PretrainedMixin[SimpleConfig]):
    """GPT test model."""

    config_class = SimpleConfig


class TestAutoRegistry:
    """Test AutoRegistry class."""

    def test_register_decorator(self):
        """Test register decorator."""
        assert "bert" in NAMESPACE["test_models"]
        assert "gpt" in NAMESPACE["test_models"]

    def test_from_pretrained_auto_detect(self):
        """Test auto-detection from config."""
        config = SimpleConfig(model_type="bert", value=999)
        model = BertTestModel(config)

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
        bert_config = SimpleConfig(model_type="bert", value=111)
        bert_model = BertTestModel(bert_config)

        # Save GPT model
        gpt_config = SimpleConfig(model_type="gpt", value=222)
        gpt_model = GPTTestModel(gpt_config)

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
        config = SimpleConfig(model_type="unknown", value=0)
        model = SimpleModel(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)

            # Should raise KeyError for unknown type
            with pytest.raises(KeyError):
                AutoTestModel.from_pretrained(tmpdir)

    def test_cannot_instantiate(self):
        """Test that AutoRegistry cannot be instantiated."""
        with pytest.raises(TypeError, match="should not be instantiated"):
            AutoTestModel()
