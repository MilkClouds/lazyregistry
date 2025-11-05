"""
HuggingFace-style pretrained model pattern using lazyregistry.

Demonstrates:
1. Basic save_pretrained/from_pretrained with config only
2. Custom state (vocabulary) beyond configuration
3. AutoRegistry for automatic model type detection
4. Both decorator and direct .registry registration methods
"""

from pathlib import Path
from typing import Any, Dict, Optional

from lazyregistry import NAMESPACE
from lazyregistry.pretrained import AutoRegistry, PathLike, PretrainedConfig, PretrainedMixin

# ============================================================================
# Example 1: Basic pretrained models (config only)
# ============================================================================


class BertConfig(PretrainedConfig):
    """BERT-specific configuration."""

    model_type: str = "bert"
    hidden_size: int = 768
    num_layers: int = 12


class GPT2Config(PretrainedConfig):
    """GPT-2-specific configuration."""

    model_type: str = "gpt2"
    hidden_size: int = 768
    num_layers: int = 12


class BaseModel(PretrainedMixin):
    """Base model class for all transformer models."""

    config_class = PretrainedConfig


class AutoModel(AutoRegistry):
    """Auto-loader for registered models."""

    registry = NAMESPACE["models"]
    config_class = PretrainedConfig
    type_key = "model_type"


@AutoModel.register_module("bert")
class BertModel(BaseModel):
    """BERT model - saves/loads config only."""

    config_class = BertConfig


@AutoModel.register_module("gpt2")
class GPT2Model(BaseModel):
    """GPT-2 model - saves/loads config only."""

    config_class = GPT2Config


# ============================================================================
# Example 2: Custom pretrained with additional state (vocabulary)
# ============================================================================


class WordPieceConfig(PretrainedConfig):
    """WordPiece tokenizer configuration."""

    model_type: str = "wordpiece"
    max_length: int = 512
    lowercase: bool = True


class BPEConfig(PretrainedConfig):
    """BPE tokenizer configuration."""

    model_type: str = "bpe"
    max_length: int = 512
    lowercase: bool = True


class BaseTokenizer(PretrainedMixin):
    """Base tokenizer with vocabulary state."""

    config_class = PretrainedConfig

    def __init__(self, *args, vocab: Optional[Dict[str, int]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab = vocab or {"<unk>": 0, "<pad>": 1}

    def save_pretrained(self, save_directory: PathLike) -> None:
        """Save config AND vocabulary."""
        super().save_pretrained(save_directory)

        # Save vocabulary sorted by index
        save_path = Path(save_directory)
        vocab_file = save_path / "vocab.txt"
        sorted_vocab = sorted(self.vocab.items(), key=lambda x: x[1])
        vocab_file.write_text("\n".join(word for word, _ in sorted_vocab))

    @classmethod
    def from_pretrained(cls, pretrained_path: PathLike, **kwargs: Any):
        """Load config AND vocabulary."""
        config_file = Path(pretrained_path) / cls.config_filename
        config = cls.config_class.model_validate_json(config_file.read_text())

        # Load vocabulary
        vocab_file = Path(pretrained_path) / "vocab.txt"
        vocab = {}
        if vocab_file.exists():
            words = vocab_file.read_text().strip().split("\n")
            vocab = {word: idx for idx, word in enumerate(words)}

        return cls(config=config, vocab=vocab, **kwargs)

    def encode(self, text: str) -> list[int]:
        """Convert text to token IDs."""
        if self.config.lowercase:
            text = text.lower()
        words = text.split()[: self.config.max_length]
        return [self.vocab.get(word, 0) for word in words]


class AutoTokenizer(AutoRegistry):
    """Auto-loader for tokenizers."""

    registry = NAMESPACE["tokenizers"]
    config_class = PretrainedConfig
    type_key = "model_type"


@AutoTokenizer.register_module("wordpiece")
class WordPieceTokenizer(BaseTokenizer):
    """WordPiece tokenizer."""

    config_class = WordPieceConfig


# Example: Direct registration without decorator
class BPETokenizer(BaseTokenizer):
    """BPE tokenizer."""

    config_class = BPEConfig


# Register using dict-style assignment (alternative to decorator)
AutoTokenizer.registry["bpe"] = BPETokenizer


# ============================================================================
# Demo
# ============================================================================

if __name__ == "__main__":
    import tempfile

    print("Example 1: Basic Model (config only)")

    # Create and save model with model-specific config
    config = BertConfig(hidden_size=1024, num_layers=24)
    model = BertModel(config=config)

    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_pretrained(tmpdir)
        print(f"Saved to {tmpdir}")

        # Auto-detect and load
        loaded = AutoModel.from_pretrained(tmpdir)
        print(f"Loaded: {type(loaded).__name__}")
        print(f"Config: {loaded.config}")

    print("\nExample 2: Tokenizer (config + vocabulary)")

    # Create tokenizer with custom vocabulary using model-specific config
    config = WordPieceConfig(max_length=128)
    vocab = {"<unk>": 0, "<pad>": 1, "hello": 2, "world": 3, "python": 4}
    tokenizer = WordPieceTokenizer(config=config, vocab=vocab)

    text = "Hello World Python"
    tokens = tokenizer.encode(text)
    tokenizer.config
    print(f"Original: {text}")
    print(f"Tokens: {tokens}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tokenizer.save_pretrained(tmpdir)
        print(f"Saved to {tmpdir}")

        # Auto-detect and load
        loaded = AutoTokenizer.from_pretrained(tmpdir)
        print(f"Loaded: {type(loaded).__name__}")
        print(f"Vocab size: {len(loaded.vocab)}")

        # Verify
        tokens_loaded = loaded.encode(text)
        assert tokens == tokens_loaded
        print(f"Tokens match: {tokens_loaded}")
        print("\nAll tests passed!")
