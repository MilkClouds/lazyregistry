"""
HuggingFace-style pretrained model pattern using lazyregistry.

Demonstrates:
1. Basic save_pretrained/from_pretrained with config only
2. Custom state (vocabulary) beyond configuration
3. AutoRegistry for automatic model type detection
"""

from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel

from lazyregistry import NAMESPACE
from lazyregistry.pretrained import AutoRegistry, PathLike, PretrainedMixin

# ============================================================================
# Example 1: Basic pretrained models (config only)
# ============================================================================


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


@AutoModel.register("bert")
class BertModel(PretrainedMixin[ModelConfig]):
    """BERT model - saves/loads config only."""

    config_class = ModelConfig


@AutoModel.register("gpt2")
class GPT2Model(PretrainedMixin[ModelConfig]):
    """GPT-2 model - saves/loads config only."""

    config_class = ModelConfig


# ============================================================================
# Example 2: Custom pretrained with additional state (vocabulary)
# ============================================================================


class TokenizerConfig(BaseModel):
    """Tokenizer configuration."""

    tokenizer_type: str
    max_length: int = 512
    lowercase: bool = True


class Tokenizer(PretrainedMixin[TokenizerConfig]):
    """Tokenizer with vocabulary state."""

    config_class = TokenizerConfig

    def __init__(self, config: TokenizerConfig, vocab: Optional[Dict[str, int]] = None):
        super().__init__(config)
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
    def from_pretrained(cls, pretrained_path: PathLike, **kwargs: Any) -> "Tokenizer":
        """Load config AND vocabulary."""
        config_file = Path(pretrained_path) / cls.config_filename
        config = cls.config_class.model_validate_json(config_file.read_text())

        # Load vocabulary
        vocab_file = Path(pretrained_path) / "vocab.txt"
        vocab = {}
        if vocab_file.exists():
            words = vocab_file.read_text().strip().split("\n")
            vocab = {word: idx for idx, word in enumerate(words)}

        return cls(config, vocab=vocab, **kwargs)

    def encode(self, text: str) -> list[int]:
        """Convert text to token IDs."""
        if self.config.lowercase:
            text = text.lower()
        words = text.split()[: self.config.max_length]
        return [self.vocab.get(word, 0) for word in words]


class AutoTokenizer(AutoRegistry):
    """Auto-loader for tokenizers."""

    registry = NAMESPACE["tokenizers"]
    config_class = TokenizerConfig
    type_key = "tokenizer_type"


@AutoTokenizer.register("wordpiece")
class WordPieceTokenizer(Tokenizer):
    """WordPiece tokenizer."""

    pass


# ============================================================================
# Demo
# ============================================================================

if __name__ == "__main__":
    import tempfile

    print("Example 1: Basic Model (config only)")

    # Create and save model
    config = ModelConfig(model_type="bert", hidden_size=1024, num_layers=24)
    model = BertModel(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_pretrained(tmpdir)
        print(f"Saved to {tmpdir}")

        # Auto-detect and load
        loaded = AutoModel.from_pretrained(tmpdir)
        print(f"Loaded: {type(loaded).__name__}")
        print(f"Config: {loaded.config}")

    print("\nExample 2: Tokenizer (config + vocabulary)")

    # Create tokenizer with custom vocabulary
    config = TokenizerConfig(tokenizer_type="wordpiece", max_length=128)
    vocab = {"<unk>": 0, "<pad>": 1, "hello": 2, "world": 3, "python": 4}
    tokenizer = WordPieceTokenizer(config, vocab=vocab)

    text = "Hello World Python"
    tokens = tokenizer.encode(text)
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
