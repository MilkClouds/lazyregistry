"""
Custom pretrained example with additional state.

Shows how to extend PretrainedMixin to save/load additional state
beyond just configuration (e.g., weights, vocabularies, etc.).
"""

from pathlib import Path
from typing import Any
from pydantic import BaseModel
from lazyregistry import NAMESPACE, AutoRegistry, PretrainedMixin


class ProcessorConfig(BaseModel):
    """Configuration for text processors."""
    processor_type: str
    max_length: int = 512
    lowercase: bool = True


class TextProcessor(PretrainedMixin[ProcessorConfig]):
    """Text processor with vocabulary state."""
    
    config_class = ProcessorConfig
    
    def __init__(self, config: ProcessorConfig, vocab: dict[str, int] | None = None):
        super().__init__(config)
        self.vocab = vocab or {}
    
    def save_pretrained(self, save_directory: str | Path) -> None:
        """Save config and vocabulary."""
        # Save config using parent method
        super().save_pretrained(save_directory)

        # Save additional state (vocabulary) - sorted by index
        save_path = Path(save_directory)
        vocab_file = save_path / "vocab.txt"
        sorted_vocab = sorted(self.vocab.items(), key=lambda x: x[1])
        vocab_file.write_text("\n".join(word for word, _ in sorted_vocab))
    
    @classmethod
    def from_pretrained(cls, pretrained_path: str | Path, **kwargs: Any) -> "TextProcessor":
        """Load config and vocabulary."""
        # Load config
        config_file = Path(pretrained_path) / cls.config_filename
        config = cls.config_class.model_validate_json(config_file.read_text())
        
        # Load vocabulary
        vocab_file = Path(pretrained_path) / "vocab.txt"
        if vocab_file.exists():
            words = vocab_file.read_text().strip().split("\n")
            vocab = {word: idx for idx, word in enumerate(words)}
        else:
            vocab = {}
        
        return cls(config, vocab=vocab, **kwargs)
    
    def process(self, text: str) -> list[int]:
        """Convert text to token IDs."""
        if self.config.lowercase:
            text = text.lower()
        
        words = text.split()[:self.config.max_length]
        return [self.vocab.get(word, 0) for word in words]


class AutoProcessor(AutoRegistry):
    """Auto-loader for text processors."""
    registry = NAMESPACE["processors"]
    config_class = ProcessorConfig
    type_key = "processor_type"


@AutoProcessor.register("simple")
class SimpleProcessor(TextProcessor):
    """Simple word-based processor."""
    pass


@AutoProcessor.register("advanced")
class AdvancedProcessor(TextProcessor):
    """Advanced processor with additional features."""
    
    def process(self, text: str) -> list[int]:
        """Process with additional normalization."""
        # Remove punctuation
        text = "".join(c for c in text if c.isalnum() or c.isspace())
        return super().process(text)


if __name__ == "__main__":
    import tempfile
    
    # Create processor with vocabulary
    config = ProcessorConfig(
        processor_type="simple",
        max_length=128,
        lowercase=True
    )
    vocab = {"<unk>": 0, "hello": 1, "world": 2, "python": 3}
    processor = SimpleProcessor(config, vocab=vocab)
    
    # Test processing
    text = "Hello World Python"
    tokens = processor.process(text)
    print(f"Original: {text}")
    print(f"Tokens: {tokens}")
    
    # Save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        processor.save_pretrained(tmpdir)
        print(f"\nSaved to {tmpdir}")
        
        # Load using AutoProcessor
        loaded = AutoProcessor.from_pretrained(tmpdir)
        print(f"Loaded: {type(loaded).__name__}")
        print(f"Vocab size: {len(loaded.vocab)}")
        
        # Test loaded processor
        tokens_loaded = loaded.process(text)
        print(f"Tokens (loaded): {tokens_loaded}")
        assert tokens == tokens_loaded, "Tokens should match!"
        print("\n✓ Save/load successful!")

