"""Tests for example files."""

import subprocess
import sys
from pathlib import Path


def test_plugin_system_example():
    """Test that plugin_system.py runs without errors."""
    example_file = Path(__file__).parent.parent / "examples" / "plugin_system.py"
    result = subprocess.run([sys.executable, str(example_file)], capture_output=True, text=True, timeout=10)

    assert result.returncode == 0, f"Example failed with stderr: {result.stderr}"
    assert "Available plugins:" in result.stdout
    assert "hello world" in result.stdout
    assert "HELLO WORLD" in result.stdout
    assert "DLROW OLLEH" in result.stdout


def test_pretrained_example():
    """Test that pretrained.py runs without errors."""
    example_file = Path(__file__).parent.parent / "examples" / "pretrained.py"
    result = subprocess.run([sys.executable, str(example_file)], capture_output=True, text=True, timeout=10)

    assert result.returncode == 0, f"Example failed with stderr: {result.stderr}"
    assert "Example 1: Basic Model" in result.stdout
    assert "Example 2: Tokenizer" in result.stdout
    assert "BertModel" in result.stdout
    assert "WordPieceTokenizer" in result.stdout
    assert "All tests passed!" in result.stdout
