# Claude Code Guidelines for huggingface-fine-tuning

## Project Overview

This is an educational repository for fine-tuning transformer models with Hugging Face. It contains hands-on labs and examples covering datasets, tokenization, training, and model publishing.

## Package Manager

**ONLY use `uv` for all Python operations:**

```bash
# Install dependencies
uv sync --all-extras

# Run Python scripts
uv run python examples/loading/load_csv.py

# Run tests
uv run pytest tests/ -v

# Run linting
uv run ruff check examples/
```

**DO NOT use:**
- `pip install`
- `python -m venv`
- `conda`

## Project Structure

```
examples/           # Code examples organized by topic
├── loading/        # Dataset loading
├── transform/      # Data transformations
├── tokenize/       # Tokenization
├── augment/        # Data augmentation
├── imbalance/      # Imbalanced data handling
├── models/         # Pre-trained models
├── training/       # Trainer API
├── custom/         # Custom configurations
├── callback/       # Training callbacks
├── inferencing/    # Inference
└── publishing/     # Hub publishing

labs/               # Lab instructions (Markdown)
tests/              # Test suite
```

## Code Style

- Use `ruff` for linting and formatting
- Follow PEP 8 conventions
- Add docstrings to functions and classes
- Use type hints where appropriate

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=examples
```

## Common Tasks

### Add a New Example

1. Create file in appropriate `examples/` subdirectory
2. Add clear docstring explaining the example
3. Ensure it runs standalone: `uv run python examples/category/new_example.py`

### Update Dependencies

1. Edit `pyproject.toml`
2. Run `uv sync --all-extras`
3. Test that examples still work

## Quality Standards

- PMAT repo-score: Target A+ (90+)
- PMAT demo-score: Target A+ (9.0+)
- All examples must have valid Python syntax
- Pre-commit hooks must pass
