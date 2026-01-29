# Contributing to Fine-tuning with Hugging Face

Thank you for your interest in contributing to this project!

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/hf-finetuning.git
   cd hf-finetuning
   ```

3. Install dependencies:
   ```bash
   uv sync --all-extras
   ```

4. Install pre-commit hooks:
   ```bash
   uv run pre-commit install
   ```

## Development Workflow

### Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=examples --cov-report=term-missing
```

### Code Style

We use `ruff` for linting and formatting:

```bash
# Check code style
uv run ruff check examples/

# Format code
uv run ruff format examples/
```

### Adding Examples

When adding new examples:

1. Create a new directory under `examples/` if needed
2. Add clear docstrings explaining the example
3. Include any necessary data files
4. Update the README if adding a new category

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Ensure all tests pass
4. Update documentation as needed
5. Submit a pull request

## Code of Conduct

Please be respectful and constructive in all interactions.

## Questions?

Open an issue for any questions about contributing.
