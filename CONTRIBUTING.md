# Contributing to Neural Dimensionality Tracker

Thank you for your interest in contributing to NDT! This document provides guidelines and instructions for contributing.

## Development Setup

1. Fork and clone the repository:
```bash
git clone https://github.com/yourusername/ndt.git
cd ndt
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install in development mode:
```bash
pip install -e ".[dev]"
```

## Code Style

We follow strict code quality standards:

- **Formatting**: Black (line length 100)
- **Import sorting**: isort
- **Linting**: flake8 (max complexity 10)
- **Type hints**: All functions must have type annotations
- **Docstrings**: Google-style docstrings for all public functions

Run formatters:
```bash
black src/ndt tests examples
isort src/ndt tests examples
```

Check linting:
```bash
flake8 src/ndt tests --max-line-length=100
```

## Testing

All contributions must include tests. We aim for >90% test coverage.

Run tests:
```bash
pytest tests/ -v --cov=ndt --cov-report=html
```

View coverage:
```bash
open htmlcov/index.html  # On macOS
# Or navigate to htmlcov/index.html in your browser
```

### Writing Tests

- Place tests in `tests/` matching the module structure
- Use pytest fixtures from `tests/conftest.py`
- Test both normal cases and edge cases
- Include docstrings explaining what each test does

Example:
```python
def test_stable_rank_identity_matrix(identity_matrix):
    """Identity matrix should have stable rank equal to its dimension."""
    sr = stable_rank(identity_matrix)
    assert sr == pytest.approx(50.0, rel=0.01)
```

## Pull Request Process

1. Create a feature branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes with tests

3. Run all checks:
```bash
black src/ndt tests
isort src/ndt tests
flake8 src/ndt tests
pytest tests/ -v --cov=ndt
```

4. Commit with descriptive messages:
```bash
git commit -m "Add feature: description"
```

5. Push and create a Pull Request:
```bash
git push origin feature/your-feature-name
```

6. Fill out the PR template completely

## Types of Contributions

### Bug Fixes
- Include a test that fails before the fix
- Reference the issue number in the PR

### New Features
- Discuss in an issue first for major features
- Include comprehensive tests
- Update documentation and examples
- Add to README if user-facing

### Documentation
- Fix typos, clarify explanations
- Add examples
- Improve docstrings

### Performance Improvements
- Include benchmarks showing improvement
- Ensure no functionality changes
- Add tests if needed

## Reporting Bugs

Use the bug report template and include:
- Minimal code to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Full error traceback

## Suggesting Features

Use the feature request template and include:
- Use case description
- Proposed API (if applicable)
- Alternatives considered

## Code Review

All submissions require review. We review for:
- Correctness
- Test coverage
- Code style adherence
- Documentation quality
- Performance implications

## Release Process

Maintainers handle releases:
1. Update version in `pyproject.toml` and `__version__.py`
2. Update CHANGELOG.md
3. Create git tag: `git tag v0.1.0`
4. Push tag: `git push --tags`
5. GitHub Actions automatically publishes to PyPI

## Questions?

- Open an issue for questions
- Check existing issues and PRs first
- Email maintainers for sensitive issues

Thank you for contributing to NDT!
