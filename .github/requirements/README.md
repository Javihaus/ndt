# Pinned CI requirements

Every workflow installs its dependencies from these files with
`pip install --require-hashes`, then the package itself with
`pip install --no-deps -e .`. Dependabot updates them weekly. To regenerate
by hand (from this folder):

```bash
uv pip compile ../../pyproject.toml --extra dev --universal --python-version 3.10 --generate-hashes -o dev.txt
uv pip compile ../../pyproject.toml --extra docs --python-platform linux --python-version 3.12 --generate-hashes -o docs.txt
uv pip compile build.in --python-platform linux --python-version 3.12 --generate-hashes -o build.txt
uv pip compile ../../pyproject.toml fuzz-extra.in --python-platform linux --python-version 3.12 --generate-hashes -o fuzz.txt
```
