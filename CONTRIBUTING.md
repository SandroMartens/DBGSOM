# Contributing to DBGSOM

## Setup

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/SandroMartens/DBGSOM.git
cd DBGSOM
uv sync --group dev
```

## Running Tests

```bash
uv run pytest -m "not slow"        # fast tests only
uv run pytest                      # all tests (slow regression included)
```

## Linting

```bash
uv run ruff format .
```

## Opening Issues

Open an issue before submitting a PR when:

- the change affects the public API or algorithm behavior
- you are unsure whether the change fits the project scope

Bug fixes and documentation improvements can go straight to a PR.

## Pull Requests

- Target the `main` branch
- Keep PRs focused — one concern per PR
- Add or update tests for any changed behavior
- Update docstrings if parameters or return values change

## Reporting Security Issues

Do **not** open a public issue. Email sandro.martens@googlemail.com directly.
