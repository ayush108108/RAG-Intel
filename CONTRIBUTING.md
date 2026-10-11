# Contributing to RAG-Intel

Thanks for helping make retrieval infrastructure easier to inspect and improve.

---

## Getting Started

We use [`uv`](https://github.com/astral-sh/uv) to manage Python dependencies and virtual environments.

### 1. Requirements
- Python `>= 3.12`
- `uv` installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Docker (optional, if you're working on the Docker build)

### 2. Setting Up Your Local Environment

```bash
# Clone the repository
git clone https://github.com/ayush108108/RAG-Intel.git
cd RAG-Intel

# Fast offline contributor setup
uv sync --extra dev --frozen
```

---

## Local Development Workflow

### Running Tests
Make sure unit tests pass before submitting changes:
```bash
uv run pytest -q
```

### Code Formatting & Checks
We use `ruff` to keep code clean and formatted:
```bash
# Check for lint errors
uv run ruff check src tests examples

# Apply auto-formatting
uv run ruff format --check src tests examples
```

### Testing the CLI
```bash
uv run rag-intel --help
uv run rag-intel evaluate --relevant a,b --ranked b,c,a --k 3
```

### Testing the Docker Container
```bash
docker compose run --rm rag-intel
```

---

## Submitting Pull Requests

1. Create a new branch: `git checkout -b my-feature-name`
2. Keep changes focused and easy to review.
3. Add focused tests for behavior changes.
4. Keep ranking policy explicit; do not add silent scoring defaults.
5. Make sure tests, formatting, lint, and `uv build` pass.
6. Push your branch and submit a Pull Request to `main`.

If you run into questions or issues while setting up, feel free to open a GitHub issue or discussion!
