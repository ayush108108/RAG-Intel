# Use official Python 3.12 slim image
FROM python:3.12-slim-bookworm

# Install uv from official Astral image
COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /uvx /bin/

# Set working directory & environment variables
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# Install system runtime dependencies (OpenMP for FAISS/PyTorch, Tesseract for OCR, Poppler for PDF parsing)
RUN (apt-get update -y || apt-get update -y --fix-missing) && \
    apt-get install -y --no-install-recommends \
    libgomp1 \
    tesseract-ocr \
    poppler-utils \
    git \
    || true && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml and uv.lock first for optimal Docker layer caching
COPY pyproject.toml uv.lock /app/

# Install dependencies into virtualenv using uv sync with BuildKit caching
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --all-extras --frozen --no-dev --no-install-project

# Copy source code and project files
COPY src /app/src
COPY config /app/config
COPY examples /app/examples
COPY docs /app/docs
COPY sample_data /app/sample_data
COPY README.md LICENSE /app/

# Sync local package into virtualenv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --all-extras --frozen --no-dev

# Ensure runtime data persistence directory exists
RUN mkdir -p /app/data

# Discoverable default; mount data and pass a command/config for real use.
ENTRYPOINT ["rag-intel"]
CMD ["--help"]
