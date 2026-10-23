
FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /bin/

ENV UV_NO_DEV=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

RUN uv sync --locked --no-dev --no-editable

RUN mkdir -p /app/documents /app/data && \
    chown -R 65534:65534 /app

USER 65534:65534

ENTRYPOINT ["ai-knowledge-mcp-server"]
