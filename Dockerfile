# syntax=docker/dockerfile:1

# docker build -t dinora:v0.3.0 .
# docker run --rm -i --network none --cpus 1 --memory 512m dinora:v0.3.0
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV VIRTUAL_ENV=/app/.venv UV_NO_CACHE=1
WORKDIR /src

# CPU runtime versions used by the original v0.3.0 standalone build.
RUN uv venv "$VIRTUAL_ENV" \
 && uv pip install "chess==1.11.2" "numpy==2.2.5" "onnxruntime==1.22.0"

COPY pyproject.toml README.md ./
COPY src/dinora ./src/dinora
RUN uv pip install --no-deps .

FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv

# Original release weights, verified at build time. No network needed at play time.
ADD --checksum=sha256:efc7d3978ac3ecefd113f47bef62170e285ec64d2cc477f2712bff3219fe7c57 \
    https://github.com/Saegl/dinora/releases/download/v0.3.0/default.onnx /app/models/default.onnx

ENTRYPOINT [".venv/bin/python", "-m", "dinora", "--searcher", "mcts", \
            "--model", "onnx", "--device", "cpu", "--limit-threads"]
