# syntax=docker/dockerfile:1

# Dinora as a machineplay engine image.
#
#   docker build -t dinora .
#   docker run --rm -i dinora     # then type UCI commands (uci, position, go)
#
# machineplay runs the image as `docker run --rm -i --network none
# --cpuset-cpus N --cpus 1 --memory 512m ... dinora` and pipes UCI over
# stdin/stdout, so the container only has to launch the engine on stdio.
# That sandbox is why this image is CPU + ONNX: no GPU, no network at play
# time, and a single core.

ARG PYTHON_VERSION=3.12

FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim AS builder

ENV VIRTUAL_ENV=/app/.venv \
    UV_NO_CACHE=1

WORKDIR /src

# Only what the UCI engine imports at play time. The rest of the project's
# dependencies (torch, onnxruntime-gpu, wandb, cairosvg, ...) are for training
# and tooling and would multiply the image size for nothing.
# Versions mirror uv.lock — bump them together.
RUN uv venv "$VIRTUAL_ENV" \
 && uv pip install \
      "chess==1.10.0" \
      "numpy==2.1.1" \
      "onnxruntime==1.19.2"

# Then the engine, with --no-deps: the runtime set above is deliberate, so
# don't let pyproject's dependency list back in through the project install.
COPY pyproject.toml README.md ./
COPY src/dinora ./src/dinora
RUN uv pip install --no-deps .


FROM python:${PYTHON_VERSION}-slim-bookworm

# UCI is a line-based protocol over stdio: never buffer output.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

# Weights. `models/` is gitignored, so this builds against whatever net is in
# the working copy; a fresh clone has no `default.onnx` and must bring its own,
# or fall back to the released one (an older net than a locally retrained file):
#   wget -P models https://github.com/Saegl/dinora/releases/download/v0.3.0/default.onnx
# `<cwd>/models/default.onnx` is one of the places the engine looks
# (dinora.models.registry.search_places), and WORKDIR is /app.
COPY models/default.onnx ./models/default.onnx

# --model onnx: no torch in this image, so skip the torch-first fallback.
# --limit-threads: onnxruntime and numpy otherwise size their thread pools from
# the *host's* core count (/proc/cpuinfo isn't namespaced) and thrash on the one
# core the container is pinned to.
ENTRYPOINT [".venv/bin/python", "-m", "dinora", \
            "--model", "onnx", "--device", "cpu", "--limit-threads"]
