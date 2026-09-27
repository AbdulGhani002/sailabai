# One image definition for the Python services.
#   api:    FastAPI + tiles (no PyTorch, small)          EXTRAS=api
#   worker: daily scrapers + twin loop (CPU PyTorch)      EXTRAS=api,ml,data,geo  TORCH=cpu
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    SAILAB_DATA_DIR=/data SAILAB_RUNS_DIR=/data/runs SAILAB_RESULTS_DIR=/app/results
WORKDIR /app

ARG EXTRAS=api
ARG TORCH=none

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY results ./results

RUN if [ "$TORCH" = "cpu" ]; then \
      pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu; \
    fi && pip install ".[${EXTRAS}]"

RUN useradd --create-home sailab && mkdir -p /data && chown -R sailab /data /app
USER sailab

EXPOSE 8000
CMD ["uvicorn", "sailab.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
