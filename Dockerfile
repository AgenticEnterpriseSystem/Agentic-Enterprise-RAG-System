# syntax=docker/dockerfile:1
FROM node:22-bookworm-slim@sha256:4b150b48b4116e2a22f2c39cc27025fbf4f153e2078fd3b14d159fdb5c7279da AS frontend
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/index.html frontend/vite.config.js ./
COPY frontend/src ./src
RUN npm run build

FROM python:3.11-slim-bookworm@sha256:94a8fee35252d06ab75856f15f12c7bae67b10e8a6da93c7fded82b2610926cd AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    RAG_FRONTEND_DIST=/app/frontend/dist \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    TOKENIZERS_PARALLELISM=false
WORKDIR /app
ARG INSTALL_HYBRID=true
COPY requirements.lock.txt ./
COPY scripts/docker_dependencies.py ./scripts/docker_dependencies.py
RUN test "$INSTALL_HYBRID" = true || test "$INSTALL_HYBRID" = false
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/* && \
    if [ "$INSTALL_HYBRID" = true ]; then \
      python -m pip install --no-cache-dir --no-deps \
        --index-url https://download.pytorch.org/whl/cpu 'torch==2.14.1+cpu'; \
    fi && \
    python scripts/docker_dependencies.py --hybrid "$INSTALL_HYBRID" --output /tmp/runtime-requirements.txt && \
    python -m pip install --no-cache-dir --no-deps -r /tmp/runtime-requirements.txt && \
    python -m pip check && rm /tmp/runtime-requirements.txt

RUN useradd --create-home --uid 10001 app && \
    mkdir -p /app/.cache/models /data/corpus /data/indexes /state && \
    chown -R app:app /app/.cache /data /state
COPY src ./src
COPY alembic.ini ./
COPY migrations ./migrations
COPY scripts/local_data.py scripts/local_stack.py ./scripts/
COPY --from=frontend /web/dist ./frontend/dist

# Source verification and interactive account/policy administration are separate
# operator jobs. Git never enters the serving stage and no credentials are baked.
FROM runtime AS operator
USER root
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*
USER app
CMD ["python", "-m", "src.db.migrate"]

FROM runtime AS app
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8000/api/ready", timeout=3)' || exit 1
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
