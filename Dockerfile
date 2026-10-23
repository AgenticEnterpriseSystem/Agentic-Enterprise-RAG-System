# syntax=docker/dockerfile:1
FROM node:22-bookworm-slim AS frontend
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/index.html frontend/vite.config.js ./
COPY frontend/src ./src
RUN npm run build

FROM python:3.11-slim-bookworm AS app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    RAG_FRONTEND_DIST=/app/frontend/dist
WORKDIR /app

# BM25/fixtures don't import the dense encoder. Keep versions in requirements.txt
# and opt into the larger CPU model stack only when it is actually needed.
ARG INSTALL_HYBRID=false
COPY requirements.txt ./
RUN test "$INSTALL_HYBRID" = true || test "$INSTALL_HYBRID" = false
RUN if [ "$INSTALL_HYBRID" = true ]; then \
      apt-get update && apt-get install -y --no-install-recommends libgomp1 && \
      rm -rf /var/lib/apt/lists/* && \
      python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch; \
    fi && \
    INSTALL_HYBRID="$INSTALL_HYBRID" python -c 'import os; from pathlib import Path; omitted={"pytest"}; omitted.update({"sentence-transformers", "faiss-cpu"} if os.environ["INSTALL_HYBRID"] == "false" else set()); Path("/tmp/runtime-requirements.txt").write_text("\n".join(line for line in Path("requirements.txt").read_text().splitlines() if line.split("==")[0] not in omitted))' && \
    python -m pip install --no-cache-dir -r /tmp/runtime-requirements.txt

RUN useradd --create-home --uid 10001 app && \
    mkdir -p /app/.cache/models && chown -R app:app /app/.cache
COPY src ./src
COPY --from=frontend /web/dist ./frontend/dist
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8000/api/ready", timeout=3)' || exit 1
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
