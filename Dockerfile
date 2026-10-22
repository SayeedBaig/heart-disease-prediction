FROM python:3.11.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    HF_HOME=/home/cardioai/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    CHAT_SESSION_CLEANUP_INTERVAL_SECONDS=900 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    TF_NUM_INTRAOP_THREADS=1 \
    TF_NUM_INTEROP_THREADS=1

WORKDIR /app

# Runtime libraries used by OpenCV, LightGBM, XGBoost, and TensorFlow.
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 cardioai \
    && useradd --uid 10001 --gid cardioai --create-home --shell /usr/sbin/nologin cardioai \
    && mkdir -p /app/uploads/ecg /app/uploads/echo /home/cardioai/.cache/huggingface \
    && chown -R cardioai:cardioai /app /home/cardioai

COPY requirements.txt ./requirements.txt
# Use CPU-only PyTorch and, on AMD64, the smaller CPU XGBoost distribution.
# XGBoost's CPU distribution does not publish ARM64 wheels, so ARM64 uses the
# pinned standard distribution. Keep requirements.txt as the version source.
RUN python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        "torch==2.11.0+cpu" \
    && if [ "$(dpkg --print-architecture)" = "amd64" ]; then \
        python -m pip install --no-cache-dir "xgboost-cpu==3.2.0"; \
    else \
        python -m pip install --no-cache-dir "xgboost==3.2.0"; \
    fi \
    && python -c "from pathlib import Path; p=Path('requirements.txt'); lines=p.read_text().splitlines(); p=Path('/tmp/requirements-docker.txt'); p.write_text('\\n'.join(line for line in lines if not line.startswith(('torch==','xgboost=='))) + '\\n')" \
    && python -m pip install --no-cache-dir -r /tmp/requirements-docker.txt \
    && python -m pip check

# Copy only the runtime code, trained artifacts, migrations, and vector store.
# Datasets, frontend dependencies, local environments, and secrets stay out.
COPY --chown=cardioai:cardioai api ./api
COPY --chown=cardioai:cardioai clinical_module/agent ./clinical_module/agent
COPY --chown=cardioai:cardioai clinical_module/alerts ./clinical_module/alerts
COPY --chown=cardioai:cardioai ecg_module ./ecg_module
COPY --chown=cardioai:cardioai echo_module ./echo_module
COPY --chown=cardioai:cardioai fusion_module ./fusion_module
COPY --chown=cardioai:cardioai digital_twin ./digital_twin
COPY --chown=cardioai:cardioai rag ./rag
COPY --chown=cardioai:cardioai reports ./reports
COPY --chown=cardioai:cardioai alembic ./alembic
COPY --chown=cardioai:cardioai alembic.ini ./alembic.ini

# Pin the embedding model revision so readiness does not depend on a first
# request or an outbound model download at container startup.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='sentence-transformers/all-MiniLM-L6-v2', revision='1110a243fdf4706b3f48f1d95db1a4f5529b4d41')" \
    && chown -R cardioai:cardioai /home/cardioai/.cache/huggingface

USER cardioai:cardioai

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; port=os.environ.get('PORT', '8000'); urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=3)"]

CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
