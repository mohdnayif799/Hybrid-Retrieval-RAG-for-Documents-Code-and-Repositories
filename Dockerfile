# Hybrid-Retrieval RAG - production image for the Streamlit UI (Cloud Run).
#
#   docker build -t rag-qa .
#   docker run --rm -p 8080:8080 -e GOOGLE_API_KEY rag-qa
#
# Two stages. The builder has everything needed to *produce* the app (pip,
# network access to PyPI, the PyTorch index and Hugging Face); the runtime
# stage gets only the finished results: the virtualenv, the model weights and
# the code. Nothing a stage leaves behind reaches the final image unless it is
# copied across explicitly.
#
# Two Cloud Run rules shape this file:
#   * A container has 4 minutes to start listening, and that limit cannot be
#     raised. So nothing is downloaded at startup: both models are fetched
#     here, at build time, and the app is forced offline at runtime.
#   * Files written at runtime live in the instance's MEMORY; the image itself
#     does not count. So everything large is baked into the image, read-only.

# Pinned to an exact Python patch and Debian release so a rebuild next month
# gets the same interpreter; plain "3.12-slim" moves silently. Declared before
# the first FROM so both stages share one value: the venv's "python" is only a
# symlink to /usr/local/bin/python (see /opt/venv/pyvenv.cfg), so stage 2 must
# have the same interpreter at the same path or the copied venv breaks.
# Its SQLite is 3.46, well above ChromaDB's 3.35 minimum, which is why no
# pysqlite3 workaround is needed.
ARG PYTHON_IMAGE=python:3.12.14-slim-trixie


# =============================================================================
# Stage 1: builder
# =============================================================================

# "slim" rather than the full image: every dependency here ships a prebuilt
# manylinux wheel for Python 3.12, so no compiler or -dev headers are needed.
FROM ${PYTHON_IMAGE} AS builder

# No pip download cache (it would only bloat this stage's layers) and no
# "new pip available" check (an extra network call with nothing to act on).
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# A virtualenv gives the dependencies one self-contained directory, which is
# what makes the multi-stage copy clean: stage 2 takes /opt/venv and nothing
# else from pip. Same path in both stages, because a venv hardcodes its path.
RUN python -m venv /opt/venv

# Putting the venv first on PATH means "pip" and "python" below are the venv's,
# with no need to "activate" (activation does not persist across RUN lines).
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /app

# The dependency lists alone, BEFORE any source code. Docker caches each layer
# and reuses it while its inputs are unchanged, so editing app code skips the
# slow installs below; only a change to these two files re-runs them.
# constraints.txt pins every package to the exact version that was tested.
COPY requirements.txt constraints.txt ./

# PyTorch from the CPU-only index, on its own and FIRST. Nothing in
# requirements.txt names torch, but sentence-transformers and EasyOCR both
# depend on it, and a plain install from PyPI picks the CUDA build: several GB
# of NVIDIA libraries that are useless here, because Cloud Run has no GPU.
# Installing it first means the next step finds torch already satisfied and
# leaves it alone. torchvision comes from the same index because EasyOCR needs
# it and its build must match torch's exactly.
RUN pip install torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu \
        -c constraints.txt

# Everything else, from PyPI, at the pinned versions. The check afterwards
# fails the build if any NVIDIA/CUDA package slipped in anyway (for example
# if a future dependency pulled in a GPU build of torch), instead of silently
# shipping an image several GB larger.
RUN pip install -r requirements.txt -c constraints.txt \
 && if pip list 2>/dev/null | grep -qi '^nvidia-'; then \
        echo "CUDA packages were installed - the CPU-only torch step was bypassed" >&2; \
        exit 1; \
    fi

# Where the two models are stored. The runtime stage sets the same variables,
# so the app finds the weights at exactly these paths.
#   HF_HOME             sentence-transformers / Hugging Face cache
#   EASYOCR_MODULE_PATH EasyOCR looks for its weights under <this>/model
ENV HF_HOME=/opt/models/huggingface \
    EASYOCR_MODULE_PATH=/opt/models/easyocr

# Must match EMBEDDING_MODEL in src/vector_store.py. If the two ever differ,
# the app fails at the first query (it runs offline and cannot fetch the
# missing model), and CI catches it by loading the model through the app's own
# code inside the built image.
ARG EMBEDDING_MODEL=all-MiniLM-L6-v2

# Download both models now, by loading them exactly the way the app does, so
# the files on disk are precisely the ones the app will ask for:
#   * all-MiniLM-L6-v2 (~90 MB) for embeddings
#   * EasyOCR's English detector + recognizer (~95 MB) for scanned PDFs
# Then drop download leftovers: hf-xet's chunk cache (a second copy of the
# same bytes) and empty lock files. chmod makes sure the unprivileged runtime
# user can read everything, whatever permissions the downloaders chose.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}', device='cpu')" \
 && python -c "import easyocr; easyocr.Reader(['en'], gpu=False, verbose=False)" \
 && rm -rf "${HF_HOME}/xet" "${HF_HOME}/hub/.locks" \
 && chmod -R a+rX /opt/models


# =============================================================================
# Stage 2: runtime
# =============================================================================

# Fresh copy of the same base: no pip, no build caches, no stage-1 history.
# This is the only stage that ships.
FROM ${PYTHON_IMAGE} AS runtime

# Image metadata, readable with "docker inspect" and shown by registries.
LABEL org.opencontainers.image.title="Hybrid-Retrieval RAG" \
      org.opencontainers.image.description="Hybrid BM25 + dense retrieval RAG over documents and GitHub repositories (Streamlit)" \
      org.opencontainers.image.source="https://github.com/mohdnayif799/Hybrid-Retrieval-RAG-for-Documents-Code-and-Repositories"

# PYTHONDONTWRITEBYTECODE: the app user cannot write into /app, so skip trying.
# PYTHONUNBUFFERED: logs reach "docker logs" / Cloud Logging immediately
#   instead of sitting in a buffer (and being lost if the container dies).
# PATH: use the venv copied from the builder.
# HF_HOME / EASYOCR_MODULE_PATH: where the baked-in weights are (see stage 1).
# HF_HUB_OFFLINE: never contact Hugging Face at runtime. The model is already
#   on disk; without this, every cold start would first check the Hub for a
#   newer version, which costs startup time and fails outright if the Hub is
#   slow or down.
# ANONYMIZED_TELEMETRY: ChromaDB's usage telemetry, off for a server.
# STREAMLIT_SERVER_HEADLESS: never try to open a browser, never prompt for an
#   email on first run - a prompt would block startup with no one to answer.
# STREAMLIT_BROWSER_GATHER_USAGE_STATS: no telemetry from a server deployment.
# STREAMLIT_SERVER_FILE_WATCHER_TYPE: the code never changes inside an image,
#   so watching it for hot-reload only costs CPU.
# PORT: default 8080. Cloud Run injects its own PORT, which overrides this.
# RAG_MODEL: the Gemini model the UI starts on. Flash-Lite because one query
#   makes 2-3 model calls and its free daily quota is far larger than Flash's.
# RAG_STORE_DIR: where each chat's vector store is created (see below).
# RAG_REPO_MAX_*: smaller repository limits than the app's defaults. The store
#   a visitor is building is the one store the eviction policy never removes,
#   so on a shared demo these limits are what bound a single visitor.
#   (Uploads are capped the same way, in .streamlit/config.toml.)
# None of these are secrets, so they are safe to bake in.
#
# GOOGLE_API_KEY is deliberately NOT here. Anything in ENV is baked into the
# image and readable by anyone who can pull it ("docker inspect" prints it).
# The key is supplied only at runtime: "docker run -e" locally, or a Cloud Run
# environment variable / Secret Manager reference in production.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    HF_HOME=/opt/models/huggingface \
    HF_HUB_OFFLINE=1 \
    EASYOCR_MODULE_PATH=/opt/models/easyocr \
    ANONYMIZED_TELEMETRY=False \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    PORT=8080 \
    RAG_MODEL=gemini-3.5-flash-lite \
RAG_STORE_DIR=/app/stores \
RAG_STORE_MAX_COUNT=8 \
RAG_STORE_MAX_MB=128 \
RAG_STORE_IDLE_MINUTES=30 \
RAG_REPO_MAX_FILES=300 \
RAG_REPO_MAX_ARCHIVE_MB=20 \
RAG_REPO_MAX_TEXT_MB=20

# A dedicated unprivileged user. If an attacker ever gets code execution in the
# app, they land as a user who cannot modify the code, the venv or the models -
# all of which stay owned by root. A fixed, high UID/GID (10001) cannot collide
# with an account in the base image and stays stable across rebuilds. A home
# directory is created because Streamlit looks for per-user config under
# ~/.streamlit. The login shell is disabled: no one should log in as this.
#
# /app/stores is the one directory the app writes to: each chat's ChromaDB
# store is created there (RAG_STORE_DIR above), and deleted again by the
# eviction policy in src/store_registry.py. So it is the only directory handed
# to the app user; everything else under /app stays owned by root.
RUN groupadd --gid 10001 app \
 && useradd --uid 10001 --gid app --create-home --home-dir /home/app \
        --shell /usr/sbin/nologin app \
 && mkdir -p /app/stores \
 && chown app:app /app/stores

WORKDIR /app

# Layers ordered from least to most frequently changed: dependencies, then the
# models, then the source. A code edit only re-runs the last two COPYs.
# Everything is owned by root (COPY's default), so the app user can read it
# but not change it.
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/models /opt/models

# Only what runs: the entry point, the package, the Streamlit config (upload
# limit) and the bundled example repository (a 60 KB tag tarball, so "Try an
# example" never calls GitHub). Tests, benchmark data, evaluation scripts and
# docs stay out, and .dockerignore keeps secrets out of the build context.
# The config has to be copied explicitly: without it the app still starts,
# just silently back on Streamlit's 200 MB upload limit.
COPY .streamlit/config.toml .streamlit/config.toml
COPY examples/ examples/
COPY app.py demo_key.py ./
COPY src/ src/

# Drop root for everything that follows, including the running app. Numeric
# IDs let Kubernetes-style runtimes check "runAsNonRoot" without reading
# /etc/passwd.
USER 10001:10001

# Documents the default port. It does not publish anything: "docker run -p"
# does that locally, and Cloud Run routes to $PORT on its own.
EXPOSE 8080

# Lets "docker ps" show healthy/unhealthy. Streamlit serves /_stcore/health.
# Python's urllib is used because slim images ship neither curl nor wget.
# Shell form here on purpose, so ${PORT} is expanded when the check runs.
# (Cloud Run ignores HEALTHCHECK and uses its own startup probe.)
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:${PORT}/_stcore/health', timeout=4)" || exit 1

# Why not the plain exec form, CMD ["streamlit", ..., "--server.port=${PORT}"]?
#   No shell runs it, so "${PORT}" would reach Streamlit as literal text.
# Why not the shell form, CMD streamlit run ... ?
#   /bin/sh becomes PID 1 and does not forward SIGTERM, so "docker stop" and
#   Cloud Run scale-down would wait out the grace period and then SIGKILL.
# So: exec form that starts a shell explicitly (the shell expands ${PORT}),
# then "exec" replaces the shell with Streamlit. Streamlit becomes PID 1 and
# receives SIGTERM directly. 0.0.0.0 = listen on every interface, not just
# loopback: traffic from "docker run -p" or Cloud Run arrives on the
# container's network interface, never on 127.0.0.1.
CMD ["sh", "-c", "exec streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT}"]
