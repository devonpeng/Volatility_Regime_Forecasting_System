# RV Prediction Lab — Streamlit app
# Build from this directory: docker build -t rv-lab .
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Layout: /app/project/{app.py,src,...} so `import project...` works
WORKDIR /app/project

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON=3.12 \
    PYTHONPATH=/app \
    ALLOW_ANON=0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Teaching-pack parquets + grid runs baked in; custom_* runs excluded via .dockerignore
COPY . .

RUN mkdir -p data cache/precomputed

EXPOSE 8501

# Use the image venv directly so startup does not re-resolve Python
CMD [".venv/bin/streamlit", "run", "streamlit-docker-test/app.py", \
     "--server.address=0.0.0.0", "--server.port=8080"]

# CMD .venv/bin/streamlit run streamlit-docker-test/app.py \
#     --server.address=0.0.0.0 \
#     --server.port=${PORT:-8501}