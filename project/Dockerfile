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
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON=3.12 \
    PYTHONPATH=/app \
    ALLOW_ANON=0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Teaching-pack parquets + grid runs baked in
COPY . .

# Ensure precomputed data exists (regenerate if missing)
RUN if [ ! -f cache/precomputed/manifest.json ]; then \
      echo "Precomputed data missing, running precompute..." && \
      uv run python scripts/precompute_data.py || true; \
    else \
      echo "Precomputed data found, skipping..."; \
    fi

EXPOSE 8080

CMD ["sh", "-c", ".venv/bin/streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT --server.enableCORS=false"]
