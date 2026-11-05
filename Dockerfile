FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements-api.txt /app/
COPY packages/ml /app/packages/ml
RUN python -m pip install -r requirements-api.txt ./packages/ml

COPY backend /app/backend
COPY frontend/index.html frontend/app.js frontend/api.js frontend/format.js frontend/styles.css /app/frontend/
RUN useradd --system --create-home --uid 10001 faultwise && \
    mkdir -p /app/data /app/artifacts/ml /app/artifacts/reports /app/artifacts/evaluation && \
    chown -R faultwise:faultwise /app/artifacts

USER faultwise
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["python", "-m", "uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
