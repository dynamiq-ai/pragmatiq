# PragmatIQ image for the demo / batch jobs (CPU-capable).
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
COPY pyproject.toml README.md ./
COPY pragmatiq ./pragmatiq
RUN pip install --no-cache-dir ".[demo]"
COPY demo ./demo
COPY configs ./configs
EXPOSE 8501
CMD ["streamlit", "run", "demo/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
