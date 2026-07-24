FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    JAX_ENABLE_X64=True

WORKDIR /workspace
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir ".[dev,stats]"
COPY . .
CMD ["pytest", "-q", "-m", "not scientific"]

