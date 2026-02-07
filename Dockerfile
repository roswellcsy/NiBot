# ---- build stage ----
FROM python:3.13-slim AS builder

WORKDIR /build
COPY pyproject.toml .
COPY nibot/ nibot/

RUN pip install --no-cache-dir --prefix=/install . && \
    pip install --no-cache-dir --prefix=/install ".[all]" || \
    echo "Optional extras not available, continuing with base install"

# ---- runtime stage ----
FROM python:3.13-slim

LABEL maintainer="NiBot"
LABEL description="NiBot AI Agent Framework"

RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash nibot

COPY --from=builder /install /usr/local
COPY nibot/ /app/nibot/
WORKDIR /app

RUN mkdir -p /home/nibot/.nibot/workspace && \
    chown -R nibot:nibot /home/nibot/.nibot

USER nibot

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -sf http://localhost:9100/health || exit 1

ENV NIBOT_LOG__JSON_FORMAT=true
ENV NIBOT_LOG__LEVEL=INFO
ENV NIBOT_HEALTH__ENABLED=true
ENV NIBOT_HEALTH__HOST=0.0.0.0

ENTRYPOINT ["python", "-m", "nibot"]
CMD ["--config", "/home/nibot/.nibot/config.json"]
