# Task MCP Server - Multi-stage build
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create venv and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements from backend
COPY ../../requirements.txt .
RUN pip install --upgrade pip setuptools && \
    pip install -r requirements.txt

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY ../../app /app/app
COPY task_mcp /app/task_mcp

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Run MCP server
CMD ["python", "-m", "task_mcp.main"]
