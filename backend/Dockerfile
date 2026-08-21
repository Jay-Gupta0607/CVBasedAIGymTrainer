# Build stage - Install all dependencies
FROM python:3.11-slim as builder
WORKDIR /app
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

# ML dependencies stage - Install heavy ML dependencies
FROM python:3.11-slim as ml-deps
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
RUN pip install --no-cache-dir onnxruntime-gpu==1.18.1

# Runtime stage - CUDA enabled
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 as runtime
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from ml-deps stage
COPY --from=ml-deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=ml-deps /usr/local/bin /usr/local/bin

# Copy application code
COPY . /app

# Create temp directory
RUN mkdir -p /tmp/gym_trainer

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV ONNXRUNTIME_EXECUTION_PROVIDERS=TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/api/v1/health', timeout=5).raise_for_status()" || exit 1

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]