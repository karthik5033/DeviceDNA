FROM python:3.11-slim

WORKDIR /app

# Install PyTorch CPU-only first (much smaller: ~200MB vs 755MB CUDA version)
# Separated to maximize Docker layer caching and avoid timeout on the largest download
RUN pip install --no-cache-dir --timeout 300 \
    torch==2.3.0+cpu torchvision==0.18.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 120 -r requirements.txt

# Source code is mounted as a volume at runtime, so no COPY needed
