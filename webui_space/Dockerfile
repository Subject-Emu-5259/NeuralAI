# NeuralAI Web Chat — Hugging Face Docker Space
# Serves the existing Flask web UI (from-scratch/web_ui) via the unified
# neural_core_service.py backend. CPU-only (SmolLM2-360M is tiny).
FROM python:3.12-slim

WORKDIR /app

# System deps for torch/transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cached layer)
COPY webui_space/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy only what the web service needs (NOT the 1.5GB checkpoints/nextcloud).
# The LoRA adapter is pulled at runtime from the HF Hub (ADAPTER_REPO).
COPY services/ /app/services/
COPY from-scratch/web_ui/ /app/from-scratch/web_ui/
COPY data/ /app/data/
COPY tools/ /app/tools/

# The service reads REPO_ROOT from its own location
ENV REPO_ROOT=/app
ENV PORT=7860
ENV ADAPTER_REPO=Subject-Emu-5259/NeuralAI
ENV QUANTIZE=1
ENV HF_HUB_DISABLE_TELEMETRY=1
EXPOSE 7860

CMD ["python", "services/neural_core_service.py"]
