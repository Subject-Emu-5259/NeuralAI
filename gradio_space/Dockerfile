# NeuralAI v2 — Render Deployment
# Gradio Space container for the SmolLM2-360M LoRA adapter demo

FROM python:3.11-slim

WORKDIR /app

# System dependencies for torch/transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY gradio_space/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY gradio_space/app.py .

# Render sets PORT; Gradio binds to 0.0.0.0
ENV PORT=7860
EXPOSE 7860

CMD ["python", "app.py"]
