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
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the base model at BUILD time so the container binds its
# port fast. Otherwise the 720MB model downloads during startup and
# Railway's healthcheck times out (crash loop). Cached to
# /root/.cache/huggingface; the runtime from_pretrained() reuses it.
RUN python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('HuggingFaceTB/SmolLM2-360M-Instruct', trust_remote_code=True)"

# Copy application code
COPY app.py .

# Render sets PORT; Gradio binds to 0.0.0.0
ENV PORT=7860
EXPOSE 7860

CMD ["python", "app.py"]
