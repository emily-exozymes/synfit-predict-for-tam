FROM mambaorg/micromamba:1.5.8

USER root
WORKDIR /app

# Install Python + PyTorch (CUDA 12.1) + SynFit deps.
# Source code is mounted at runtime, not copied here.
RUN micromamba install -y -n base -c conda-forge \
    python=3.10 \
    "setuptools<81" \
    pip \
    git \
    && micromamba clean -a -y

ENV PATH=/opt/conda/bin:$PATH

# PyTorch 2.4.0 + CUDA 12.1 wheels (matches SynFit README)
RUN pip install --no-cache-dir \
    torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# SynFit core deps (subset needed for inference; training-only deps trimmed)
RUN pip install --no-cache-dir \
    "transformers>=4.40,<4.50" \
    "tokenizers>=0.19,<0.21" \
    "huggingface_hub>=0.23" \
    "accelerate>=0.30" \
    "peft>=0.10" \
    "safetensors>=0.4" \
    numpy==1.26.4 \
    scipy==1.15.3 \
    "pandas>=2.0" \
    "scikit-learn>=1.3" \
    "einops>=0.7" \
    biopython==1.85 \
    PyYAML \
    tqdm

# Pre-cache the ESM2 backbone so the container can run without internet at runtime.
# Tamarind runtime containers have NO outbound network - model weights MUST be baked in here.
RUN python -c "from transformers import EsmForMaskedLM, EsmTokenizer; \
    EsmForMaskedLM.from_pretrained('facebook/esm2_t33_650M_UR50D'); \
    EsmTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D')"

RUN mkdir -p inputs out && chmod -R 777 /app
