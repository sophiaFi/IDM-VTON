FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# libgl1 + libglib2.0 for opencv; gcc/g++ for pycocotools/basicsr build; ffmpeg for av (PyAV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    gcc \
    g++ \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# basicsr and fvcore use legacy setup.py builds that fail with setuptools>=67.
RUN pip install --no-cache-dir "setuptools<67" wheel

# torchvision first — basicsr's setup.py imports torch at metadata-collection time.
RUN pip install --no-cache-dir torchvision==0.15.2

# opencv-python-headless replaces opencv-python to avoid X11 deps in a headless container.
# onnxruntime-gpu instead of onnxruntime for CUDA 11.8 inference (same API, compatible version).
# peft is imported by train_with_measurements.py but absent from environment.yaml.
RUN pip install --no-cache-dir \
    accelerate==0.25.0 \
    torchmetrics==1.2.1 \
    tqdm==4.66.1 \
    transformers==4.36.2 \
    diffusers==0.25.0 \
    einops==0.7.0 \
    bitsandbytes==0.39.0 \
    scipy==1.11.1 \
    opencv-python-headless \
    gradio==4.24.0 \
    fvcore \
    cloudpickle \
    omegaconf \
    pycocotools \
    basicsr \
    av \
    onnxruntime-gpu==1.16.2 \
    "peft==0.7.1" \
    "huggingface_hub==0.23.4"

# Source code and IP-Adapter Python module
COPY src/ src/
COPY ip_adapter/ ip_adapter/
COPY train_with_measurements.py .

# Small checkpoint files only (.onnx, .pth for preprocessing models).
# Large HuggingFace weights (SDXL, IDM-VTON, image encoder) are downloaded at job start.
COPY ckpt/humanparsing/ ckpt/humanparsing/
COPY ckpt/openpose/ckpts/ ckpt/openpose/ckpts/
COPY ckpt/image_encoder/config.json ckpt/image_encoder/config.json

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/huggingface
