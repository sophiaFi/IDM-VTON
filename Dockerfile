FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torchvision first — some packages' setup.py imports torch at metadata-collection time.
RUN pip install --no-cache-dir torchvision==0.15.2

RUN pip install --no-cache-dir \
    accelerate==0.25.0 \
    torchmetrics==1.2.1 \
    tqdm==4.66.1 \
    transformers==4.36.2 \
    diffusers==0.25.0 \
    einops==0.7.0 \
    bitsandbytes==0.39.0 \
    scipy==1.11.1 \
    peft==0.7.1 \
    huggingface_hub==0.23.4

COPY src/ src/
COPY ip_adapter/ ip_adapter/
COPY train_with_measurements.py .

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/huggingface
