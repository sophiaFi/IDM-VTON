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
    huggingface_hub==0.23.4 \
    tensorboard==2.14.0 \
    onnxruntime-gpu==1.16.3 \
    opencv-python-headless==4.8.1.78 \
    matplotlib==3.7.4 \
    scikit-image==0.21.0 \
    pycocotools \
    fvcore \
    yacs \
    omegaconf \
    termcolor \
    tabulate \
    iopath \
    cloudpickle \
    av

COPY src/ src/
COPY ip_adapter/ ip_adapter/
COPY configs/ configs/
COPY preprocess/ preprocess/
COPY gradio_demo/utils_mask.py gradio_demo/utils_mask.py
COPY gradio_demo/apply_net.py gradio_demo/apply_net.py
COPY gradio_demo/detectron2/ gradio_demo/detectron2/
COPY gradio_demo/densepose/ gradio_demo/densepose/
COPY train_with_measurements.py .
COPY train_hpo.py .

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/huggingface
