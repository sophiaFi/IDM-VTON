import sys
import os
import argparse

sys.path.append('./')
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from PIL import Image
import gradio as gr
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from src.measurement_encoder import MeasurementEncoder, normalize_measurements
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from diffusers import DDPMScheduler, AutoencoderKL
from typing import List

import torch
from transformers import AutoTokenizer
import numpy as np
from utils_mask import get_mask_location
from torchvision import transforms
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
from torchvision.transforms.functional import to_pil_image

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--checkpoint_dir",
    type=str,
    default=None,
    help="Path to a training checkpoint folder containing lora/, conv_in.pt, measurement_encoder.pt",
)
parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
args_cli, _ = parser.parse_known_args()

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def pil_to_binary_mask(pil_image, threshold=0):
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            if binary_mask[i, j]:
                mask[i, j] = 1
    mask = (mask * 255).astype(np.uint8)
    return Image.fromarray(mask)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
base_path = "yisol/IDM-VTON"
example_path = os.path.join(os.path.dirname(__file__), "example")

print("Loading base model weights...")
unet = UNet2DConditionModel.from_pretrained(
    base_path,
    subfolder="unet",
    torch_dtype=torch.float16,
)
unet.requires_grad_(False)

tokenizer_one = AutoTokenizer.from_pretrained(base_path, subfolder="tokenizer", use_fast=False)
tokenizer_two = AutoTokenizer.from_pretrained(base_path, subfolder="tokenizer_2", use_fast=False)
noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")

text_encoder_one = CLIPTextModel.from_pretrained(
    base_path, subfolder="text_encoder", torch_dtype=torch.float16
)
text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
    base_path, subfolder="text_encoder_2", torch_dtype=torch.float16
)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    base_path, subfolder="image_encoder", torch_dtype=torch.float16
)
vae = AutoencoderKL.from_pretrained(base_path, subfolder="vae", torch_dtype=torch.float16)
UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
    base_path, subfolder="unet_encoder", torch_dtype=torch.float16
)

# ---------------------------------------------------------------------------
# Load fine-tuned checkpoint (LoRA + conv_in + MeasurementEncoder)
# ---------------------------------------------------------------------------
cross_attn_dim = unet.config.cross_attention_dim  # 2048 for SDXL
measurement_encoder = MeasurementEncoder(
    num_measurements=7,
    hidden_dim=256,
    output_dim=cross_attn_dim,
    dropout=0.0,
    use_fourier=False,
)

if args_cli.checkpoint_dir is not None:
    ckpt = args_cli.checkpoint_dir
    print(f"Loading checkpoint from {ckpt} ...")

    # LoRA adapters
    lora_path = os.path.join(ckpt, "lora")
    if os.path.isdir(lora_path):
        from peft import PeftModel
        unet = PeftModel.from_pretrained(unet, lora_path)
        unet = unet.merge_and_unload()
        print("  LoRA weights merged.")
    else:
        print(f"  WARNING: no lora/ folder found in {ckpt}, skipping LoRA.")

    # Expanded conv_in weights (9→13 channels)
    conv_in_path = os.path.join(ckpt, "conv_in.pt")
    if os.path.isfile(conv_in_path):
        # Expand conv_in to 13 channels if still at 9
        if unet.conv_in.in_channels == 9:
            conv_new = torch.nn.Conv2d(
                in_channels=13,
                out_channels=unet.conv_in.out_channels,
                kernel_size=3,
                padding=1,
            )
            torch.nn.init.zeros_(conv_new.weight)
            conv_new.weight.data[:, :9] = unet.conv_in.weight.data
            conv_new.bias.data = unet.conv_in.bias.data
            unet.conv_in = conv_new
        unet.conv_in.load_state_dict(torch.load(conv_in_path, map_location="cpu"))
        print("  conv_in weights loaded.")
    else:
        print(f"  WARNING: no conv_in.pt found in {ckpt}, skipping.")

    # MeasurementEncoder weights
    menc_path = os.path.join(ckpt, "measurement_encoder.pt")
    if os.path.isfile(menc_path):
        measurement_encoder.load_state_dict(torch.load(menc_path, map_location="cpu"))
        print("  MeasurementEncoder weights loaded.")
    else:
        print(f"  WARNING: no measurement_encoder.pt found in {ckpt}, using random weights.")
else:
    print("No --checkpoint_dir given; running with base IDM-VTON weights (measurements ignored).")

measurement_encoder.eval()
measurement_encoder.requires_grad_(False)

# ---------------------------------------------------------------------------
# Preprocessing models
# ---------------------------------------------------------------------------
parsing_model = Parsing(0)
openpose_model = OpenPose(0)

UNet_Encoder.requires_grad_(False)
image_encoder.requires_grad_(False)
vae.requires_grad_(False)
unet.requires_grad_(False)
text_encoder_one.requires_grad_(False)
text_encoder_two.requires_grad_(False)

tensor_transfrom = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
)

pipe = TryonPipeline.from_pretrained(
    base_path,
    unet=unet,
    vae=vae,
    feature_extractor=CLIPImageProcessor(),
    text_encoder=text_encoder_one,
    text_encoder_2=text_encoder_two,
    tokenizer=tokenizer_one,
    tokenizer_2=tokenizer_two,
    scheduler=noise_scheduler,
    image_encoder=image_encoder,
    torch_dtype=torch.float16,
)
pipe.unet_encoder = UNet_Encoder


# ---------------------------------------------------------------------------
# Inference function
# ---------------------------------------------------------------------------
def start_tryon(
    dict,
    garm_img,
    garment_des,
    is_checked,
    is_checked_crop,
    denoise_steps,
    seed,
    body_bust,
    body_height,
    body_hips,
    body_waist,
    garment_bust,
    garment_length,
    garment_sleeve_length,
):
    openpose_model.preprocessor.body_estimation.model.to(device)
    pipe.to(device)
    pipe.unet_encoder.to(device)
    measurement_encoder.to(device)

    garm_img = garm_img.convert("RGB").resize((768, 1024))
    human_img_orig = dict["background"].convert("RGB")

    if is_checked_crop:
        width, height = human_img_orig.size
        target_width = int(min(width, height * (3 / 4)))
        target_height = int(min(height, width * (4 / 3)))
        left = (width - target_width) / 2
        top = (height - target_height) / 2
        right = (width + target_width) / 2
        bottom = (height + target_height) / 2
        cropped_img = human_img_orig.crop((left, top, right, bottom))
        crop_size = cropped_img.size
        human_img = cropped_img.resize((768, 1024))
    else:
        human_img = human_img_orig.resize((768, 1024))

    if is_checked:
        keypoints = openpose_model(human_img.resize((384, 512)))
        model_parse, _ = parsing_model(human_img.resize((384, 512)))
        mask, mask_gray = get_mask_location("hd", "upper_body", model_parse, keypoints)
        mask = mask.resize((768, 1024))
    else:
        mask = pil_to_binary_mask(dict["layers"][0].convert("RGB").resize((768, 1024)))

    mask_gray = (1 - transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
    mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)

    human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

    densepose_args = apply_net.create_argument_parser().parse_args((
        "show",
        "./configs/densepose_rcnn_R_50_FPN_s1x.yaml",
        "./ckpt/densepose/model_final_162be9.pkl",
        "dp_segm", "-v", "--opts", "MODEL.DEVICE", "cuda",
    ))
    pose_img = densepose_args.func(densepose_args, human_img_arg)
    pose_img = pose_img[:, :, ::-1]
    pose_img = Image.fromarray(pose_img).resize((768, 1024))

    # Build measurement token
    measurements_dict = {
        "body_bust": body_bust,
        "body_height": body_height,
        "body_hips": body_hips,
        "body_waist": body_waist,
        "garment_bust": garment_bust,
        "garment_length": garment_length,
        "garment_sleeve_length": garment_sleeve_length,
    }
    normalized = normalize_measurements(measurements_dict).unsqueeze(0).to(device)  # [1, 7]
    with torch.no_grad():
        measurement_tokens = measurement_encoder(normalized)  # [1, 1, 2048]
        measurement_tokens = measurement_tokens.to(torch.float16)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            prompt = "model is wearing " + garment_des
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
            with torch.inference_mode():
                (
                    prompt_embeds,
                    negative_prompt_embeds,
                    pooled_prompt_embeds,
                    negative_pooled_prompt_embeds,
                ) = pipe.encode_prompt(
                    prompt,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                    negative_prompt=negative_prompt,
                )

                prompt_c = "a photo of " + garment_des
                negative_prompt_c = "monochrome, lowres, bad anatomy, worst quality, low quality"
                with torch.inference_mode():
                    (
                        prompt_embeds_c,
                        _,
                        _,
                        _,
                    ) = pipe.encode_prompt(
                        [prompt_c],
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=False,
                        negative_prompt=[negative_prompt_c],
                    )

                pose_img_t = tensor_transfrom(pose_img).unsqueeze(0).to(device, torch.float16)
                garm_tensor = tensor_transfrom(garm_img).unsqueeze(0).to(device, torch.float16)
                generator = torch.Generator(device).manual_seed(seed) if seed is not None else None

                images = pipe(
                    prompt_embeds=prompt_embeds.to(device, torch.float16),
                    negative_prompt_embeds=negative_prompt_embeds.to(device, torch.float16),
                    pooled_prompt_embeds=pooled_prompt_embeds.to(device, torch.float16),
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, torch.float16),
                    num_inference_steps=denoise_steps,
                    generator=generator,
                    strength=1.0,
                    pose_img=pose_img_t,
                    text_embeds_cloth=prompt_embeds_c.to(device, torch.float16),
                    cloth=garm_tensor,
                    mask_image=mask,
                    image=human_img,
                    height=1024,
                    width=768,
                    ip_adapter_image=garm_img.resize((768, 1024)),
                    guidance_scale=2.0,
                    measurement_tokens=measurement_tokens,
                )[0]

    if is_checked_crop:
        out_img = images[0].resize(crop_size)
        human_img_orig.paste(out_img, (int(left), int(top)))
        return human_img_orig, mask_gray
    else:
        return images[0], mask_gray


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
garm_list = os.listdir(os.path.join(example_path, "cloth"))
garm_list_path = [os.path.join(example_path, "cloth", g) for g in garm_list]

human_list = os.listdir(os.path.join(example_path, "human"))
human_list_path = [os.path.join(example_path, "human", h) for h in human_list]

human_ex_list = []
for ex_human in human_list_path:
    human_ex_list.append({"background": ex_human, "layers": None, "composite": None})

image_blocks = gr.Blocks().queue()
with image_blocks as demo:
    gr.Markdown("## IDM-VTON + Measurements")
    gr.Markdown(
        "Virtual try-on conditioned on body and garment measurements. "
        "Upload a person photo, a garment image, enter measurements, then click **Try-on**."
    )
    with gr.Row():
        with gr.Column():
            imgs = gr.ImageEditor(
                sources="upload",
                type="pil",
                label="Person photo — draw mask or use auto-masking",
                interactive=True,
            )
            with gr.Row():
                is_checked = gr.Checkbox(
                    label="Auto-generate mask", info="Takes ~5 seconds", value=True
                )
                is_checked_crop = gr.Checkbox(
                    label="Auto-crop & resize", value=False
                )
            gr.Examples(inputs=imgs, examples_per_page=10, examples=human_ex_list)

        with gr.Column():
            garm_img = gr.Image(label="Garment", sources="upload", type="pil")
            with gr.Row():
                prompt = gr.Textbox(
                    placeholder="Garment description, e.g. Short Sleeve Round Neck T-shirt",
                    show_label=False,
                )
            gr.Examples(inputs=garm_img, examples_per_page=8, examples=garm_list_path)

        with gr.Column():
            masked_img = gr.Image(
                label="Masked preview", elem_id="masked-img", show_share_button=False
            )

        with gr.Column():
            image_out = gr.Image(
                label="Result", elem_id="output-img", show_share_button=False
            )

    with gr.Column():
        gr.Markdown("### Measurements (cm)")
        with gr.Row():
            body_bust = gr.Number(label="Body bust", value=105, minimum=60, maximum=160)
            body_height = gr.Number(label="Body height", value=172, minimum=140, maximum=220)
            body_hips = gr.Number(label="Body hips", value=107, minimum=60, maximum=170)
            body_waist = gr.Number(label="Body waist", value=91, minimum=50, maximum=150)
        with gr.Row():
            garment_bust = gr.Number(label="Garment bust", value=115, minimum=60, maximum=180)
            garment_length = gr.Number(label="Garment length", value=54, minimum=20, maximum=120)
            garment_sleeve_length = gr.Number(
                label="Garment sleeve length", value=30, minimum=0, maximum=100
            )

    with gr.Column():
        try_button = gr.Button(value="Try-on")
        with gr.Accordion(label="Advanced Settings", open=False):
            with gr.Row():
                denoise_steps = gr.Number(
                    label="Denoising steps", minimum=20, maximum=40, value=30, step=1
                )
                seed = gr.Number(
                    label="Seed", minimum=-1, maximum=2147483647, step=1, value=42
                )

    try_button.click(
        fn=start_tryon,
        inputs=[
            imgs, garm_img, prompt, is_checked, is_checked_crop, denoise_steps, seed,
            body_bust, body_height, body_hips, body_waist,
            garment_bust, garment_length, garment_sleeve_length,
        ],
        outputs=[image_out, masked_img],
        api_name="tryon",
    )

image_blocks.launch(share=args_cli.share)
