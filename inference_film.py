"""
inference_film.py — Run IDM-VTON + FiLM inference on FIT dataset samples.

Two modes:

  Without --checkpoint_dir (default):
    Loads yisol/IDM-VTON pretrained weights and runs inference on FIT data.
    Measurements are read but not injected (the pretrained model has no
    FiLM conditioning). Use this to verify the data pipeline and get a
    baseline before any fine-tuning.

  With --checkpoint_dir <path>:
    Loads the base model specified by --pretrained_model_name_or_path (must be
    the SAME base used during training, default yisol/IDM-VTON), then overlays
    LoRA conv weights + conv_in + MeasurementEncoder + FiLMGeneratorSet from
    the checkpoint. Measurements are injected via FiLM modulation
    (element-wise γ/β scaling applied after norm2 in each target ResNet block),
    NOT by appending tokens to the cross-attention sequence.

    Checkpoint layout expected:
      checkpoint-<step>/
        unet_lora.pt           ← trainable UNet params (LoRA A/B + norms + conv_in)
        film_generators.pt     ← FiLMGeneratorSet state dict
        measurement_encoder.pt ← MeasurementEncoder state dict

    IMPORTANT: --pretrained_model_name_or_path must match the training base model.
    Training uses yisol/IDM-VTON (13-channel UNet). Using a different base will
    corrupt outputs because LoRA deltas are additive offsets on the training base.

Example (quick single-sample test, no checkpoint):
    python inference_film.py --data_dir data/data_mini_test --num_samples 1

Example (after training):
    python inference_film.py \\
        --data_dir data/data_mini_test \\
        --checkpoint_dir output/checkpoint-500 \\
        --output_dir result_film \\
        --num_samples 4
"""

import argparse
import math
import os
import sys
from typing import Optional

import torch
from PIL import Image
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)

# Preprocessing models live inside gradio_demo/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "gradio_demo"))

from ip_adapter.ip_adapter import Resampler
from src.fit_dataset import FITDatasetWithMeasurements
from src.film_generator import FiLMGeneratorSet
from src.measurement_encoder import MeasurementEncoder
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# FiLM hook infrastructure (mirrors train_with_film.py)
# ---------------------------------------------------------------------------

class _FiLMContext:
    embedding: Optional[torch.Tensor] = None


class LoRAConv2d(torch.nn.Module):
    """Frozen Conv2d with a trainable low-rank bypass ΔW = B·A."""

    def __init__(self, conv: torch.nn.Conv2d, rank: int = 4) -> None:
        super().__init__()
        self.conv = conv
        self.conv.requires_grad_(False)
        C_in, C_out = conv.in_channels, conv.out_channels
        k, pad = conv.kernel_size[0], conv.padding[0]
        self.lora_A = torch.nn.Conv2d(C_in, rank, 1, bias=False)
        self.lora_B = torch.nn.Conv2d(rank, C_out, k, padding=pad, bias=False)
        torch.nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.lora_B(self.lora_A(x))


def _apply_lora_to_resnets(resnets: list, rank: int) -> None:
    for resnet in resnets:
        resnet.conv1 = LoRAConv2d(resnet.conv1, rank=rank)
        resnet.conv2 = LoRAConv2d(resnet.conv2, rank=rank)
        if getattr(resnet, "conv_shortcut", None) is not None:
            resnet.conv_shortcut = LoRAConv2d(resnet.conv_shortcut, rank=rank)


def _get_target_resnets(unet) -> list:
    resnets = []
    resnets.extend(list(unet.down_blocks[-1].resnets))
    resnets.extend(list(unet.mid_block.resnets))
    resnets.extend(list(unet.up_blocks[0].resnets))
    return resnets


def _register_film_hooks(unet, film_generators: FiLMGeneratorSet, ctx: _FiLMContext) -> list:
    target_resnets = _get_target_resnets(unet)
    if len(target_resnets) != len(film_generators.heads):
        raise RuntimeError(
            f"FiLM head count mismatch: {len(target_resnets)} target ResNet blocks "
            f"but {len(film_generators.heads)} FiLM heads."
        )
    hooks = []
    for resnet, head in zip(target_resnets, film_generators.heads):
        def _make_hook(h):
            def _hook(module, inp, out):
                if ctx.embedding is None:
                    return out
                emb = ctx.embedding
                B_out = out.shape[0]
                B_emb = emb.shape[0]
                if B_out != B_emb:
                    emb = emb.repeat(B_out // B_emb, 1)
                # Head weights are float32; run it in float32 and cast outputs back
                gamma, beta = h(emb.to(device=out.device, dtype=torch.float32))
                gamma = gamma.to(dtype=out.dtype)
                beta = beta.to(dtype=out.dtype)
                return gamma[:, :, None, None] * out + beta[:, :, None, None]
            return _hook
        hooks.append(resnet.norm2.register_forward_hook(_make_hook(head)))
    return hooks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_densepose_predictor(device: str, model_path: str):
    import apply_net
    from detectron2.engine.defaults import DefaultPredictor
    args = apply_net.create_argument_parser().parse_args([
        "show",
        os.path.join(_REPO_ROOT, "configs", "densepose_rcnn_R_50_FPN_s1x.yaml"),
        model_path,
        "dp_segm", "-v",
        "--opts", "MODEL.DEVICE", device,
    ])
    cfg = apply_net.ShowAction.setup_config(args.cfg, args.model, args, [])
    predictor = DefaultPredictor(cfg)
    context = apply_net.ShowAction.create_context(args, cfg)
    return predictor, context


def _expand_conv_in_13ch(unet):
    """Expand unet.conv_in from 9 → 13 input channels (new channels zero-initialised)."""
    old = unet.conv_in
    conv_new = torch.nn.Conv2d(
        in_channels=13,
        out_channels=old.out_channels,
        kernel_size=old.kernel_size,
        padding=old.padding,
    )
    torch.nn.init.zeros_(conv_new.weight)
    conv_new.weight.data[:, :old.in_channels] = old.weight.data
    conv_new.bias.data = old.bias.data
    unet.conv_in = conv_new
    unet.config["in_channels"] = 13
    unet.config.in_channels = 13
    return unet


def _pad_to_multiple(t: torch.Tensor, multiple: int = 8):
    _, _, h, w = t.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return t, (0, 0)
    return torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), mode="reflect"), (pad_h, pad_w)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="IDM-VTON FiLM inference on FIT dataset.")
    parser.add_argument(
        "--pretrained_model_name_or_path", type=str, default="yisol/IDM-VTON",
        help=(
            "Pretrained model. Must be the same base model used during training. "
            "Training uses yisol/IDM-VTON by default. "
            "The LoRA deltas are additive offsets on top of these weights."
        ),
    )
    parser.add_argument(
        "--pretrained_garmentnet_path", type=str, default=None,
        help=(
            "GarmentNet base. Defaults to --pretrained_model_name_or_path with "
            "subfolder 'unet_encoder' (correct for yisol/IDM-VTON)."
        ),
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default=None,
        help=(
            "Path to a training checkpoint directory containing unet_lora.pt, "
            "film_generators.pt, and measurement_encoder.pt. "
            "If omitted the pretrained model is used as-is (no FiLM conditioning)."
        ),
    )
    parser.add_argument(
        "--ip_adapter_path", type=str, default="ckpt/ip_adapter/ip-adapter-plus_sdxl_vit-h.bin",
        help="Path to the IP-Adapter weights. Only needed when --checkpoint_dir is set.",
    )
    parser.add_argument(
        "--image_encoder_path", type=str, default=None,
        help="Path to the CLIP image encoder. Only needed when --checkpoint_dir is set.",
    )
    parser.add_argument(
        "--lora_rank", type=int, default=4,
        help="LoRA rank used during training (must match). Only used when --checkpoint_dir is set.",
    )
    parser.add_argument("--data_dir", type=str, default="data/data_mini_test")
    parser.add_argument("--output_dir", type=str, default="result_film")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of dataset samples to process. -1 = all.")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=682)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_tokens", type=int, default=16,
                        help="Number of IP-Adapter image tokens (must match training).")
    parser.add_argument("--mixed_precision", type=str, default=None,
                        choices=["no", "fp16", "bf16"])
    parser.add_argument(
        "--densepose_path", type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpt", "densepose", "model_final_162be9.pkl"),
        help="Path to the DensePose checkpoint. Defaults to ckpt/densepose/ next to this script.",
    )
    parser.add_argument(
        "--parsing_gpu_id", type=int, default=0,
        help="GPU id passed to Parsing and OpenPose constructors.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    device = accelerator.device

    if args.seed is not None:
        set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    use_checkpoint = args.checkpoint_dir is not None

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    film_ctx = None
    film_generators = None
    measurement_encoder = None

    if not use_checkpoint:
        # Pretrained IDM-VTON: no FiLM conditioning.
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet",
            torch_dtype=weight_dtype,
        )
        unet_encoder = UNet2DConditionModel_ref.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet_encoder",
            torch_dtype=weight_dtype,
        )
        unet_encoder.config.addition_embed_type = None
        unet_encoder.config["addition_embed_type"] = None

        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="image_encoder",
            torch_dtype=weight_dtype,
        )

        pipe = TryonPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=unet,
            unet_encoder=unet_encoder,
            image_encoder=image_encoder,
            feature_extractor=CLIPImageProcessor(),
            torch_dtype=weight_dtype,
            add_watermarker=False,
            safety_checker=None,
        ).to(device)

    else:
        # ── Checkpoint mode ───────────────────────────────────────────────────
        garmentnet_path = args.pretrained_garmentnet_path or args.pretrained_model_name_or_path
        garmentnet_subfolder = "unet" if args.pretrained_garmentnet_path else "unet_encoder"

        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False,
        )
        tokenizer_2 = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer_2", use_fast=False,
        )
        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder",
        ).to(device, dtype=weight_dtype)
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder_2",
        ).to(device, dtype=weight_dtype)
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae",
            torch_dtype=weight_dtype,
        ).to(device)

        if args.image_encoder_path:
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                args.image_encoder_path,
            ).to(device, dtype=weight_dtype)
        else:
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="image_encoder",
            ).to(device, dtype=weight_dtype)

        unet_encoder = UNet2DConditionModel_ref.from_pretrained(
            garmentnet_path, subfolder=garmentnet_subfolder,
        ).to(device, dtype=weight_dtype)
        unet_encoder.config.addition_embed_type = None
        unet_encoder.config["addition_embed_type"] = None
        unet_encoder.requires_grad_(False)

        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet",
            low_cpu_mem_usage=False, device_map=None,
        )

        # Load IP-Adapter weights
        ip_bin = args.ip_adapter_path or os.path.join(
            args.pretrained_model_name_or_path, "ip_adapter", "ip-adapter-plus_sdxl_vit-h.bin"
        )
        state_dict = torch.load(ip_bin, map_location="cpu")
        adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
        adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

        image_proj_model = Resampler(
            dim=image_encoder.config.hidden_size,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=args.num_tokens,
            embedding_dim=image_encoder.config.hidden_size,
            output_dim=unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(device, dtype=torch.float32)
        image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        unet.encoder_hid_proj = image_proj_model

        if unet.conv_in.in_channels == 9:
            unet = _expand_conv_in_13ch(unet)

        cross_attn_dim = unet.config.cross_attention_dim

        # Derive FiLM channel dims before LoRA wrapping (norm2 is unchanged by LoRA)
        target_resnets_raw = _get_target_resnets(unet)
        film_channel_dims = [r.norm2.num_channels for r in target_resnets_raw]

        # Apply LoRA wrapping to target ResNet blocks
        if args.lora_rank > 0:
            _apply_lora_to_resnets(_get_target_resnets(unet), rank=args.lora_rank)

        # Load trainable UNet params (LoRA A/B + norms + time_emb_proj + conv_in)
        unet_lora_state = torch.load(
            os.path.join(args.checkpoint_dir, "unet_lora.pt"), map_location="cpu"
        )
        unet.load_state_dict(unet_lora_state, strict=False)

        unet = unet.to(device, dtype=weight_dtype)
        unet.requires_grad_(False)

        # Build and load FiLMGeneratorSet
        film_generators = FiLMGeneratorSet(
            input_dim=cross_attn_dim, channel_dims=film_channel_dims
        ).to(device)
        film_state = torch.load(
            os.path.join(args.checkpoint_dir, "film_generators.pt"), map_location="cpu"
        )
        film_generators.load_state_dict(film_state)
        film_generators.eval()

        # Build and load MeasurementEncoder
        measurement_encoder = MeasurementEncoder(
            num_measurements=9,
            hidden_dim=256,
            output_dim=cross_attn_dim,
            dropout=0.1,
            use_fourier=False,
        ).to(device)
        menc_state = torch.load(
            os.path.join(args.checkpoint_dir, "measurement_encoder.pt"), map_location="cpu"
        )
        measurement_encoder.load_state_dict(menc_state)
        measurement_encoder.eval()

        # Register FiLM hooks (once; updated via film_ctx.embedding per batch)
        film_ctx = _FiLMContext()
        _register_film_hooks(unet, film_generators, film_ctx)

        pipe = TryonPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=unet,
            unet_encoder=unet_encoder,
            vae=vae,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            image_encoder=image_encoder,
            feature_extractor=CLIPImageProcessor(),
            torch_dtype=weight_dtype,
            add_watermarker=False,
            safety_checker=None,
        ).to(device)

    pipe.set_progress_bar_config(disable=False)

    def _align(v: int, m: int = 8) -> int:
        return v + (m - v % m) % m

    pad_h = _align(args.height) - args.height
    pad_w = _align(args.width) - args.width
    pipe_height = args.height + pad_h
    pipe_width = args.width + pad_w

    # ── Preprocessing models ──────────────────────────────────────────────────
    from preprocess.humanparsing.run_parsing import Parsing
    from preprocess.openpose.run_openpose import OpenPose
    import apply_net
    from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation

    parsing_model = Parsing(args.parsing_gpu_id)
    openpose_model = OpenPose(args.parsing_gpu_id)
    openpose_model.preprocessor.body_estimation.model.to(device)

    densepose_predictor, densepose_context = _build_densepose_predictor(str(device), args.densepose_path)

    def densepose_fn(pil_image):
        img_bgr = convert_PIL_to_numpy(_apply_exif_orientation(pil_image), format="BGR")
        with torch.no_grad():
            instances = densepose_predictor(img_bgr)["instances"]
            vis = apply_net.ShowAction.execute_on_outputs(
                densepose_context, {"image": img_bgr}, instances
            )
        from PIL import Image as _Image
        return _Image.fromarray(vis[:, :, ::-1])

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = FITDatasetWithMeasurements(
        data_root=args.data_dir,
        phase="inference",
        size=(args.height, args.width),
        parsing_model=parsing_model,
        openpose_model=openpose_model,
        densepose_fn=densepose_fn,
    )
    if args.num_samples > 0:
        indices = list(range(min(args.num_samples, len(dataset))))
        dataset = torch.utils.data.Subset(dataset, indices)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

    # ── Inference loop ────────────────────────────────────────────────────────
    for batch in dataloader:
        person_images = batch["person_image"].to(device, dtype=weight_dtype)
        garment_images = batch["garment_image"].to(device, dtype=weight_dtype)
        garment_clips = batch["garment_image_clip"]
        pose_imgs = batch["pose"].to(device, dtype=weight_dtype)
        masks = batch["mask"].to(device, dtype=weight_dtype)
        prompts = batch["text_prompts"]
        prompts_cloth = batch["text_prompts_cloth"]
        person_filenames = batch["person_filename"]

        bsz = person_images.shape[0]

        img_emb_list = [garment_clips[i] for i in range(bsz)]
        image_embeds = torch.cat(img_emb_list, dim=0).to(device, dtype=weight_dtype)

        if not isinstance(prompts, list):
            prompts = list(prompts)
        if not isinstance(prompts_cloth, list):
            prompts_cloth = list(prompts_cloth)
        negative_prompts = [negative_prompt] * bsz

        with torch.no_grad():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = pipe.encode_prompt(
                prompts,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompts,
            )
            (prompt_embeds_c, _, _, _) = pipe.encode_prompt(
                prompts_cloth,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
                negative_prompt=negative_prompts,
            )

            # Set FiLM context: measurements → [B, 2048] embedding for hook modulation.
            # During CFG the UNet receives 2B inputs; the hook handles batch mismatch
            # automatically by repeating the embedding.
            if film_ctx is not None:
                measurements = batch["measurements"].to(device, dtype=torch.float32)
                meas_emb = measurement_encoder(
                    measurements, measurement_dropout_prob=0.0
                ).squeeze(1)  # [B, cross_attn_dim]
                film_ctx.embedding = meas_emb.to(dtype=torch.float32)

            generator = torch.Generator(device).manual_seed(args.seed) if args.seed else None

            person_input = (person_images + 1.0) / 2.0
            if pad_h > 0 or pad_w > 0:
                person_input, _ = _pad_to_multiple(person_input)
                pose_imgs_in, _ = _pad_to_multiple(pose_imgs)
                masks_in, _ = _pad_to_multiple(masks)
                garment_images_in, _ = _pad_to_multiple(garment_images)
            else:
                pose_imgs_in = pose_imgs
                masks_in = masks
                garment_images_in = garment_images

            images = pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                strength=1.0,
                pose_img=pose_imgs_in,
                text_embeds_cloth=prompt_embeds_c,
                cloth=garment_images_in,
                mask_image=masks_in,
                image=person_input,
                height=pipe_height,
                width=pipe_width,
                guidance_scale=args.guidance_scale,
                ip_adapter_image=image_embeds,
            )[0]

            if film_ctx is not None:
                film_ctx.embedding = None

            if pad_h > 0 or pad_w > 0:
                images = [img.crop((0, 0, args.width, args.height)) for img in images]

        crop_boxes = batch["person_crop_box"]
        input_person_filenames = batch["input_person_filename"]

        for i, (img, fname) in enumerate(zip(images, person_filenames)):
            stem = os.path.splitext(os.path.basename(fname))[0]
            out_path = os.path.join(args.output_dir, f"{stem}_result.png")

            left, top, right, bottom = (int(crop_boxes[k][i]) for k in range(4))
            crop_w = right - left
            crop_h = bottom - top

            orig_person = Image.open(
                os.path.join(args.data_dir, input_person_filenames[i])
            ).convert("RGB")
            result_resized = img.resize((crop_w, crop_h), Image.LANCZOS)

            res_path = os.path.join(args.output_dir, f"{stem}_generated.png")
            result_resized.save(res_path)
            print(f"Saved: {res_path}")

            mask_crop = Image.fromarray(
                masks[i, 0].cpu().float().mul(255).byte().numpy()
            ).resize((crop_w, crop_h), Image.NEAREST)

            inv_mask = mask_crop.convert("L").point(lambda p: 0 if p > 127 else 255)
            orig_crop = orig_person.crop((left, top, right, bottom))
            result_resized.paste(orig_crop, mask=inv_mask)
            orig_person.paste(result_resized, (left, top))
            orig_person.save(out_path)
            print(f"Saved: {out_path}")

            person_vis = ((person_images[i].cpu().float() + 1.0) / 2.0).clamp(0, 1)
            mask_vis = masks[i, 0].cpu().float()
            r = (person_vis[0] * (1 - 0.5 * mask_vis) + 0.5 * mask_vis).clamp(0, 1)
            g = (person_vis[1] * (1 - 0.5 * mask_vis)).clamp(0, 1)
            b = (person_vis[2] * (1 - 0.5 * mask_vis)).clamp(0, 1)
            vis_arr = torch.stack([r, g, b], dim=0).permute(1, 2, 0).mul(255).byte().cpu().numpy()
            mask_out = os.path.join(args.output_dir, f"{stem}_mask.png")
            Image.fromarray(vis_arr).save(mask_out)
            print(f"Saved: {mask_out}")


if __name__ == "__main__":
    main()
