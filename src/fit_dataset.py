"""
FIT Dataset with Body and Garment Measurements for IDM-VTON training.
"""

import json
import os
import random
from typing import Literal, Tuple

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
from transformers import CLIPImageProcessor

from measurement_encoder import normalize_measurements


class FITDatasetWithMeasurements(data.Dataset):
    """
    Dataset for the FIT dataset with body and garment measurements.

    Expected directory structure:
        data_root/
          cloth/              # garment images
          person/             # person in arbitrary garment (not directly loaded)
          target/             # ground truth: person wearing the cloth from cloth/
          image-densepose/    # densepose maps for target images (precomputed)
          agnostic-mask/      # agnostic masks for target images (precomputed)
          measurements.json   # list of records (see below)

    measurements.json format:
        [
          {
            "id": 0,
            "cloth": "cloth/00000.png",
            "target": "target/00000.png",
            "person": "person/00000.png",
            "body_bust": 108.177001953125,
            "body_height": 174.3249969482422,
            "body_hips": 109.65899658203125,
            "body_waist": 96.4999008178711,
            "garment_bust": 124.40354919433594,
            "garment_length": 54.28818130493164,
            "garment_sleeve_length": 17.44197654724121
          },
          ...
        ]

    Returned dict keys:
        person_image      [3, H, W]       [-1, 1]        target/ image (diffusion ground truth)
        garment_image     [3, H, W]       [-1, 1]        cloth/ image for GarmentNet
        garment_image_clip [1, 3, 224, 224] CLIP range  cloth/ image for IP-Adapter
        pose              [3, H, W]       [-1, 1]        precomputed densepose map
        mask              [1, H, W]       {0, 1}         inpaint mask (1 = garment region)
        masked_person     [3, H, W]       [-1, 1]        person_image with garment zeroed out
        text_prompts      str             —              person caption
        text_prompts_cloth str            —              garment caption
        measurements      [7]             z-scored       body + garment measurements
        person_filename   str             —              target filename (for saving/debugging)
        cloth_filename    str             —              cloth filename (for saving/debugging)
    """

    MEASUREMENT_KEYS = [
        "body_bust", "body_height", "body_hips", "body_waist",
        "garment_bust", "garment_length", "garment_sleeve_length",
    ]

    def __init__(
        self,
        data_root: str,
        phase: Literal["train", "test"] = "train",
        size: Tuple[int, int] = (512, 682),
    ):
        """
        Args:
            data_root: Path to the fit-mini dataset root.
            phase:     "train" enables augmentations; "test" disables them.
            size:      (height, width) of loaded images.
        """
        super().__init__()
        self.data_root = data_root
        self.phase = phase
        self.height, self.width = size

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.to_tensor = transforms.ToTensor()
        self.norm = transforms.Normalize([0.5], [0.5])
        self.clip_processor = CLIPImageProcessor()
        self.flip_transform = transforms.RandomHorizontalFlip(p=1)

        with open(os.path.join(data_root, "measurements.json"), "r") as f:
            self.records = json.load(f)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        id = record["id"]
        target_name = record["target"]
        cloth_name = record["cloth"]

        # Load images
        cloth_pil = Image.open(
            os.path.join(self.data_root, cloth_name)
        ).convert("RGB")

        target_pil = Image.open(
            os.path.join(self.data_root, target_name)
        ).convert("RGB").resize((self.width, self.height))

        target_stem = os.path.splitext(os.path.basename(target_name))[0]
        mask_pil = Image.open(
            os.path.join(self.data_root, "agnostic-mask", f"{target_stem}.png")
        ).resize((self.width, self.height))

        densepose_pil = Image.open(
            os.path.join(self.data_root, "image-densepose", f"{target_stem}.png")
        ).convert("RGB").resize((self.width, self.height))

        # Initial tensor conversion
        person_image = self.transform(target_pil)   # [3, H, W], [-1, 1]
        cloth_pil = cloth_pil.resize(
            (self.width,self.height)
        )
        mask = self.to_tensor(mask_pil)[:1]          # [1, H, W], [0, 1]
        pose = self.to_tensor(densepose_pil)         # [3, H, W], [0, 1]

        # Training augmentations
        if self.phase == "train":
            # Horizontal flip: applied consistently across cloth (PIL), person, mask, pose
            if random.random() > 0.5:
                cloth_pil = self.flip_transform(cloth_pil)
                person_image = self.flip_transform(person_image)
                mask = self.flip_transform(mask)
                pose = self.flip_transform(pose)

            #TODO: augmentation

        # Garment tensors — built after all cloth_pil augmentations are done
        garment_image_clip = self.clip_processor(
            images=cloth_pil, return_tensors="pt"
        ).pixel_values                              # [1, 3, 224, 224]
        garment_image = self.transform(cloth_pil)   # [3, H, W], [-1, 1]

        # Finalize mask and pose
        mask = mask.clamp(0, 1)
        mask[mask < 0.5] = 0.0
        mask[mask >= 0.5] = 1.0
        pose = self.norm(pose)                      # [0, 1] -> [-1, 1]

        # Zero out the garment region from the person image
        masked_person = person_image * (1.0 - mask)

        # Z-score normalize measurements using precomputed dataset statistics
        measurement_dict = {k: record[k] for k in self.MEASUREMENT_KEYS}
        measurements = normalize_measurements(measurement_dict)  # [7]

        return {
            "person_image": person_image,
            "garment_image": garment_image,
            "garment_image_clip": garment_image_clip,
            "pose": pose,
            "mask": mask,
            "masked_person": masked_person,
            "text_prompts": "model is wearing a garment", #TODO
            "text_prompts_cloth": "a photo of a garment", #TODO
            "measurements": measurements,
            "person_filename": target_name,
            "cloth_filename": cloth_name,
        }


if __name__ == "__main__":
    import sys
    import tempfile
    import numpy as np
    from torch.utils.data import DataLoader

    # Ensure repo root is on path when running this file directly
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    with tempfile.TemporaryDirectory() as tmp:
        # Build a minimal dummy dataset on disk
        for subdir in ["cloth", "target", "image-densepose", "agnostic-mask"]:
            os.makedirs(os.path.join(tmp, subdir))

        dummy_rgb = Image.fromarray(
            np.random.randint(0, 255, (256, 192, 3), dtype=np.uint8)
        )
        dummy_gray = Image.fromarray(
            np.random.randint(0, 255, (256, 192), dtype=np.uint8)
        )

        records = []
        for i in range(4):
            image_name = f"{i:04d}.png"
            dummy_rgb.save(os.path.join(tmp, "cloth", image_name))
            dummy_rgb.save(os.path.join(tmp, "target", image_name))
            dummy_rgb.save(os.path.join(tmp, "image-densepose", image_name))
            dummy_gray.save(
                os.path.join(tmp, "agnostic-mask", image_name)
            )
            records.append({
                "id": i,
                "cloth": f"cloth/{image_name}",
                "target": f"target/{image_name}",
                "person": f"person/{image_name}",
                "body_bust": 95.0 + i,
                "body_height": 170.0,
                "body_hips": 100.0,
                "body_waist": 80.0,
                "garment_bust": 105.0,
                "garment_length": 55.0,
                "garment_sleeve_length": 30.0,
            })

        with open(os.path.join(tmp, "measurements.json"), "w") as f:
            json.dump(records, f)

        dataset = FITDatasetWithMeasurements(tmp, phase="train", size=(512, 384))
        print(f"Dataset size: {len(dataset)}")

        item = dataset[0]
        print("\nItem keys and shapes:")
        for k, v in item.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: shape={tuple(v.shape)}, range=[{v.min():.2f}, {v.max():.2f}]")
            else:
                print(f"  {k}: {v!r}")

        loader = DataLoader(dataset, batch_size=2, shuffle=True)
        batch = next(iter(loader))
        print(f"\nDataLoader batch:")
        print(f"  person_image:      {tuple(batch['person_image'].shape)}")
        print(f"  garment_image:     {tuple(batch['garment_image'].shape)}")
        print(f"  garment_image_clip:{tuple(batch['garment_image_clip'].shape)}")
        print(f"  pose:              {tuple(batch['pose'].shape)}")
        print(f"  mask:              {tuple(batch['mask'].shape)}")
        print(f"  masked_person:     {tuple(batch['masked_person'].shape)}")
        print(f"  measurements:      {tuple(batch['measurements'].shape)}")
        print("\nAll checks passed.")
