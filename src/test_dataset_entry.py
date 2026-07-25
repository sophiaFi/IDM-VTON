import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from fit_dataset import FITDatasetWithMeasurements


DATA_ROOT = "../data_mini_test"


dataset = FITDatasetWithMeasurements(
    data_root=DATA_ROOT,
    phase="test",
    size=(682,512),
)


print("Dataset size:", len(dataset))


# Test first sample
idx = 0

record = dataset.records[idx]

print("\nJSON entry:")
for k,v in record.items():
    print(f"{k}: {v}")


sample = dataset[idx]


print("\nReturned fields:")
for k,v in sample.items():

    if isinstance(v, torch.Tensor):
        print(
            f"{k:25s}",
            "shape=", tuple(v.shape),
            "range=",
            (float(v.min()), float(v.max()))
        )

    else:
        print(
            f"{k:25s}",
            v
        )


# ---- sanity checks ----

assert sample["person_image"].shape == (3,682,512)

assert sample["garment_image"].shape == (3,682,512)

assert sample["pose"].shape == (3,682,512)

assert sample["mask"].shape == (1,682,512)

assert sample["masked_person"].shape == (3,682,512)

assert sample["garment_image_clip"].shape == (1,3,224,224)

assert sample["measurements"].shape == (7,)


print("\nAll tensor checks passed!")


# ---- visualize ----

def show(img, title):

    img = img.detach().cpu()

    if img.ndim == 3:
        img = img.permute(1,2,0)

    img = img.numpy()

    if img.min() < 0:
        img = (img+1)/2

    plt.imshow(img)
    plt.title(title)
    plt.axis("off")


plt.figure(figsize=(15,8))

items = [
    ("person",
     sample["person_image"]),

    ("garment",
     sample["garment_image"]),

    ("pose",
     sample["pose"]),

    ("mask",
     sample["mask"]),

    ("masked person",
     sample["masked_person"]),
]


for i,(name,img) in enumerate(items):

    plt.subplot(2,3,i+1)
    show(img,name)


plt.tight_layout()
plt.show()
