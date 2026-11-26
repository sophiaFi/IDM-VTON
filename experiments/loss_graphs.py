import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ea = EventAccumulator("./tb_logs/")
ea.Reload()

scalars = ["train_loss", "train_fit_loss", "menc_grad_norm", "gpu_alloc_gb", "gpu_reserved_gb"]

fig, axes = plt.subplots(len(scalars), 1, figsize=(10, 3 * len(scalars)))
for ax, tag in zip(axes, scalars):
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    ax.plot(steps, values)
    ax.set_title(tag)
    ax.set_xlabel("step")

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()