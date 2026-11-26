"""CPU smoke test for the Measurement Adapter (Extension 3). Needs torch + diffusers only.

    python smoke_test_measurement_adapter.py

Builds a tiny UNet-like stack of diffusers Attention layers with the IP-Adapter processor and checks:
  1. at init (zero-initialised to_v) outputs equal the baseline processor exactly,
  2. after perturbing to_v the output changes and grads reach adapter, gates, encoder,
  3. CFG batch doubling and all three scale modes work,
  4. the garment IP-Adapter state-dict keys are unchanged by the swap.
"""
import torch
import torch.nn as nn
from diffusers.models.attention_processor import Attention

from ip_adapter.attention_processor import IPAttnProcessor2_0
from src.measurement_adapter import (
    MeasurementAdapter, MeasurementAdapterContext, MeasurementIPAttnProcessor2_0,
    install_measurement_adapter,
)
from src.measurement_encoder import MeasurementEncoder

HID, CTX, N_IP, N_TXT, N_BLOCKS = 32, 64, 16, 7, 3


class TinyUNet(nn.Module):
    """Mimics the parts of UNet2DConditionModel that install_measurement_adapter touches."""

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([
            Attention(query_dim=HID, cross_attention_dim=CTX, heads=4, dim_head=8,
                      processor=IPAttnProcessor2_0(HID, CTX, num_tokens=N_IP))
            for _ in range(N_BLOCKS)
        ])

    @property
    def attn_processors(self):
        return {f"blocks.{i}.attn2.processor": b.processor for i, b in enumerate(self.blocks)}

    def set_attn_processor(self, procs):
        for i, b in enumerate(self.blocks):
            b.set_processor(procs[f"blocks.{i}.attn2.processor"])

    def forward(self, sample, timestep, encoder_hidden_states):
        h = sample
        for b in self.blocks:
            h = h + b(h, encoder_hidden_states=encoder_hidden_states)
        return h


def run(scale_mode):
    torch.manual_seed(0)
    unet = TinyUNet().requires_grad_(False)
    keys_before = set(unet.state_dict().keys())
    x = torch.randn(2, 10, HID)
    ctx_states = torch.randn(2, N_TXT + N_IP, CTX)
    t = torch.tensor([10, 900])
    base = unet(x, t, ctx_states)

    enc = MeasurementEncoder(num_measurements=9, hidden_dim=64, output_dim=48, use_fourier=True, fourier_scale=0.25)
    adapter = MeasurementAdapter(48, [HID] * N_BLOCKS, num_tokens=4, token_dim=16,
                                 scale_mode=scale_mode, num_timestep_bins=5)
    ctx = MeasurementAdapterContext()
    install_measurement_adapter(unet, adapter, ctx)
    assert all(isinstance(b.processor, MeasurementIPAttnProcessor2_0) for b in unet.blocks)
    assert set(unet.state_dict().keys()) == keys_before, "IP-Adapter state-dict keys changed"

    meas = torch.randn(2, 9)
    ctx.tokens = adapter(enc(meas).squeeze(1))
    out0 = unet(x, t, ctx_states)
    assert torch.allclose(out0, base, atol=1e-6), f"[{scale_mode}] not baseline-preserving at init"

    for v in adapter.to_v:
        nn.init.normal_(v.weight, std=0.1)
    ctx.tokens = adapter(enc(meas).squeeze(1))
    out1 = unet(x, t, ctx_states)
    assert (out1 - base).abs().max() > 1e-4, f"[{scale_mode}] measurement branch has no effect"
    out1.sum().backward()
    for name, p in [("gate", adapter.gate_table), ("to_k", adapter.to_k[0].weight),
                    ("to_v", adapter.to_v[0].weight), ("token_proj", adapter.token_proj.weight),
                    ("encoder", enc.encoder[0].weight)]:
        assert p.grad is not None and p.grad.abs().sum() > 0, f"[{scale_mode}] no grad on {name}"

    # CFG: UNet batch is 2x the token batch
    ctx.tokens = adapter(enc(meas[:1]).squeeze(1))
    unet(x, t, ctx_states)  # UNet batch 2 vs 1 token set -> repeated like CFG
    ctx.tokens = None
    assert torch.allclose(unet(x, t, ctx_states), base, atol=1e-6), "tokens=None must give the baseline"
    print(f"[{scale_mode}] OK")


if __name__ == "__main__":
    for mode in ("global", "block", "timestep"):
        run(mode)
    print("All checks passed.")
