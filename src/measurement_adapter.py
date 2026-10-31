"""
Measurement Adapter for IDM-VTON Extension 3.

Mirrors the garment IP-Adapter for measurements: the measurement embedding is turned into
a few tokens, and every cross-attention layer of TryonNet gets its own trainable K/V
projections for them. The resulting attention output is added as a third parallel branch,
scaled by a trainable gate:

    h = Attn(text) + s_ip * Attn(garment IP tokens) + g(block, t) * Attn(measurement tokens)

The measurement tokens are never appended to the shared context sequence. They travel
through a side channel (`MeasurementAdapterContext`), and a UNet forward pre-hook matches
them to the batch size (classifier-free guidance doubles it) and computes the gates from
the current timestep.

Baseline preservation: to_v of every block is zero-initialised, so the measurement branch
contributes exactly 0 at step 0 and the pretrained IDM-VTON behaviour is kept.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

SCALE_MODES = ("global", "block", "timestep")


class MeasurementAdapterContext:
    """Plain-Python (non-Module) holder shared by the training loop, the hook and all processors.

    `tokens` must stay set until after `accelerator.backward()`: gradient checkpointing
    recomputes the forward pass during backward and reads `active_tokens` / `gates` again.
    """

    def __init__(self, num_train_timesteps: int = 1000) -> None:
        self.num_train_timesteps = num_train_timesteps
        self.adapter: Optional["MeasurementAdapter"] = None
        self.tokens: Optional[torch.Tensor] = None         # [B, K, token_dim], set by the caller
        self.active_tokens: Optional[torch.Tensor] = None  # [B_in, K, token_dim], set by the hook
        self.gates: Optional[torch.Tensor] = None          # [B_in, n_blocks], set by the hook


class MeasurementAdapter(nn.Module):
    """Token projector + per-block K/V projections + trainable attention gates."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: List[int],
        num_tokens: int = 4,
        token_dim: int = 512,
        scale_mode: str = "block",
        num_timestep_bins: int = 10,
    ) -> None:
        """
        Args:
            input_dim:         dimension of the measurement embedding (MeasurementEncoder output)
            hidden_sizes:      inner attention dim of every target cross-attention layer
            num_tokens:        number of measurement tokens K
            token_dim:         dimension of each token (= input dim of the per-block K/V)
            scale_mode:        "global" (one gate), "block" (one gate per block) or
                               "timestep" (per block, piecewise-linear over the timestep)
            num_timestep_bins: number of gate knots over t in [0, 1] for scale_mode="timestep"
        """
        super().__init__()
        if scale_mode not in SCALE_MODES:
            raise ValueError(f"scale_mode must be one of {SCALE_MODES}, got {scale_mode!r}")
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.scale_mode = scale_mode
        n_blocks = len(hidden_sizes)

        self.token_proj = nn.Linear(input_dim, num_tokens * token_dim)
        self.token_norm = nn.LayerNorm(token_dim)

        self.to_k = nn.ModuleList([nn.Linear(token_dim, h, bias=False) for h in hidden_sizes])
        self.to_v = nn.ModuleList([nn.Linear(token_dim, h, bias=False) for h in hidden_sizes])
        for v in self.to_v:
            nn.init.zeros_(v.weight)

        rows = 1 if scale_mode == "global" else n_blocks
        bins = num_timestep_bins if scale_mode == "timestep" else 1
        self.gate_table = nn.Parameter(torch.ones(rows, bins))

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """[B, input_dim] -> [B, K, token_dim]"""
        x = self.token_proj(embedding).view(embedding.shape[0], self.num_tokens, self.token_dim)
        return self.token_norm(x)

    def compute_gates(self, t_norm: torch.Tensor) -> torch.Tensor:
        """t_norm: [B] in [0, 1] (1 = pure noise). Returns [B, n_blocks]."""
        table = self.gate_table.t()  # [bins, rows]
        pos = t_norm.clamp(0, 1) * (table.shape[0] - 1)
        lo = pos.floor().long()
        hi = (lo + 1).clamp(max=table.shape[0] - 1)
        frac = (pos - lo.float()).unsqueeze(1).to(table.dtype)
        gates = table[lo] * (1 - frac) + table[hi] * frac
        return gates.expand(-1, len(self.to_k)) if gates.shape[1] == 1 else gates


class MeasurementIPAttnProcessor2_0(nn.Module):
    """`IPAttnProcessor2_0` (text + garment IP branch) plus the measurement branch.

    Shares `to_k_ip` / `to_v_ip` with the processor it replaces, so the UNet state-dict keys
    are unchanged. The adapter is reached through the context and is deliberately not
    registered as a submodule, so it stays outside the (frozen) UNet's parameter tree.
    """

    def __init__(self, ip_processor, block_idx: int, ctx: MeasurementAdapterContext) -> None:
        super().__init__()
        self.hidden_size = ip_processor.hidden_size
        self.cross_attention_dim = ip_processor.cross_attention_dim
        self.scale = ip_processor.scale
        self.num_tokens = ip_processor.num_tokens
        self.to_k_ip = ip_processor.to_k_ip
        self.to_v_ip = ip_processor.to_v_ip
        self.block_idx = block_idx
        self._ctx = ctx

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, scale=1.0):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
            ip_hidden_states = None
        else:
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            encoder_hidden_states, ip_hidden_states = (
                encoder_hidden_states[:, :end_pos, :],
                encoder_hidden_states[:, end_pos:, :],
            )
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # branch 1: text / person attention
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # branch 2: garment IP-Adapter attention
        ip_key = self.to_k_ip(ip_hidden_states).view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        ip_value = self.to_v_ip(ip_hidden_states).view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        ip_out = F.scaled_dot_product_attention(query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False)
        ip_out = ip_out.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim).to(query.dtype)

        hidden_states = hidden_states + self.scale * ip_out

        # branch 3: measurement attention (own K/V, trainable gate)
        ctx = self._ctx
        if ctx.active_tokens is not None:
            hidden_states = hidden_states + self._measurement_branch(query, batch_size, attn.heads, head_dim, ctx)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states

    def _measurement_branch(self, query, batch_size, heads, head_dim, ctx: MeasurementAdapterContext):
        k_proj = ctx.adapter.to_k[self.block_idx]
        v_proj = ctx.adapter.to_v[self.block_idx]
        tokens = ctx.active_tokens.to(k_proj.weight.dtype)

        m_key = k_proj(tokens).view(batch_size, -1, heads, head_dim).transpose(1, 2).to(query.dtype)
        m_value = v_proj(tokens).view(batch_size, -1, heads, head_dim).transpose(1, 2).to(query.dtype)

        out = F.scaled_dot_product_attention(query, m_key, m_value, attn_mask=None, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(batch_size, -1, heads * head_dim).to(query.dtype)

        gate = ctx.gates[:, self.block_idx].view(-1, 1, 1).to(out.dtype)
        return gate * out


def get_cross_attention_processors(unet) -> List[Tuple[str, nn.Module]]:
    """Ordered (name, processor) pairs of all IP cross-attention (attn2) layers of TryonNet."""
    procs = [(n, p) for n, p in unet.attn_processors.items() if n.endswith("attn2.processor")]
    for name, p in procs:
        if not hasattr(p, "to_k_ip"):
            raise RuntimeError(
                f"{name} has no IP-Adapter K/V projections. Load the IP-Adapter weights before "
                "installing the measurement adapter."
            )
    return procs


def get_hidden_sizes(unet) -> List[int]:
    return [p.to_k_ip.out_features for _, p in get_cross_attention_processors(unet)]


def _make_pre_hook(ctx: MeasurementAdapterContext):
    def _pre_hook(module, args, kwargs):
        if ctx.tokens is None:
            ctx.active_tokens = None
            ctx.gates = None
            return None
        sample = args[0] if len(args) > 0 else kwargs["sample"]
        timestep = args[1] if len(args) > 1 else kwargs["timestep"]
        b_in = sample.shape[0]

        tokens = ctx.tokens
        if tokens.shape[0] != b_in:
            if b_in % tokens.shape[0] != 0:
                raise RuntimeError(f"Cannot match {tokens.shape[0]} measurement tokens to UNet batch {b_in}.")
            # CFG: UNet batch is [uncond; cond]; both halves get the same measurements.
            tokens = tokens.repeat(b_in // tokens.shape[0], 1, 1)

        t = torch.as_tensor(timestep, device=sample.device).float().reshape(-1)
        if t.numel() == 1:
            t = t.expand(b_in)
        ctx.active_tokens = tokens
        ctx.gates = ctx.adapter.compute_gates(t / ctx.num_train_timesteps)
        return None

    return _pre_hook


def install_measurement_adapter(unet, adapter: MeasurementAdapter, ctx: MeasurementAdapterContext):
    """Swap TryonNet's attn2 processors for the measurement-aware version and register the
    batch/timestep pre-hook. `unet` and `adapter` must be the unwrapped modules.

    Returns the hook handle (call .remove() to uninstall the hook).
    """
    procs = get_cross_attention_processors(unet)
    if len(procs) != len(adapter.to_k):
        raise RuntimeError(
            f"Adapter has {len(adapter.to_k)} blocks but TryonNet has {len(procs)} cross-attention layers."
        )
    ctx.adapter = adapter
    new_procs = dict(unet.attn_processors)
    for idx, (name, proc) in enumerate(procs):
        new_procs[name] = MeasurementIPAttnProcessor2_0(proc, idx, ctx)
    unet.set_attn_processor(new_procs)
    return unet.register_forward_pre_hook(_make_pre_hook(ctx), with_kwargs=True)
