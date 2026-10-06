"""
Measurement Encoder for extending IDM-VTON
"""

import torch
import torch.nn as nn

class MeasurementEncoder(nn.Module):
    """
    Encoder for body/garment measurements + pairwise ease differences.
    Output: Embedding for cross attention in UNet
    """
    def __init__(
        self,
        num_measurements=9,      # 7 base measurements + 2 ease features (computed in normalize_measurements)
        hidden_dim=256,
        output_dim=2048,         # SDXL cross-attention dim (CLIP-L 768 + OpenCLIP 1280)
        dropout=0.1,
        use_fourier=False        # Optional: Fourier Features (similar to FIT)) #TODO: remove?
    ):
        super().__init__()

        self.use_fourier = use_fourier
        mlp_input_dim = num_measurements

        if use_fourier:
            self.fourier = FourierFeatureProjection(mlp_input_dim, hidden_dim)
            input_dim = hidden_dim
        else:
            input_dim = mlp_input_dim
        
        # MLP Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, output_dim),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, measurements, measurement_dropout_prob=0.0):
        """
        Args:
            measurements: [B, 9] - normalized measurements from normalize_measurements()
                [body_bust, body_height, body_hips, body_waist,
                 garment_bust, garment_length, garment_sleeve_length,
                 bust_ease, hem_drop]
            measurement_dropout_prob: float - probability to drop measurements

        Returns:
            embeddings: [B, 1, output_dim] - measurement token for Cross-Attention
        """
        if self.training and measurement_dropout_prob > 0:
            mask = torch.rand_like(measurements) > measurement_dropout_prob
            measurements = measurements * mask

        features = self.fourier(measurements) if self.use_fourier else measurements

        embeddings = self.encoder(features)
        return embeddings.unsqueeze(1)


class FourierFeatureProjection(nn.Module):
    """Optional: Fourier Features as in FIT paper"""
    def __init__(self, input_dim, output_dim, scale=1.0):
        super().__init__()
        self.register_buffer('weight', torch.randn(input_dim, output_dim // 2) * scale)
    
    def forward(self, x):
        x_proj = 2 * torch.pi * x @ self.weight
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


# FIT dataset train-split statistics (cm).  Indices 0-6 are the seven base measurements;
# indices 7-8 are derived ease features computed in normalize_measurements().
# bust_ease and hem_drop stats are estimated from the marginal distributions —
# recompute from the actual train split if precision matters.
_MEASURE_MEAN = torch.tensor([
    105.118,  # body_bust
    172.126,  # body_height
    106.641,  # body_hips
     91.401,  # body_waist
    114.514,  # garment_bust
     53.868,  # garment_length
     30.192,  # garment_sleeve_length
      9.396,  # bust_ease  = garment_bust - body_bust  (estimated)
    118.258,  # hem_drop   = body_height - garment_length  (estimated)
])

_MEASURE_STD = torch.tensor([
    10.726,   # body_bust
     8.791,   # body_height
     9.568,   # body_hips
    13.970,   # body_waist
    13.072,   # garment_bust
     8.594,   # garment_length
    18.197,   # garment_sleeve_length
    10.0,     # bust_ease  (estimated)
    12.3,     # hem_drop   (estimated)
])


def normalize_measurements(measurements_dict):
    """
    Normalize measurements for model input.

    Args:
        measurements_dict: dict with keys:
            body_bust, body_height, body_hips, body_waist,
            garment_bust, garment_length, garment_sleeve_length

    Returns:
        torch.Tensor [9]: z-scored measurements (7 base + bust_ease + hem_drop)
    """
    body_bust      = measurements_dict['body_bust']
    body_height    = measurements_dict['body_height']
    body_hips      = measurements_dict['body_hips']
    body_waist     = measurements_dict['body_waist']
    garment_bust   = measurements_dict['garment_bust']
    garment_length = measurements_dict['garment_length']
    garment_sleeve = measurements_dict['garment_sleeve_length']

    # Derived features computed in raw cm space so their differences are meaningful.
    bust_ease = garment_bust  - body_bust    # positive → garment is roomier than body
    hem_drop  = body_height   - garment_length  # how far the body extends below the hem

    raw = torch.tensor([
        body_bust, body_height, body_hips, body_waist,
        garment_bust, garment_length, garment_sleeve,
        bust_ease, hem_drop,
    ])
    return (raw - _MEASURE_MEAN) / _MEASURE_STD


# Test
if __name__ == "__main__":
    encoder = MeasurementEncoder()

    # Dummy measurements (9-dim: 7 base + bust_ease + hem_drop from normalize_measurements)
    batch_size = 4
    measurements = torch.randn(batch_size, 9)
    
    # Forward pass
    embeddings = encoder(measurements)
    print(f"Input shape: {measurements.shape}")
    print(f"Output shape: {embeddings.shape}")  # [4, 1, 768]
    
    # Check parameter count
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"Total parameters: {total_params:,}")  # ~5-10M
