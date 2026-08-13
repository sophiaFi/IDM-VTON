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
        num_measurements=7,      # raw inputs: 4 body + 3 garment (2 ease diffs appended in forward)
        hidden_dim=256,
        output_dim=768,          # Match CLIP embedding dim
        dropout=0.1,
        use_fourier=False        # Optional: Fourier Features (similar to FIT)) #TODO: remove?
    ):
        super().__init__()

        self.use_fourier = use_fourier
        mlp_input_dim = num_measurements + 2  # +2 for bust_ease and hem_drop computed in forward()

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
            measurements: [B, 7] - normalized measurements
                [body_bust, body_height, body_hips, body_waist,
                 garment_bust, garment_length, garment_sleeve_length]
            measurement_dropout_prob: float - probability to drop measurements

        Returns:
            embeddings: [B, 1, output_dim] - measurement token for Cross-Attention
        """
        # Measurement dropout for robustness
        if self.training and measurement_dropout_prob > 0:
            mask = torch.rand_like(measurements) > measurement_dropout_prob
            measurements = measurements * mask

        # Ease differences: garment minus body for paired dimensions.
        # sleeve_length has no body counterpart so is excluded.
        bust_ease = measurements[:, 4] - measurements[:, 0]  # garment_bust - body_bust
        hem_drop  = measurements[:, 1] - measurements[:, 5]  # body_height - garment_length

        features = torch.cat([
            measurements,
            bust_ease.unsqueeze(1),
            hem_drop.unsqueeze(1),
        ], dim=1)  # [B, 9]
        
        if self.use_fourier:
            features = self.fourier(features)
        
        # Encode
        embeddings = self.encoder(features)
        
        # Add sequence dimension for cross attention: [B, output_dim] -> [B, 1, output_dim]
        embeddings = embeddings.unsqueeze(1)
        
        return embeddings


class FourierFeatureProjection(nn.Module):
    """Optional: Fourier Features as in FIT paper"""
    def __init__(self, input_dim, output_dim, scale=1.0):
        super().__init__()
        self.register_buffer('weight', torch.randn(input_dim, output_dim // 2) * scale)
    
    def forward(self, x):
        x_proj = 2 * torch.pi * x @ self.weight
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


def normalize_measurements(measurements_dict):
    """
    Normalize measurements for model input
    
    Args:
        measurements_dict: dict with keys:
            - body_bust, body_height, body_hips, body_waist
            - garment_bust, garment_length, garment_sleeve_length
    
    Returns:
        torch.Tensor [7]: normalized measurements
    """
    # Constants for normalization (calculated from FIT dataset statistics)
    MEAN = torch.tensor([
        105.534,  # body_bust (cm)
        171.581, # body_height (cm)
        107.012,  # body_hips (cm)
        92.043,  # body_waist (cm)
        115.716,  # garment_bust (cm)
        53.553,  # garment_length (cm)
        29.687,  # garment_sleeve_length (cm)
    ])
    
    STD = torch.tensor([
        11.761,  # body_bust
        9.497,  # body_height
        10.762,  # body_hips
        15.063,  # body_waist
        13.396,  # garment_bust
        9.575,  # garment_length
        17.973,  # garment_sleeve_length
    ])
    
    # Extract measurements
    measurements = torch.tensor([
        measurements_dict['body_bust'],
        measurements_dict['body_height'],
        measurements_dict['body_hips'],
        measurements_dict['body_waist'],
        measurements_dict['garment_bust'],
        measurements_dict['garment_length'],
        measurements_dict['garment_sleeve_length'],
    ])
    
    # Normalize: (x - mean) / std
    normalized = (measurements - MEAN) / STD
    
    return normalized


# Test
if __name__ == "__main__":
    encoder = MeasurementEncoder()
    
    # Dummy measurements
    batch_size = 4
    measurements = torch.randn(batch_size, 7)  # raw 7-dim input; encoder adds bust_ease internally
    
    # Forward pass
    embeddings = encoder(measurements)
    print(f"Input shape: {measurements.shape}")
    print(f"Output shape: {embeddings.shape}")  # [4, 1, 768]
    
    # Check parameter count
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"Total parameters: {total_params:,}")  # ~5-10M
