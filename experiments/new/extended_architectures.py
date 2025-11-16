"""
Extended Architecture Suite for TAP Experiments

This extends the original framework with:
1. Larger ResNet variants (ResNet-50, ResNet-101)
2. Large Transformers (GPT-style, BERT-style)
3. Vision Transformers (ViT variants)
4. Hybrid architectures

Ready to run once PyTorch is installed.
"""

import torch
import torch.nn as nn
from typing import Dict, List
import numpy as np


# ============================================================================
# LARGE RESNET ARCHITECTURES
# ============================================================================

class ResNetBlock(nn.Module):
    """Residual block for ResNet."""

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class LargeResNet(nn.Module):
    """Larger ResNet variants (ResNet-50, ResNet-101 style)."""

    def __init__(self, num_classes=10, depth='50'):
        super().__init__()
        self.depth = depth
        self.in_channels = 64

        # Initial conv (adapted for 32x32 images)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # ResNet layers
        if depth == '50':
            layers = [3, 4, 6, 3]  # ResNet-50 configuration
        elif depth == '101':
            layers = [3, 4, 23, 3]  # ResNet-101 configuration
        else:
            layers = [2, 2, 2, 2]  # ResNet-18 (default)

        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = []
        layers.append(ResNetBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(ResNetBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def get_architecture_params(self) -> Dict:
        total_params = sum(p.numel() for p in self.parameters())
        depth_map = {'18': 18, '50': 50, '101': 101}
        return {
            'depth': depth_map.get(self.depth, 18),
            'width': 256,  # Average channel width
            'min_width': 64,
            'max_width': 512,
            'num_params': total_params,
            'connectivity': total_params / (3 + 512)
        }


# ============================================================================
# LARGE TRANSFORMER ARCHITECTURES
# ============================================================================

class LargeTransformer(nn.Module):
    """Large Transformer for classification (GPT/BERT style)."""

    def __init__(self, input_dim: int, d_model: int = 512, nhead: int = 8,
                 num_layers: int = 12, num_classes: int = 10,
                 seq_len: int = 64, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.seq_len = seq_len

        # Input projection
        self.input_proj = nn.Linear(input_dim // seq_len, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, d_model))
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Classification head
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.seq_len, -1)

        # Project and add positional encoding
        x = self.input_proj(x)
        x = x + self.pos_encoder
        x = self.dropout(x)

        # Transformer
        x = self.transformer(x)

        # Global average pooling
        x = x.mean(dim=1)
        x = self.norm(x)

        # Classification
        x = self.fc(x)

        return x

    def get_architecture_params(self) -> Dict:
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'depth': self.num_layers,
            'width': self.d_model,
            'min_width': self.d_model,
            'max_width': self.d_model * 4,
            'num_params': total_params,
            'connectivity': total_params / (self.d_model * self.num_layers)
        }


# ============================================================================
# VISION TRANSFORMER (VIT)
# ============================================================================

class VisionTransformer(nn.Module):
    """Vision Transformer for image classification."""

    def __init__(self, img_size=32, patch_size=4, in_channels=3,
                 d_model=384, num_layers=12, nhead=6, num_classes=10,
                 mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.d_model = d_model
        self.num_layers = num_layers

        # Patch embedding
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_channels, d_model,
                                     kernel_size=patch_size, stride=patch_size)

        # Class token and position embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, d_model))
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * mlp_ratio,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Classification head
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        batch_size = x.size(0)

        # Patch embedding
        x = self.patch_embed(x)  # (B, d_model, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, d_model)

        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add position embedding
        x = x + self.pos_embed
        x = self.dropout(x)

        # Transformer
        x = self.transformer(x)

        # Classification (use class token)
        x = self.norm(x[:, 0])
        x = self.head(x)

        return x

    def get_architecture_params(self) -> Dict:
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'depth': self.num_layers,
            'width': self.d_model,
            'min_width': self.d_model,
            'max_width': self.d_model * 4,
            'num_params': total_params,
            'connectivity': total_params / (self.d_model * self.num_layers)
        }


# ============================================================================
# HYBRID ARCHITECTURES
# ============================================================================

class HybridCNNTransformer(nn.Module):
    """Hybrid CNN + Transformer architecture."""

    def __init__(self, num_classes=10, cnn_channels=[64, 128, 256],
                 d_model=256, num_transformer_layers=4, nhead=8):
        super().__init__()
        self.cnn_channels = cnn_channels
        self.d_model = d_model
        self.num_transformer_layers = num_transformer_layers

        # CNN feature extractor
        cnn_layers = []
        in_ch = 3
        for out_ch in cnn_channels:
            cnn_layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ])
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)

        # Calculate feature map size
        self.feature_size = 32 // (2 ** len(cnn_channels))  # After pooling
        self.num_patches = self.feature_size ** 2

        # Project CNN features to transformer dimension
        self.proj = nn.Linear(cnn_channels[-1], d_model)

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, d_model))

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_transformer_layers)

        # Classification head
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # CNN feature extraction
        x = self.cnn(x)  # (B, C, H, W)

        # Reshape for transformer
        batch_size = x.size(0)
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)

        # Project to transformer dimension
        x = self.proj(x)  # (B, num_patches, d_model)

        # Add positional encoding
        x = x + self.pos_embed

        # Transformer
        x = self.transformer(x)

        # Global average pooling
        x = x.mean(dim=1)
        x = self.norm(x)

        # Classification
        x = self.fc(x)

        return x

    def get_architecture_params(self) -> Dict:
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'depth': len(self.cnn_channels) + self.num_transformer_layers,
            'width': (np.mean(self.cnn_channels) + self.d_model) / 2,
            'min_width': min(self.cnn_channels),
            'max_width': max(self.cnn_channels + [self.d_model]),
            'num_params': total_params,
            'connectivity': total_params / (len(self.cnn_channels) + self.num_transformer_layers)
        }


# ============================================================================
# ARCHITECTURE FACTORY - EXTENDED
# ============================================================================

def create_extended_architecture(arch_name: str, input_dim: int,
                                num_classes: int, in_channels: int = 3) -> nn.Module:
    """
    Factory for extended architecture suite.

    New architectures:
    - resnet50, resnet101: Large ResNets
    - transformer_large, transformer_xlarge: Large Transformers
    - vit_small, vit_base, vit_large: Vision Transformers
    - hybrid_cnn_transformer: Hybrid architecture
    """

    architectures = {
        # Large ResNets
        'resnet50': lambda: LargeResNet(num_classes, depth='50'),
        'resnet101': lambda: LargeResNet(num_classes, depth='101'),

        # Large Transformers
        'transformer_large': lambda: LargeTransformer(
            input_dim, d_model=512, nhead=8, num_layers=12,
            num_classes=num_classes
        ),
        'transformer_xlarge': lambda: LargeTransformer(
            input_dim, d_model=768, nhead=12, num_layers=24,
            num_classes=num_classes
        ),

        # Vision Transformers
        'vit_small': lambda: VisionTransformer(
            img_size=32, patch_size=4, d_model=384,
            num_layers=12, nhead=6, num_classes=num_classes
        ),
        'vit_base': lambda: VisionTransformer(
            img_size=32, patch_size=4, d_model=768,
            num_layers=12, nhead=12, num_classes=num_classes
        ),
        'vit_large': lambda: VisionTransformer(
            img_size=32, patch_size=2, d_model=1024,
            num_layers=24, nhead=16, num_classes=num_classes
        ),

        # Hybrid architectures
        'hybrid_cnn_transformer': lambda: HybridCNNTransformer(
            num_classes, cnn_channels=[64, 128, 256],
            d_model=256, num_transformer_layers=4
        ),
        'hybrid_large': lambda: HybridCNNTransformer(
            num_classes, cnn_channels=[128, 256, 512],
            d_model=512, num_transformer_layers=8
        ),
    }

    if arch_name not in architectures:
        raise ValueError(f"Unknown architecture: {arch_name}. "
                        f"Available: {list(architectures.keys())}")

    return architectures[arch_name]()


# ============================================================================
# ARCHITECTURE CATALOG
# ============================================================================

EXTENDED_ARCHITECTURE_CATALOG = {
    # Original architectures (from phase1_calibration.py)
    'original': [
        'mlp_shallow_2', 'mlp_medium_5', 'mlp_deep_10', 'mlp_verydeep_15',
        'mlp_narrow', 'mlp_medium', 'mlp_wide', 'mlp_verywide',
        'cnn_shallow', 'cnn_medium', 'cnn_deep',
        'resnet18',
        'transformer_shallow', 'transformer_medium', 'transformer_deep',
        'transformer_narrow', 'transformer_wide'
    ],

    # Extended architectures (this file)
    'large_resnets': [
        'resnet50',      # ~23M params
        'resnet101',     # ~42M params
    ],

    'large_transformers': [
        'transformer_large',   # ~40M params, 12 layers, 512-dim
        'transformer_xlarge',  # ~125M params, 24 layers, 768-dim
    ],

    'vision_transformers': [
        'vit_small',     # ~22M params, ViT-S/4
        'vit_base',      # ~86M params, ViT-B/4
        'vit_large',     # ~300M params, ViT-L/2
    ],

    'hybrid': [
        'hybrid_cnn_transformer',  # ~15M params
        'hybrid_large',            # ~45M params
    ]
}


def get_architecture_recommendations(dataset_size: str = 'small',
                                    compute_budget: str = 'medium') -> List[str]:
    """
    Recommend architectures based on dataset size and compute budget.

    Args:
        dataset_size: 'small' (<50K), 'medium' (50K-500K), 'large' (>500K)
        compute_budget: 'low' (CPU), 'medium' (1 GPU), 'high' (multi-GPU)

    Returns:
        List of recommended architecture names
    """
    recommendations = []

    if compute_budget == 'low':
        # CPU-friendly architectures
        recommendations.extend([
            'mlp_shallow_2', 'mlp_medium_5', 'cnn_shallow',
            'transformer_shallow'
        ])
    elif compute_budget == 'medium':
        # Single GPU
        recommendations.extend([
            'mlp_deep_10', 'cnn_deep', 'resnet18', 'resnet50',
            'transformer_medium', 'transformer_large',
            'vit_small', 'hybrid_cnn_transformer'
        ])
    else:  # high
        # Multi-GPU or large single GPU
        recommendations.extend([
            'resnet101', 'transformer_xlarge',
            'vit_base', 'vit_large', 'hybrid_large'
        ])

    # Filter by dataset size
    if dataset_size == 'small':
        # Avoid very large models that might overfit
        recommendations = [a for a in recommendations
                          if 'xlarge' not in a and 'vit_large' not in a]

    return recommendations


if __name__ == "__main__":
    print("Extended Architecture Suite for TAP Experiments")
    print("=" * 70)

    print("\nArchitecture Categories:")
    for category, archs in EXTENDED_ARCHITECTURE_CATALOG.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        for arch in archs:
            print(f"  - {arch}")

    print("\n" + "=" * 70)
    print("Total architectures:", sum(len(archs)
                                     for archs in EXTENDED_ARCHITECTURE_CATALOG.values()))

    print("\nRecommendations:")
    print("\nSmall dataset + Low compute:")
    print("  ", get_architecture_recommendations('small', 'low'))

    print("\nMedium dataset + Medium compute:")
    print("  ", get_architecture_recommendations('medium', 'medium'))

    print("\nLarge dataset + High compute:")
    print("  ", get_architecture_recommendations('large', 'high'))
