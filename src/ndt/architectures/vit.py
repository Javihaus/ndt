"""Handler for Vision Transformer (ViT) architectures."""

from typing import List

import torch.nn as nn

from ndt.architectures.base import ArchitectureHandler


class ViTHandler(ArchitectureHandler):
    """Handler for Vision Transformer architectures.

    Monitors patch embeddings, attention layers, and MLP blocks.
    """

    def validate_model(self, model: nn.Module) -> bool:
        """Check if model is a Vision Transformer.

        Args:
            model: The neural network model

        Returns:
            True if model appears to be a ViT
        """
        # ViTs typically have both conv (for patch embedding) and attention
        has_conv = False
        has_attention = False
        has_patch_embed = False

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Conv1d)):
                has_conv = True
                if "patch" in name.lower() or "embed" in name.lower():
                    has_patch_embed = True
            if isinstance(module, nn.MultiheadAttention):
                has_attention = True

        # ViT has conv (patch embedding) + attention, or explicit patch embedding
        return (has_conv and has_attention) or has_patch_embed

    def get_activation_layers(self, model: nn.Module) -> List[nn.Module]:
        """Get key layers to monitor in ViT.

        Monitors: patch embedding, attention layers, and MLP blocks.

        Args:
            model: The neural network model

        Returns:
            List of layer modules to track
        """
        layers = []

        # 1. Patch embedding layer
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if "patch" in name.lower() or "embed" in name.lower():
                    layers.append(module)
                    break  # Usually just one

        # 2. Attention layers
        attn_layers = [m for m in model.modules() if isinstance(m, nn.MultiheadAttention)]
        layers.extend(attn_layers)

        # 3. MLP/FFN blocks (sample to avoid too many)
        mlp_layers = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and ("mlp" in name.lower() or "ffn" in name.lower()):
                mlp_layers.append(module)

        # Sample MLP layers if too many (keep first Linear of each MLP block)
        if len(mlp_layers) > 10:
            mlp_layers = mlp_layers[::2]

        layers.extend(mlp_layers)

        return layers

    def get_layer_names(self, model: nn.Module) -> List[str]:
        """Generate names for ViT layers.

        Args:
            model: The neural network model

        Returns:
            List of layer names
        """
        layers = self.get_activation_layers(model)
        names = []

        patch_count = 0
        attn_count = 0
        mlp_count = 0
        other_count = 0

        for layer in layers:
            # Identify layer type from model structure
            layer_name = None
            for name, module in model.named_modules():
                if module is layer:
                    layer_name = name
                    break

            if layer_name:
                if "patch" in layer_name.lower() or "embed" in layer_name.lower():
                    names.append(f"PatchEmbed_{patch_count}")
                    patch_count += 1
                elif isinstance(layer, nn.MultiheadAttention):
                    names.append(f"Attention_{attn_count}")
                    attn_count += 1
                elif "mlp" in layer_name.lower() or "ffn" in layer_name.lower():
                    names.append(f"MLP_{mlp_count}")
                    mlp_count += 1
                else:
                    names.append(f"Layer_{other_count}")
                    other_count += 1
            else:
                # Fallback naming
                if isinstance(layer, nn.MultiheadAttention):
                    names.append(f"Attention_{attn_count}")
                    attn_count += 1
                else:
                    names.append(f"Layer_{other_count}")
                    other_count += 1

        return names
