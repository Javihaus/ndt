"""Handler for Transformer architectures (BERT, GPT, etc.)."""

from typing import List

import torch.nn as nn

from ndt.architectures.base import ArchitectureHandler


class TransformerHandler(ArchitectureHandler):
    """Handler for Transformer-based architectures.

    Monitors attention outputs and feed-forward network outputs in each layer.
    Compatible with BERT, GPT, and similar transformer models.
    """

    def validate_model(self, model: nn.Module) -> bool:
        """Check if model is a Transformer (contains MultiheadAttention).

        Args:
            model: The neural network model

        Returns:
            True if model appears to be a Transformer
        """
        has_attention = False
        for module in model.modules():
            if isinstance(module, nn.MultiheadAttention):
                has_attention = True
                break
            # Also check for common transformer layer names
            if module.__class__.__name__ in ["TransformerEncoderLayer", "TransformerDecoderLayer"]:
                has_attention = True
                break
        return has_attention

    def get_activation_layers(self, model: nn.Module) -> List[nn.Module]:
        """Get attention and FFN layers for monitoring.

        Args:
            model: The neural network model

        Returns:
            List of layer modules to track
        """
        layers = []

        # Look for MultiheadAttention modules
        for module in model.modules():
            if isinstance(module, nn.MultiheadAttention):
                layers.append(module)

        # Also look for Linear layers in feed-forward blocks
        # Typically the first Linear in each transformer block's FFN
        in_transformer_block = False
        for name, module in model.named_modules():
            if "encoder" in name.lower() or "decoder" in name.lower() or "layer" in name.lower():
                in_transformer_block = True

            if in_transformer_block and isinstance(module, nn.Linear):
                # Add FFN layers (typically 2 per transformer block)
                if "fc1" in name or "fc2" in name or "ffn" in name.lower():
                    layers.append(module)

        # If no structured layers found, fall back to sampling Linear layers
        if not layers:
            linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
            # Sample every 2nd Linear layer to avoid too many
            layers = linear_layers[::2] if len(linear_layers) > 10 else linear_layers

        return layers

    def get_layer_names(self, model: nn.Module) -> List[str]:
        """Generate names for Transformer layers.

        Args:
            model: The neural network model

        Returns:
            List of layer names
        """
        layers = self.get_activation_layers(model)
        names = []

        attn_count = 0
        ffn_count = 0
        fc_count = 0

        for layer in layers:
            if isinstance(layer, nn.MultiheadAttention):
                names.append(f"Attention_{attn_count}")
                attn_count += 1
            elif isinstance(layer, nn.Linear):
                # Try to determine if it's an FFN layer
                layer_name = None
                for name, module in model.named_modules():
                    if module is layer:
                        layer_name = name
                        break

                if layer_name and ("fc" in layer_name.lower() or "ffn" in layer_name.lower()):
                    names.append(f"FFN_{ffn_count}_{layer.out_features}")
                    ffn_count += 1
                else:
                    names.append(f"FC_{fc_count}_{layer.out_features}")
                    fc_count += 1

        return names
