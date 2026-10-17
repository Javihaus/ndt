"""Tests for architecture handlers and the auto-detection registry."""

import pytest
import torch.nn as nn

from ndt.architectures import ArchitectureHandler
from ndt.architectures import CNNHandler
from ndt.architectures import MLPHandler
from ndt.architectures import TransformerHandler
from ndt.architectures import ViTHandler
from ndt.architectures import detect_architecture
from ndt.architectures import get_handler
from ndt.architectures.registry import get_layers_for_model

# ---------------------------------------------------------------------------
# Tiny model builders
# ---------------------------------------------------------------------------


class TinyResNet(nn.Module):
    """Minimal ResNet-style model with layer1/layer2 groups."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3)
        self.layer1 = nn.Sequential(nn.Conv2d(4, 4, 3), nn.Conv2d(4, 6, 3))
        self.layer2 = nn.Sequential(nn.Conv2d(6, 8, 3), nn.Conv2d(8, 12, 3))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc_hidden = nn.Linear(12, 7)
        self.fc = nn.Linear(7, 3)


class AttentionOnly(nn.Module):
    """Model whose only special module is a raw MultiheadAttention."""

    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(8, 2)
        self.head = nn.Linear(8, 2)


class FFNBlock(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, 2)
        self.fc1 = nn.Linear(d, 16)
        self.fc2 = nn.Linear(16, d)


class FFNTransformer(nn.Module):
    """Transformer with explicitly named fc1/fc2 feed-forward layers."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([FFNBlock(), FFNBlock()])
        self.head = nn.Linear(8, 2)


class TransformerEncoderLayer(nn.Module):
    """Custom class named like the torch layer but without MultiheadAttention."""

    def __init__(self, n_linear):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(4, 4) for _ in range(n_linear)])


class ViTBlock(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, 2)
        self.mlp = nn.Sequential(nn.Linear(d, 16), nn.GELU(), nn.Linear(16, d))


class TinyViT(nn.Module):
    def __init__(self, depth=2):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 8, kernel_size=4, stride=4)
        self.blocks = nn.ModuleList([ViTBlock() for _ in range(depth)])
        self.head = nn.Linear(8, 10)


def tiny_transformer():
    layer = nn.TransformerEncoderLayer(d_model=8, nhead=2, dim_feedforward=16)
    return nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class TestArchitectureHandlerBase:
    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            ArchitectureHandler()

    def test_concrete_subclass_and_get_name(self):
        class DummyHandler(ArchitectureHandler):
            def validate_model(self, model):
                super().validate_model(model)
                return True

            def get_activation_layers(self, model):
                super().get_activation_layers(model)
                return [model]

            def get_layer_names(self, model):
                super().get_layer_names(model)
                return ["root"]

        handler = DummyHandler()
        model = nn.Linear(2, 2)
        assert handler.get_name() == "Dummy"
        assert handler.validate_model(model) is True
        assert handler.get_activation_layers(model) == [model]
        assert handler.get_layer_names(model) == ["root"]

    @pytest.mark.parametrize(
        "handler,name",
        [
            (MLPHandler(), "MLP"),
            (CNNHandler(), "CNN"),
            (TransformerHandler(), "Transformer"),
            (ViTHandler(), "ViT"),
        ],
    )
    def test_builtin_names(self, handler, name):
        assert handler.get_name() == name


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class TestMLPHandler:
    def test_validate(self, small_mlp, simple_cnn):
        h = MLPHandler()
        assert h.validate_model(small_mlp)
        assert not h.validate_model(simple_cnn)
        assert not h.validate_model(AttentionOnly())
        assert not h.validate_model(nn.Sequential(nn.ReLU()))

    def test_layers_and_names(self, small_mlp):
        h = MLPHandler()
        layers = h.get_activation_layers(small_mlp)
        assert layers == [small_mlp[0], small_mlp[2], small_mlp[4]]
        assert h.get_layer_names(small_mlp) == [
            "Linear_0_10x20",
            "Linear_1_20x15",
            "Linear_2_15x5",
        ]


# ---------------------------------------------------------------------------
# CNN
# ---------------------------------------------------------------------------


class TestCNNHandler:
    def test_validate(self, simple_cnn, small_mlp):
        h = CNNHandler()
        assert h.validate_model(simple_cnn)
        assert h.validate_model(nn.Sequential(nn.Conv1d(1, 2, 3)))
        assert h.validate_model(nn.Sequential(nn.Conv3d(1, 2, 3)))
        assert not h.validate_model(small_mlp)

    def test_simple_cnn_layers(self, simple_cnn):
        h = CNNHandler()
        assert not h._is_resnet(simple_cnn)
        layers = h.get_activation_layers(simple_cnn)
        assert layers == [simple_cnn[0], simple_cnn[3], simple_cnn[7]]
        assert h.get_layer_names(simple_cnn) == ["Conv_0_16ch", "Conv_1_32ch", "FC_0_10"]

    def test_cnn_without_linear(self):
        model = nn.Sequential(nn.Conv2d(1, 2, 3), nn.Conv2d(2, 5, 3))
        h = CNNHandler()
        assert h.get_activation_layers(model) == [model[0], model[1]]
        assert h.get_layer_names(model) == ["Conv_0_2ch", "Conv_1_5ch"]

    def test_resnet_layers(self):
        model = TinyResNet()
        h = CNNHandler()
        assert h._is_resnet(model)
        layers = h.get_activation_layers(model)
        # Last conv of each layer group, then the final Linear (not the first).
        assert layers == [model.layer1[1], model.layer2[1], model.fc]
        assert h.get_layer_names(model) == ["Conv_0_6ch", "Conv_1_12ch", "FC_0_3"]


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------


class TestTransformerHandler:
    def test_validate(self, small_mlp):
        h = TransformerHandler()
        assert h.validate_model(tiny_transformer())
        assert h.validate_model(AttentionOnly())
        assert h.validate_model(TransformerEncoderLayer(1))  # matched by class name
        assert not h.validate_model(small_mlp)

    def test_torch_encoder_layers_are_attention_modules(self):
        model = tiny_transformer()
        h = TransformerHandler()
        layers = h.get_activation_layers(model)
        attn = [m for m in model.modules() if isinstance(m, nn.MultiheadAttention)]
        assert len(attn) == 2
        assert layers == attn
        assert h.get_layer_names(model) == ["Attention_0", "Attention_1"]

    def test_named_ffn_layers(self):
        model = FFNTransformer()
        h = TransformerHandler()
        layers = h.get_activation_layers(model)
        expected = [
            model.layers[0].attn,
            model.layers[1].attn,
            model.layers[0].fc1,
            model.layers[0].fc2,
            model.layers[1].fc1,
            model.layers[1].fc2,
        ]
        assert layers == expected
        assert h.get_layer_names(model) == [
            "Attention_0",
            "Attention_1",
            "FFN_0_16",
            "FFN_1_8",
            "FFN_2_16",
            "FFN_3_8",
        ]

    def test_fallback_keeps_all_linears_when_few(self):
        model = TransformerEncoderLayer(4)
        h = TransformerHandler()
        layers = h.get_activation_layers(model)
        assert layers == list(model.proj)
        assert h.get_layer_names(model) == ["FC_0_4", "FC_1_4", "FC_2_4", "FC_3_4"]

    def test_fallback_samples_every_other_linear_when_many(self):
        model = TransformerEncoderLayer(12)
        layers = TransformerHandler().get_activation_layers(model)
        assert layers == list(model.proj)[::2]
        assert len(layers) == 6


# ---------------------------------------------------------------------------
# ViT
# ---------------------------------------------------------------------------


class TestViTHandler:
    def test_validate(self, simple_cnn, small_mlp):
        h = ViTHandler()
        assert h.validate_model(TinyViT())
        # Explicit patch embedding alone is enough.
        patch_only = nn.Module()
        patch_only.patch_embed = nn.Conv2d(3, 4, 2)
        assert h.validate_model(patch_only)
        # Generic conv + attention also counts.
        conv_attn = nn.Module()
        conv_attn.conv = nn.Conv2d(3, 4, 2)
        conv_attn.attn = nn.MultiheadAttention(4, 1)
        assert h.validate_model(conv_attn)
        assert not h.validate_model(simple_cnn)
        assert not h.validate_model(small_mlp)
        assert not h.validate_model(tiny_transformer())

    def test_layers_and_names(self):
        model = TinyViT(depth=2)
        h = ViTHandler()
        layers = h.get_activation_layers(model)
        expected = [
            model.patch_embed,
            model.blocks[0].attn,
            model.blocks[1].attn,
            model.blocks[0].mlp[0],
            model.blocks[0].mlp[2],
            model.blocks[1].mlp[0],
            model.blocks[1].mlp[2],
        ]
        assert layers == expected
        assert h.get_layer_names(model) == [
            "PatchEmbed_0",
            "Attention_0",
            "Attention_1",
            "MLP_0",
            "MLP_1",
            "MLP_2",
            "MLP_3",
        ]

    def test_many_mlp_layers_are_subsampled(self):
        model = TinyViT(depth=6)  # 12 MLP linears
        layers = ViTHandler().get_activation_layers(model)
        mlp = [m for b in model.blocks for m in (b.mlp[0], b.mlp[2])]
        assert layers[0] is model.patch_embed
        assert layers[1:7] == [b.attn for b in model.blocks]
        assert layers[7:] == mlp[::2]

    def test_naming_of_unnamed_and_other_layers(self, monkeypatch):
        model = TinyViT(depth=1)
        h = ViTHandler()
        foreign_attn = nn.MultiheadAttention(8, 2)
        foreign_linear = nn.Linear(2, 2)
        monkeypatch.setattr(
            h,
            "get_activation_layers",
            lambda m: [model.head, foreign_attn, foreign_linear],
        )
        # "head" is in the model but matches no keyword; the others are not in it.
        assert h.get_layer_names(model) == ["Layer_0", "Attention_0", "Layer_1"]

    def test_root_attention_uses_fallback_name(self):
        model = nn.MultiheadAttention(4, 1)
        # The root module has the empty name "", so the fallback branch names it.
        assert ViTHandler().get_layer_names(model) == ["Attention_0"]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_detect_architecture(self, small_mlp, simple_cnn):
        assert isinstance(detect_architecture(small_mlp), MLPHandler)
        assert isinstance(detect_architecture(simple_cnn), CNNHandler)
        assert isinstance(detect_architecture(tiny_transformer()), TransformerHandler)
        assert isinstance(detect_architecture(TinyViT()), ViTHandler)
        assert detect_architecture(nn.Sequential(nn.ReLU())) is None

    @pytest.mark.parametrize("name,cls", [("mlp", MLPHandler), ("MLP", MLPHandler)])
    def test_get_handler_explicit(self, small_mlp, name, cls):
        assert isinstance(get_handler(small_mlp, name), cls)

    def test_get_handler_explicit_cnn_and_vit(self, simple_cnn):
        assert isinstance(get_handler(simple_cnn, "cnn"), CNNHandler)
        assert isinstance(get_handler(TinyViT(), "vit"), ViTHandler)
        assert isinstance(get_handler(tiny_transformer(), "transformer"), TransformerHandler)

    def test_get_handler_unknown_name(self, small_mlp):
        with pytest.raises(ValueError, match="Unknown architecture: rnn"):
            get_handler(small_mlp, "rnn")

    def test_get_handler_mismatch(self, small_mlp):
        with pytest.raises(ValueError, match="does not appear to be a CNN"):
            get_handler(small_mlp, "cnn")

    def test_get_handler_auto(self, simple_cnn):
        assert isinstance(get_handler(simple_cnn), CNNHandler)

    def test_get_handler_auto_fails(self):
        with pytest.raises(ValueError, match="Could not auto-detect"):
            get_handler(nn.Sequential(nn.ReLU()))

    def test_get_layers_for_model(self, small_mlp):
        layers, names = get_layers_for_model(small_mlp)
        assert layers == [small_mlp[0], small_mlp[2], small_mlp[4]]
        assert names == ["Linear_0_10x20", "Linear_1_20x15", "Linear_2_15x5"]
