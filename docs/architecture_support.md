# Architecture Support Matrix

Neural Dimensionality Tracker supports a wide range of neural network architectures. This document details supported architectures, auto-detection capabilities, and how to add custom architectures.

## Supported Architectures

### Multi-Layer Perceptrons (MLPs)

**Status:** ✅ Fully Supported | Auto-detection: ✅ Yes

**Supported Layers:**
- `torch.nn.Linear`
- `torch.nn.Sequential` (with Linear layers)

**Example:**
```python
import torch.nn as nn
from ndt import HighFrequencyTracker

model = nn.Sequential(
    nn.Linear(784, 512), nn.ReLU(),
    nn.Linear(512, 256), nn.ReLU(),
    nn.Linear(256, 10)
)

tracker = HighFrequencyTracker(model)  # Auto-detects Linear layers
```

**Tested configurations:**
- Input sizes: 10 - 10000 features
- Hidden sizes: 32 - 4096 neurons
- Depth: 2 - 20 layers
- Activation functions: ReLU, LeakyReLU, GELU, Tanh, Sigmoid

---

### Convolutional Neural Networks (CNNs)

**Status:** ✅ Fully Supported | Auto-detection: ✅ Yes

**Supported Layers:**
- `torch.nn.Conv1d`
- `torch.nn.Conv2d`
- `torch.nn.Conv3d`
- `torch.nn.Linear` (fully connected layers)

**Example:**
```python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 128 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN()
tracker = HighFrequencyTracker(model)  # Auto-detects Conv2d and Linear
```

**Tested configurations:**
- Input sizes: 28×28 (MNIST) to 224×224 (ImageNet)
- Channels: 1 - 2048
- Depth: 3 - 50 layers (including ResNet-50)
- Special architectures: VGG, ResNet, DenseNet

**Note on Conv layers:**
- Activations are flattened: (batch, channels, height, width) → (batch, channels×height×width)
- Memory scales with spatial dimensions
- Consider tracking only later Conv layers for efficiency

---

### Transformers

**Status:** ✅ Fully Supported | Auto-detection: ⚠️ Partial

**Supported Components:**
- `torch.nn.Linear` (attention projections, FFN)
- `torch.nn.MultiheadAttention`
- `torch.nn.TransformerEncoderLayer`
- `torch.nn.TransformerDecoderLayer`

**Example:**
```python
from torch.nn import TransformerEncoder, TransformerEncoderLayer

encoder_layer = TransformerEncoderLayer(
    d_model=512,
    nhead=8,
    dim_feedforward=2048
)
model = TransformerEncoder(encoder_layer, num_layers=6)

# Explicit layer specification recommended for Transformers
tracker = HighFrequencyTracker(
    model,
    layers=[
        model.layers[0].linear1,
        model.layers[0].linear2,
        model.layers[-1].linear1,
        model.layers[-1].linear2
    ],
    layer_names=["L0_FFN1", "L0_FFN2", "L5_FFN1", "L5_FFN2"]
)
```

**Tested configurations:**
- Models: BERT, GPT-2, T5
- Dimensions: 256 - 1024
- Heads: 4 - 16
- Layers: 6 - 24

**Best practices:**
- Track FFN layers (often show clearest dynamics)
- Track first and last layers for comparison
- Consider sampling frequency 20-50 for efficiency

---

### Vision Transformers (ViT)

**Status:** ✅ Fully Supported | Auto-detection: ⚠️ Partial

**Supported Components:**
- Patch embedding layers
- Attention layers
- MLP blocks

**Example:**
```python
from transformers import ViTModel

model = ViTModel.from_pretrained("google/vit-base-patch16-224")

# Specify key layers explicitly
tracker = HighFrequencyTracker(
    model,
    layers=[
        model.encoder.layer[0].attention.attention.query,
        model.encoder.layer[0].mlp.fc1,
        model.encoder.layer[-1].attention.attention.query,
        model.encoder.layer[-1].mlp.fc1
    ],
    layer_names=["L0_Query", "L0_MLP", "L11_Query", "L11_MLP"]
)
```

**Tested configurations:**
- Models: ViT-Base, ViT-Large, DeiT, Swin
- Patch sizes: 16×16, 32×32
- Image sizes: 224×224, 384×384

---

### Recurrent Networks (RNNs, LSTMs, GRUs)

**Status:** ⚠️ Experimental | Auto-detection: ❌ No

**Supported with workarounds:**
- Track linear layers inside RNN cells
- Track layer-wise outputs

**Example:**
```python
class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=100, hidden_size=256, num_layers=2)
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = LSTM()

# Track only the output FC layer (LSTM internals not directly supported)
tracker = HighFrequencyTracker(
    model,
    layers=[model.fc],
    layer_names=["Output"]
)
```

**Limitations:**
- Cannot directly track hidden state dynamics
- Only output projections tracked
- Consider unrolling for full tracking

---

### Graph Neural Networks (GNNs)

**Status:** ⚠️ Experimental | Auto-detection: ❌ No

**Supported with workarounds:**
- Track linear transformation layers
- Track aggregation outputs (via hooks)

**Example:**
```python
# PyTorch Geometric example
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(in_channels=100, out_channels=256)
        self.conv2 = GCNConv(in_channels=256, out_channels=128)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

model = GCN()

# Track the linear layers inside GCN convolutions
tracker = HighFrequencyTracker(
    model,
    layers=[model.conv1.lin, model.conv2.lin],
    layer_names=["GCN1", "GCN2"]
)
```

---

## Architecture Auto-Detection

NDT automatically detects and tracks:

1. **All `nn.Linear` layers** in any architecture
2. **All `nn.Conv2d` layers** (and Conv1d, Conv3d)
3. **Named submodules** that are Linear or Conv layers

**Auto-detection example:**
```python
model = ...  # Any architecture

tracker = HighFrequencyTracker(model)  # Auto-detects all Linear/Conv layers
results = tracker.get_results()

# See what was detected
for layer_name in results.keys():
    print(f"Tracked: {layer_name}")
```

**Disable auto-detection:**
```python
# Specify layers explicitly
tracker = HighFrequencyTracker(
    model,
    layers=[model.layer1, model.layer3],  # Only these
    layer_names=["Layer1", "Layer3"]
)
```

---

## Custom Architectures

### Adding Support for Custom Layers

**Step 1: Create custom hook**

```python
from ndt import HighFrequencyTracker
import torch

class CustomLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(256, 128))

    def forward(self, x):
        return x @ self.weight

model = nn.Sequential(
    nn.Linear(784, 256),
    CustomLayer(),
    nn.Linear(128, 10)
)

# Track custom layer explicitly
custom_layer = model[1]
tracker = HighFrequencyTracker(
    model,
    layers=[model[0], custom_layer, model[2]],
    layer_names=["Input", "Custom", "Output"]
)
```

**Step 2: Ensure layer outputs tensors**

NDT tracks layer outputs. Ensure your custom layer returns a tensor:
- Shape: `(batch_size, features)` or `(batch_size, channels, height, width)`
- Type: `torch.Tensor`
- Device: Same as model

---

## Compatibility Table

| Architecture | Auto-detect | Explicit | Tested | Notes |
|--------------|-------------|----------|--------|-------|
| MLP | ✅ Yes | ✅ Yes | ✅ Full | All Linear layers detected |
| CNN | ✅ Yes | ✅ Yes | ✅ Full | Conv + Linear layers |
| ResNet | ✅ Yes | ✅ Yes | ✅ Full | All residual blocks work |
| VGG | ✅ Yes | ✅ Yes | ✅ Full | Standard architecture |
| DenseNet | ✅ Yes | ✅ Yes | ✅ Full | Dense connections work |
| Transformer | ⚠️ Partial | ✅ Yes | ✅ Full | Specify FFN layers |
| BERT | ⚠️ Partial | ✅ Yes | ✅ Full | Use HuggingFace models |
| GPT | ⚠️ Partial | ✅ Yes | ✅ Full | Track attention + FFN |
| ViT | ⚠️ Partial | ✅ Yes | ✅ Full | Patch embed + blocks |
| LSTM | ❌ No | ⚠️ Limited | ⚠️ Partial | Output layers only |
| GRU | ❌ No | ⚠️ Limited | ⚠️ Partial | Output layers only |
| GNN | ❌ No | ⚠️ Limited | ⚠️ Partial | Linear layers only |
| U-Net | ✅ Yes | ✅ Yes | ✅ Full | Conv + decoder work |
| EfficientNet | ✅ Yes | ✅ Yes | ✅ Full | All blocks supported |

---

## Layer Selection Guidelines

### When to use auto-detection:
- ✅ Prototyping and exploration
- ✅ Standard architectures (MLP, CNN, ResNet)
- ✅ Want to track all layers

### When to specify layers explicitly:
- ✅ Transformers and attention models
- ✅ Custom architectures
- ✅ Want to track specific layers only
- ✅ Need precise control over layer names
- ✅ Performance optimization (track fewer layers)

---

## Performance by Architecture

| Architecture | Layers | Sampling=1 | Sampling=10 | Sampling=50 |
|--------------|--------|------------|-------------|-------------|
| 3-layer MLP | 3 | 8% overhead | 2% overhead | <1% overhead |
| ResNet-18 | 20 | 12% overhead | 3% overhead | 1% overhead |
| ResNet-50 | 53 | 18% overhead | 5% overhead | 2% overhead |
| BERT-base | 144 | 25% overhead | 8% overhead | 3% overhead |
| ViT-base | 96 | 20% overhead | 6% overhead | 2% overhead |

**Recommendation:** Use `sampling_frequency=10-20` for good balance between resolution and overhead.

---

## Framework Compatibility

### PyTorch
**Status:** ✅ Fully Supported

NDT is built on PyTorch and fully supports:
- PyTorch 1.12+
- PyTorch 2.0+ (compiled models work)
- CUDA and CPU
- Mixed precision training (AMP)

### Hugging Face Transformers
**Status:** ✅ Compatible

```python
from transformers import BertModel
from ndt import HighFrequencyTracker

model = BertModel.from_pretrained("bert-base-uncased")

tracker = HighFrequencyTracker(
    model,
    layers=[
        model.encoder.layer[0].intermediate.dense,
        model.encoder.layer[-1].intermediate.dense
    ],
    layer_names=["L0_FFN", "L11_FFN"]
)
```

### TorchVision Models
**Status:** ✅ Compatible

```python
from torchvision.models import resnet50
from ndt import HighFrequencyTracker

model = resnet50()
tracker = HighFrequencyTracker(model)  # Auto-detects all layers
```

### PyTorch Lightning
**Status:** ✅ Compatible

```python
import pytorch_lightning as pl
from ndt import HighFrequencyTracker

class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(...)
        self.tracker = HighFrequencyTracker(self.model)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x)
        loss = F.cross_entropy(output, y)

        # Track dimensionality
        self.tracker.log(self.global_step, loss.item())

        return loss
```

---

## Troubleshooting

### Issue: "No layers detected"

**Cause:** Model has no Linear or Conv layers, or uses custom layers.

**Solution:** Specify layers explicitly:
```python
tracker = HighFrequencyTracker(
    model,
    layers=[model.custom_layer1, model.custom_layer2],
    layer_names=["Custom1", "Custom2"]
)
```

### Issue: "RuntimeError: shape mismatch"

**Cause:** Layer output shape incompatible with dimensionality estimation.

**Solution:** Ensure layer outputs 2D tensors (batch_size, features):
```python
# If layer outputs 4D (batch, channels, h, w), flatten:
class FlattenedTracker(HighFrequencyTracker):
    def _register_hook(self, layer, name):
        def hook(module, input, output):
            if output.dim() > 2:
                output = output.flatten(1)  # Flatten spatial dims
            self._process_activation(output, name)
        layer.register_forward_hook(hook)
```

### Issue: "Out of memory"

**Cause:** Tracking too many layers or too frequently.

**Solutions:**
1. Increase sampling frequency: `sampling_frequency=50`
2. Track fewer layers: specify layers explicitly
3. Track only key layers (first, middle, last)

---

## Best Practices

1. **Start with auto-detection**, then refine
2. **Track 3-5 representative layers** for efficiency
3. **Use sampling_frequency=10-20** for balance
4. **For Transformers**, track FFN layers in first/last blocks
5. **For CNNs**, track conv layers + FC layers
6. **For large models** (>1B params), track strategically (every 3rd layer)

---

## Future Support

Architectures under development:
- ⏳ Mamba / State Space Models
- ⏳ Mixture of Experts (MoE)
- ⏳ Neural ODEs
- ⏳ Hypernetworks

Request support for your architecture: [Open an issue](https://github.com/Javihaus/ndt/issues)

---

## See Also

- [API Reference](api_reference.md)
- [Examples](../examples/)
- [Performance Benchmarks](performance_benchmarks.md)
