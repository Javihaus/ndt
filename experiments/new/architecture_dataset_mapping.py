"""
Architecture-Dataset Mapping
==============================

Maps each architecture to appropriate datasets based on:
- Input size requirements
- Modality (vision, text, sequences)
- Model design (CNNs need images, transformers can use text/images)
"""

# Architecture catalog from phase1_calibration.py
ARCHITECTURES = [
    # Depth variation MLPs
    'mlp_shallow_2', 'mlp_medium_5', 'mlp_deep_10', 'mlp_verydeep_15',
    # Width variation MLPs
    'mlp_narrow', 'mlp_medium', 'mlp_wide', 'mlp_verywide',
    # CNNs
    'cnn_shallow', 'cnn_medium', 'cnn_deep',
    # ResNet
    'resnet18',
    # Transformers
    'transformer_shallow', 'transformer_medium', 'transformer_deep',
    'transformer_narrow', 'transformer_wide',
]

# Dataset definitions with modality and size
DATASET_INFO = {
    # Real dataset (only one we have)
    'mnist': {
        'modality': 'vision',
        'size': (1, 28, 28),
        'num_classes': 10,
        'real': True
    },

    # Synthetic vision datasets
    'mnist_synthetic': {
        'modality': 'vision',
        'size': (1, 28, 28),
        'num_classes': 10,
        'real': False
    },
    'cifar_synthetic': {
        'modality': 'vision',
        'size': (3, 32, 32),
        'num_classes': 10,
        'real': False
    },
    'imagenet_small': {
        'modality': 'vision',
        'size': (3, 224, 224),
        'num_classes': 100,
        'real': False,
        'description': 'ImageNet-like for ViT/large models'
    },
    'imagenet_large': {
        'modality': 'vision',
        'size': (3, 384, 384),
        'num_classes': 1000,
        'real': False,
        'description': 'Large images for ViT-Large'
    },

    # Text/sequence datasets
    'text_short': {
        'modality': 'text',
        'size': (128,),  # sequence length
        'num_classes': 4,
        'vocab_size': 30000,
        'real': False,
        'description': 'Short text for BERT-style'
    },
    'text_long': {
        'modality': 'text',
        'size': (512,),
        'num_classes': 10,
        'vocab_size': 50000,
        'real': False,
        'description': 'Long text for GPT-style'
    },
}


def get_compatible_datasets(arch_name: str):
    """
    Return list of compatible datasets for an architecture.

    Args:
        arch_name: Architecture name

    Returns:
        List of dataset names compatible with this architecture
    """

    # MLPs - can use any vision dataset (flatten to vector)
    if 'mlp' in arch_name:
        return ['mnist', 'mnist_synthetic', 'cifar_synthetic']

    # CNNs - need image data, prefer smaller images
    elif 'cnn' in arch_name:
        return ['mnist', 'cifar_synthetic', 'mnist_synthetic']

    # ResNets - work with various image sizes but designed for larger
    elif 'resnet' in arch_name:
        return ['cifar_synthetic', 'imagenet_small']

    # Transformers - can use text OR image patches
    elif 'transformer' in arch_name:
        # For vision transformers: use larger images
        # For now, use both text and vision to test across modalities
        return ['text_short', 'imagenet_small', 'cifar_synthetic']

    # Default
    else:
        return ['mnist', 'cifar_synthetic']


def create_experiment_plan(architectures=None, max_datasets_per_arch=3):
    """
    Create an intelligent experiment plan matching architectures to datasets.

    Args:
        architectures: List of architecture names (default: all)
        max_datasets_per_arch: Maximum datasets to test per architecture

    Returns:
        List of (architecture, dataset) tuples
    """
    if architectures is None:
        architectures = ARCHITECTURES

    experiment_plan = []

    for arch in architectures:
        compatible = get_compatible_datasets(arch)

        # Limit number of datasets per architecture
        datasets_to_use = compatible[:max_datasets_per_arch]

        for dataset in datasets_to_use:
            experiment_plan.append((arch, dataset))

    return experiment_plan


def print_experiment_plan():
    """Print the experiment plan for review."""
    plan = create_experiment_plan()

    print("=" * 70)
    print("INTELLIGENT ARCHITECTURE-DATASET EXPERIMENT PLAN")
    print("=" * 70)
    print(f"\nTotal experiments: {len(plan)}\n")

    # Group by architecture type
    mlp_exp = [(a, d) for a, d in plan if 'mlp' in a]
    cnn_exp = [(a, d) for a, d in plan if 'cnn' in a]
    resnet_exp = [(a, d) for a, d in plan if 'resnet' in a]
    transformer_exp = [(a, d) for a, d in plan if 'transformer' in a]

    print(f"MLPs ({len(mlp_exp)} experiments):")
    for arch, dataset in mlp_exp:
        real_marker = "✓ REAL" if DATASET_INFO[dataset].get('real') else "  synthetic"
        print(f"  {real_marker} | {arch:20s} × {dataset}")

    print(f"\nCNNs ({len(cnn_exp)} experiments):")
    for arch, dataset in cnn_exp:
        real_marker = "✓ REAL" if DATASET_INFO[dataset].get('real') else "  synthetic"
        print(f"  {real_marker} | {arch:20s} × {dataset}")

    print(f"\nResNets ({len(resnet_exp)} experiments):")
    for arch, dataset in resnet_exp:
        real_marker = "✓ REAL" if DATASET_INFO[dataset].get('real') else "  synthetic"
        size = DATASET_INFO[dataset]['size']
        print(f"  {real_marker} | {arch:20s} × {dataset:20s} ({size})")

    print(f"\nTransformers ({len(transformer_exp)} experiments):")
    for arch, dataset in transformer_exp:
        real_marker = "✓ REAL" if DATASET_INFO[dataset].get('real') else "  synthetic"
        modality = DATASET_INFO[dataset]['modality']
        print(f"  {real_marker} | {arch:20s} × {dataset:20s} ({modality})")

    print("\n" + "=" * 70)
    print(f"REAL data experiments: {sum(1 for a, d in plan if DATASET_INFO[d].get('real'))}")
    print(f"Synthetic experiments: {sum(1 for a, d in plan if not DATASET_INFO[d].get('real'))}")
    print("=" * 70)

    return plan


if __name__ == "__main__":
    print_experiment_plan()
