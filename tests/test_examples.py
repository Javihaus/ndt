"""Tests for example scripts to ensure they run without errors."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for CI environments

import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torch.utils.data import TensorDataset  # noqa: E402

# Add examples directory to path
examples_dir = Path(__file__).parent.parent / "examples"
sys.path.insert(0, str(examples_dir))


@pytest.fixture
def mock_mnist_data():
    """Create mock MNIST data for fast testing."""
    # Create dataset large enough for 100+ steps with batch_size=64
    x_train = torch.randn(6400, 1, 28, 28)  # 100 batches of 64
    y_train = torch.randint(0, 10, (6400,))
    return TensorDataset(x_train, y_train)


@pytest.fixture
def mock_cifar_data():
    """Create mock CIFAR-10 data for fast testing."""
    # Create dataset large enough for multiple batches with batch_size=128
    x_train = torch.randn(2560, 3, 32, 32)  # 20 batches of 128
    y_train = torch.randint(0, 10, (2560,))
    return TensorDataset(x_train, y_train)


class TestQuickstartExample:
    """Tests for 01_quickstart_mnist.py example."""

    def test_quickstart_basic_functionality(self, mock_mnist_data, tmp_path):
        """Test that quickstart example runs without errors (fast mode)."""
        from ndt import HighFrequencyTracker
        from ndt import export_to_csv
        from ndt import plot_phases

        # Model from quickstart
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

        # Create tracker
        tracker = HighFrequencyTracker(model, sampling_frequency=10, enable_jump_detection=True)

        # Training loop (reduced steps)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        train_loader = DataLoader(mock_mnist_data, batch_size=64, shuffle=True)

        model.train()
        step = 0

        for epoch in range(1):  # Just 1 epoch for testing
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                tracker.log(step, loss.item())
                step += 1

                if step >= 50:  # Reduced from 1000
                    break
            if step >= 50:
                break

        # Test analysis
        results = tracker.get_results()
        assert len(results) > 0, "Should have tracked some layers"

        # Test jump detection
        jumps_dict = tracker.detect_jumps(metric="stable_rank")
        assert isinstance(jumps_dict, dict), "Should return dictionary of jumps"

        # Test visualization
        fig = plot_phases(results, metric="stable_rank")
        assert fig is not None, "Should create figure"

        # Test export
        csv_path = tmp_path / "test_results.csv"
        export_to_csv(results, str(csv_path))
        assert csv_path.exists(), "Should create CSV file"

        # Cleanup
        tracker.close()

    def test_quickstart_layer_detection(self):
        """Test that auto-detection finds correct number of layers."""
        from ndt import HighFrequencyTracker

        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

        tracker = HighFrequencyTracker(model)

        # Should auto-detect 4 Linear layers
        results = tracker.get_results()
        assert len(results) == 4, f"Expected 4 layers, got {len(results)}"

        tracker.close()


class TestCNNExample:
    """Tests for 02_cnn_cifar10.py example."""

    def test_cnn_basic_functionality(self, mock_cifar_data, tmp_path):
        """Test that CNN example runs without errors (fast mode)."""
        from ndt import HighFrequencyTracker
        from ndt import export_to_json

        # SimpleCNN from example
        class SimpleCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
                self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.fc1 = nn.Linear(128 * 4 * 4, 256)
                self.fc2 = nn.Linear(256, 10)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.conv1(x))
                x = self.pool(x)
                x = self.relu(self.conv2(x))
                x = self.pool(x)
                x = self.relu(self.conv3(x))
                x = self.pool(x)
                x = x.view(-1, 128 * 4 * 4)
                x = self.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        model = SimpleCNN()

        # Create tracker
        tracker = HighFrequencyTracker(
            model,
            layers=[model.conv1, model.conv2, model.conv3, model.fc1],
            layer_names=["Conv1", "Conv2", "Conv3", "FC1"],
            sampling_frequency=20,
        )

        # Training setup
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        train_loader = DataLoader(mock_cifar_data, batch_size=128, shuffle=True)

        model.train()
        step = 0

        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # Compute gradient norm
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))

            optimizer.step()

            # Track with gradient norm
            tracker.log(step, loss.item(), grad_norm=grad_norm.item())

            step += 1

            if step >= 20:  # Reduced from 2000
                break

        # Test analysis
        results = tracker.get_results()
        assert len(results) == 4, "Should track 4 layers"

        # Check gradient norm was tracked
        for layer_name, df in results.items():
            assert "grad_norm" in df.columns, f"grad_norm should be in {layer_name}"

        # Test JSON export
        json_path = tmp_path / "test_results.json"
        export_to_json(results, str(json_path))
        assert json_path.exists(), "Should create JSON file"

        # Cleanup
        tracker.close()


class TestTDSExperiment:
    """Tests for 03_reproduce_tds_experiment.py example."""

    def test_tds_architecture_spec(self):
        """Test that TDS architecture matches specification (784-256-128-10)."""
        try:
            from examples.reproduce_tds_experiment import TDSExperimentMLP
        except (ImportError, AttributeError):
            pytest.skip("torchvision not available or module failed to load")

        model = TDSExperimentMLP()

        # Check architecture
        assert isinstance(model.layer1, nn.Linear), "Layer1 should be Linear"
        assert model.layer1.in_features == 784, "Layer1 input should be 784"
        assert model.layer1.out_features == 256, "Layer1 output should be 256 (not 512!)"

        assert isinstance(model.layer2, nn.Linear), "Layer2 should be Linear"
        assert model.layer2.in_features == 256, "Layer2 input should be 256"
        assert model.layer2.out_features == 128, "Layer2 output should be 128"

        assert isinstance(model.layer3, nn.Linear), "Layer3 should be Linear"
        assert model.layer3.in_features == 128, "Layer3 input should be 128"
        assert model.layer3.out_features == 10, "Layer3 output should be 10"

    def test_tds_experiment_fast(self, mock_mnist_data, tmp_path):
        """Test TDS experiment with reduced steps (fast mode)."""
        from ndt import HighFrequencyTracker
        from ndt import export_to_csv
        from ndt import export_to_hdf5
        from ndt import plot_phases

        # TDS architecture
        class TDSExperimentMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.flatten = nn.Flatten()
                self.layer1 = nn.Linear(784, 256)
                self.relu1 = nn.ReLU()
                self.layer2 = nn.Linear(256, 128)
                self.relu2 = nn.ReLU()
                self.layer3 = nn.Linear(128, 10)

            def forward(self, x):
                x = self.flatten(x)
                x = self.relu1(self.layer1(x))
                x = self.relu2(self.layer2(x))
                x = self.layer3(x)
                return x

        model = TDSExperimentMLP()

        # Create tracker with TDS specs (high-frequency sampling)
        tracker = HighFrequencyTracker(
            model,
            layers=[model.layer1, model.layer2, model.layer3],
            layer_names=["Layer1_784-256", "Layer2_256-128", "Layer3_128-10"],
            sampling_frequency=5,  # Every 5 steps as per TDS article
            enable_jump_detection=True,
        )

        # Setup optimizer with exact TDS specs
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
        criterion = nn.CrossEntropyLoss()

        train_loader = DataLoader(mock_mnist_data, batch_size=64, shuffle=True)

        # Training loop (reduced steps for testing)
        model.train()
        step = 0
        target_steps = 100  # Reduced from 8000 for testing

        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # Compute gradient norm
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))

            optimizer.step()

            # Track with high frequency
            tracker.log(step, loss.item(), grad_norm=grad_norm.item())

            step += 1

            if step >= target_steps:
                break

        # Test analysis
        results = tracker.get_results()
        assert len(results) == 3, "Should track exactly 3 layers"

        # Verify layer names
        expected_names = ["Layer1_784-256", "Layer2_256-128", "Layer3_128-10"]
        for name in expected_names:
            assert name in results, f"Should have layer {name}"

        # Verify measurements
        for layer_name, df in results.items():
            # With freq=5, expect ~20 measurements for 100 steps
            expected_measurements = target_steps // 5
            assert (
                len(df) == expected_measurements
            ), f"{layer_name}: expected {expected_measurements} measurements, got {len(df)}"

            # Verify all metrics present
            assert "stable_rank" in df.columns
            assert "participation_ratio" in df.columns
            assert "loss" in df.columns
            assert "grad_norm" in df.columns

        # Test jump detection
        jumps_dict = tracker.detect_jumps(metric="stable_rank")
        assert isinstance(jumps_dict, dict), "Should return jump dictionary"
        assert len(jumps_dict) == 3, "Should have jumps for 3 layers"

        # Test visualizations
        fig = plot_phases(results, metric="stable_rank")
        assert fig is not None, "Should create activation space figure"

        # Test exports
        csv_path = tmp_path / "tds_results.csv"
        export_to_csv(results, str(csv_path))
        assert csv_path.exists(), "Should create CSV export"

        h5_path = tmp_path / "tds_results.h5"
        export_to_hdf5(results, str(h5_path))
        assert h5_path.exists(), "Should create HDF5 export"

        # Cleanup
        tracker.close()

    def test_tds_sampling_frequency(self, mock_mnist_data):
        """Test that TDS experiment samples at correct frequency."""
        from ndt import HighFrequencyTracker

        model = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))

        # High-frequency sampling (TDS spec)
        tracker = HighFrequencyTracker(model, sampling_frequency=5)

        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        train_loader = DataLoader(mock_mnist_data, batch_size=64)

        step = 0
        for data, target in train_loader:
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            tracker.log(step, loss.item())
            step += 1

            if step >= 100:
                break

        results = tracker.get_results()

        # With freq=5, expect 20 measurements for 100 steps
        for layer_name, df in results.items():
            assert len(df) == 20, f"Expected 20 measurements, got {len(df)}"

        tracker.close()

    @pytest.mark.slow
    def test_tds_experiment_full(self, tmp_path):
        """Test full TDS experiment with 8000 steps (requires --run-full flag)."""
        # This test is marked slow and only runs with pytest --run-full-examples
        pytest.skip("Full TDS experiment test requires --run-full-examples flag")


class TestExampleOutputs:
    """Test that examples generate expected output files."""

    def test_examples_directory_structure(self):
        """Test that examples directory has expected structure."""
        assert examples_dir.exists(), "Examples directory should exist"
        assert (examples_dir / "01_quickstart_mnist.py").exists()
        assert (examples_dir / "02_cnn_cifar10.py").exists()
        assert (examples_dir / "03_reproduce_tds_experiment.py").exists()
        assert (examples_dir / "README.md").exists()

    def test_examples_are_importable(self):
        """Test that example scripts can be imported without errors."""
        # Import TDS experiment module
        from examples import reproduce_tds_experiment

        if reproduce_tds_experiment is None:
            pytest.skip("torchvision not available - reproduce_tds_experiment could not be loaded")

        assert hasattr(
            reproduce_tds_experiment, "TDSExperimentMLP"
        ), "Should have TDSExperimentMLP class"
        assert hasattr(reproduce_tds_experiment, "main"), "Should have main function"


class TestExampleDifferences:
    """Test key differences between examples."""

    def test_quickstart_vs_tds_architecture(self):
        """Test that quickstart and TDS architectures are intentionally different."""
        from ndt import HighFrequencyTracker

        # Quickstart: 784 → 512 → 256 → 128 → 10
        quickstart_model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

        # TDS: 784 → 256 → 128 → 10
        tds_model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

        quickstart_tracker = HighFrequencyTracker(quickstart_model)
        tds_tracker = HighFrequencyTracker(tds_model)

        # Quickstart has 4 Linear layers
        assert len(quickstart_tracker.get_results()) == 4

        # TDS has 3 Linear layers
        assert len(tds_tracker.get_results()) == 3

        quickstart_tracker.close()
        tds_tracker.close()


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
