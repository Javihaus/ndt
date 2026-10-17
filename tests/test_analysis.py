"""Tests for activation analysis and feature visualization tools.

``ndt.analysis`` imports scikit-learn at package import time, but scikit-learn is
not a declared dependency of the package nor part of the CI requirements, so the
whole module is skipped when it is missing.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

pytest.importorskip("sklearn")

from ndt.analysis import ActivationAnalyzer  # noqa: E402
from ndt.analysis import FeatureVisualizer  # noqa: E402
from ndt.analysis import activation_analysis  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def two_blobs(n_per=10, dim=3, sep=10.0):
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 0.1, size=(n_per, dim))
    b = rng.normal(sep, 0.1, size=(n_per, dim))
    return np.vstack([a, b])


class Flat0(nn.Module):
    """Wraps a model and flattens its whole output to 1D."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return self.inner(x).flatten()


def tiny_cnn():
    conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
    fc = nn.Linear(1, 2, bias=False)
    with torch.no_grad():
        conv.weight.zero_()
        conv.weight[0, 0, 1, 1] = 1.0  # identity kernel
        fc.weight.copy_(torch.tensor([[1.0], [-1.0]]))
    model = nn.Sequential(conv, nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), fc)
    return model, conv


def linear_model():
    lin = nn.Linear(4, 3, bias=True)
    with torch.no_grad():
        lin.weight.copy_(torch.arange(12, dtype=torch.float32).view(3, 4) - 5.0)
        lin.bias.copy_(torch.tensor([0.5, -0.5, 1.0]))
    return lin


# ---------------------------------------------------------------------------
# ActivationAnalyzer
# ---------------------------------------------------------------------------


class TestActivationAnalyzer:
    def test_init(self):
        assert ActivationAnalyzer().results == {}

    def test_flatten_activation(self):
        analyzer = ActivationAnalyzer()
        t = torch.arange(24, dtype=torch.float32).view(4, 2, 3)
        out = analyzer.flatten_activation(t)
        assert isinstance(out, np.ndarray)
        assert out.shape == (4, 6)
        np.testing.assert_array_equal(out[1], np.arange(6, 12))
        arr = np.ones((2, 3, 4))
        assert analyzer.flatten_activation(arr).shape == (2, 12)

    def test_pca_low_rank(self):
        rng = np.random.default_rng(0)
        latent = rng.normal(size=(40, 2)) * np.array([5.0, 3.0])
        mixing = rng.normal(size=(2, 6))
        data = latent @ mixing + 1e-4 * rng.normal(size=(40, 6))

        res = ActivationAnalyzer().pca_analysis(data)
        assert res["components"].shape == (6, 6)
        assert res["transformed"].shape == (40, 6)
        assert res["n_components_90"] <= 2
        assert res["n_components_95"] <= 2
        assert np.all(np.diff(res["cumulative_variance"]) >= -1e-12)
        assert res["cumulative_variance"][-1] == pytest.approx(1.0)
        assert res["cumulative_variance"][1] > 0.999
        np.testing.assert_allclose(res["mean"], data.mean(axis=0))

    def test_pca_options(self):
        data = np.random.default_rng(1).normal(size=(10, 5))
        res = ActivationAnalyzer().pca_analysis(data, n_components=3, return_transformed=False)
        assert "transformed" not in res
        assert res["explained_variance_ratio"].shape == (3,)
        assert res["n_components_90"] <= 3
        assert res["n_components_95"] <= 3

    def test_kmeans_two_blobs(self):
        data = two_blobs()
        res = ActivationAnalyzer().cluster_analysis(data, method="kmeans", n_clusters=2, n_init=3)
        labels = res["labels"]
        assert res["n_clusters"] == 2
        assert len(set(labels[:10])) == 1 and len(set(labels[10:])) == 1
        assert labels[0] != labels[10]
        assert res["cluster_centers"].shape == (2, 3)
        assert res["silhouette_score"] > 0.9
        assert res["inertia"] < 1.0

    def test_dbscan_with_noise(self):
        data = np.vstack([two_blobs(), [[100.0, -100.0, 50.0]]])
        res = ActivationAnalyzer().cluster_analysis(data, method="dbscan", eps=1.0, min_samples=3)
        assert res["n_clusters"] == 2
        assert res["n_noise"] == 1
        assert res["labels"][-1] == -1
        assert res["silhouette_score"] > 0.9

    def test_dbscan_single_cluster_has_no_silhouette(self):
        data = two_blobs()[:10]
        res = ActivationAnalyzer().cluster_analysis(data, method="dbscan", eps=1.0, min_samples=3)
        assert res["n_clusters"] == 1
        assert res["n_noise"] == 0
        assert "silhouette_score" not in res

    def test_unknown_cluster_method(self):
        with pytest.raises(ValueError, match="Unknown clustering method"):
            ActivationAnalyzer().cluster_analysis(np.zeros((4, 2)), method="spectral")

    def test_tsne_embedding(self):
        data = two_blobs(n_per=8)
        analyzer = ActivationAnalyzer()
        res = analyzer.manifold_embedding(data, method="tsne")
        assert res["embedding"].shape == (16, 2)
        assert res["method"] == "tsne"
        assert res["params"] == {"perplexity": 15, "n_components": 2}

        res = analyzer.manifold_embedding(data, method="tsne", perplexity=4.0, max_iter=250)
        assert res["params"]["perplexity"] == 4.0
        emb = res["embedding"]
        # The two well separated blobs stay separated in the embedding.
        intra = np.linalg.norm(emb[:8] - emb[:8].mean(0), axis=1).mean()
        inter = np.linalg.norm(emb[:8].mean(0) - emb[8:].mean(0))
        assert inter > intra

    def test_umap_missing(self, monkeypatch):
        monkeypatch.setattr(activation_analysis, "HAS_UMAP", False)
        with pytest.raises(ImportError, match="UMAP not installed"):
            ActivationAnalyzer().manifold_embedding(np.zeros((5, 2)), method="umap")

    def test_umap_embedding(self):
        pytest.importorskip("umap")
        data = two_blobs(n_per=10)
        res = ActivationAnalyzer().manifold_embedding(data, method="umap", n_neighbors=5)
        assert res["embedding"].shape == (20, 2)
        assert res["params"] == {"n_neighbors": 5, "n_components": 2}

    def test_unknown_embedding_method(self):
        with pytest.raises(ValueError, match="Unknown embedding method"):
            ActivationAnalyzer().manifold_embedding(np.zeros((5, 2)), method="isomap")

    def test_singular_value_analysis_known_values(self):
        res = ActivationAnalyzer().singular_value_analysis(np.diag([3.0, 4.0]))
        np.testing.assert_allclose(res["singular_values"], [4.0, 3.0])
        np.testing.assert_allclose(res["normalized_sv"], [4 / 7, 3 / 7])
        assert res["stable_rank"] == pytest.approx(25 / 16)
        assert res["participation_ratio"] == pytest.approx(49 / 25)
        p = np.array([4 / 7, 3 / 7])
        assert res["spectral_entropy"] == pytest.approx(-(p * np.log(p)).sum())
        assert res["condition_number"] == pytest.approx(4 / 3)

    def test_singular_value_analysis_rank_deficient(self):
        res = ActivationAnalyzer().singular_value_analysis(np.array([[1.0, 0.0], [0.0, 0.0]]))
        assert res["condition_number"] == np.inf
        assert res["stable_rank"] == pytest.approx(1.0)
        assert res["spectral_entropy"] == pytest.approx(0.0)

    def test_compare_identical_activations(self):
        data = np.random.default_rng(2).normal(size=(30, 5))
        res = ActivationAnalyzer().compare_activations(data, data.copy())
        assert res["subspace_overlap"] == pytest.approx(1.0, abs=1e-6)
        assert res["dim_change"] == {
            "stable_rank": pytest.approx(0.0),
            "participation_ratio": pytest.approx(0.0),
            "n_components_90": 0,
        }
        np.testing.assert_allclose(res["principal_angles"], 0.0, atol=1e-3)

    def test_compare_dimensionality_increase(self):
        rng = np.random.default_rng(3)
        before = np.outer(rng.normal(size=40), rng.normal(size=6))  # rank 1
        before += 1e-6 * rng.normal(size=before.shape)
        after = rng.normal(size=(40, 6))  # full rank
        res = ActivationAnalyzer().compare_activations(before, after)
        assert res["dim_change"]["stable_rank"] > 1.0
        assert res["dim_change"]["n_components_90"] > 0
        assert np.all(res["principal_angles"] >= 0)
        assert np.all(res["principal_angles"] <= np.pi / 2 + 1e-9)
        assert 0.0 <= res["subspace_overlap"] <= 1.0 + 1e-9

    def test_neuron_importance(self):
        acts = np.array(
            [
                [0.0, 1.0, -3.0, 0.0],
                [0.0, 3.0, 3.0, 2.0],
                [0.0, 5.0, -3.0, 0.0],
            ]
        )
        analyzer = ActivationAnalyzer()
        var = analyzer.neuron_importance(acts, method="variance")
        np.testing.assert_allclose(var["scores"], np.var(acts, axis=0))
        assert var["ranking"][0] == 2  # variance 8
        np.testing.assert_array_equal(var["dead_neurons"], [0])
        assert var["n_dead"] == 1
        assert var["method"] == "variance"
        assert len(var["top_10_indices"]) == 4

        mean_abs = analyzer.neuron_importance(acts, method="mean_abs")
        np.testing.assert_allclose(mean_abs["scores"], [0.0, 3.0, 3.0, 2 / 3])

        sparsity = analyzer.neuron_importance(acts, method="sparsity")
        np.testing.assert_allclose(sparsity["scores"], [0.0, 1.0, 1.0, 1 / 3])

        with pytest.raises(ValueError, match="Unknown importance method"):
            analyzer.neuron_importance(acts, method="shapley")

    def test_activation_statistics(self):
        acts = np.array([[0.0, 2.0], [4.0, 0.0], [-2.0, 2.0]])
        stats = ActivationAnalyzer().activation_statistics(acts)
        assert stats == {
            "mean": pytest.approx(1.0),
            "std": pytest.approx(np.std(acts)),
            "min": -2.0,
            "max": 4.0,
            "sparsity": pytest.approx(2 / 6),
            "n_samples": 3,
            "n_features": 2,
        }


# ---------------------------------------------------------------------------
# FeatureVisualizer
# ---------------------------------------------------------------------------


class TestFeatureVisualizerHooks:
    def test_register_and_remove_hooks(self):
        model = nn.Sequential(nn.Linear(4, 3), nn.Tanh(), nn.Linear(3, 2))
        vis = FeatureVisualizer(model)
        vis.register_hooks({"fc1": model[0], "fc2": model[2]})
        assert len(vis.hooks) == 2

        x = torch.randn(5, 4)
        out = model(x)
        assert set(vis.activations) == {"fc1", "fc2"}
        torch.testing.assert_close(vis.activations["fc2"], out.detach())
        assert not vis.activations["fc1"].requires_grad

        out.sum().backward()
        assert set(vis.gradients) == {"fc1", "fc2"}
        torch.testing.assert_close(vis.gradients["fc2"], torch.ones(5, 2))

        vis.remove_hooks()
        assert vis.hooks == [] and vis.activations == {} and vis.gradients == {}
        assert len(model[0]._forward_hooks) == 0

    def test_activation_without_grad_registers_no_grad_hook(self):
        model = nn.Sequential(nn.Linear(2, 2))
        vis = FeatureVisualizer(model)
        vis.register_hooks({"fc": model[0]})
        with torch.no_grad():
            model(torch.randn(1, 2))
        assert "fc" in vis.activations
        assert vis.gradients == {}
        vis.remove_hooks()


FULL_HOOK_WARNING = "ignore:Full backward hook is firing:UserWarning"


@pytest.mark.filterwarnings(FULL_HOOK_WARNING)
class TestGradCAM:
    def test_highlights_bright_region(self):
        model, conv = tiny_cnn()
        vis = FeatureVisualizer(model)
        x = torch.zeros(1, 1, 6, 6)
        x[0, 0, 1:3, 3:5] = 1.0
        x[0, 0, 4, 0] = 0.5

        cam = vis.grad_cam(x, target_class=0, target_layer=conv)
        assert cam.shape == (6, 6)
        assert cam.min() == pytest.approx(0.0)
        assert cam.max() == pytest.approx(1.0)
        # With an identity conv and uniform pooled gradient, CAM is the normalized input.
        np.testing.assert_allclose(cam, x[0, 0].numpy(), atol=1e-6)
        # Hooks are cleaned up.
        assert len(conv._forward_hooks) == 0
        assert len(conv._backward_hooks) == 0

    def test_negative_class_gives_zero_map(self):
        model, conv = tiny_cnn()
        x = torch.rand(1, 1, 5, 5)
        cam = FeatureVisualizer(model).grad_cam(x, target_class=1, target_layer=conv)
        # Class 1 has a negative weight -> all contributions are clipped by ReLU.
        np.testing.assert_allclose(cam, 0.0)

    def test_one_dimensional_output(self):
        model, conv = tiny_cnn()
        x = torch.zeros(1, 1, 4, 4)
        x[0, 0, 0, 0] = 2.0
        cam = FeatureVisualizer(Flat0(model)).grad_cam(x, target_class=0, target_layer=conv)
        assert cam[0, 0] == pytest.approx(1.0)
        assert cam.sum() == pytest.approx(1.0)


class TestGradientAttributions:
    def test_saliency_linear_equals_weights(self):
        lin = linear_model()
        vis = FeatureVisualizer(lin)
        x = torch.randn(1, 4)
        sal = vis.saliency_map(x, target_class=1)
        np.testing.assert_allclose(sal, np.abs(lin.weight[1].detach().numpy()))
        signed = vis.saliency_map(x, target_class=0, absolute=False)
        np.testing.assert_allclose(signed, lin.weight[0].detach().numpy())
        assert x.grad is None  # input is cloned, not modified

    def test_saliency_multichannel_takes_channel_max(self):
        conv = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        with torch.no_grad():
            conv.weight.copy_(torch.tensor([1.0, -4.0, 2.0]).view(1, 3, 1, 1))
        model = nn.Sequential(conv, nn.Flatten(), nn.Linear(4, 1, bias=False))
        with torch.no_grad():
            model[2].weight.fill_(1.0)
        sal = FeatureVisualizer(model).saliency_map(torch.randn(1, 3, 2, 2), target_class=0)
        assert sal.shape == (2, 2)
        np.testing.assert_allclose(sal, 4.0)

    def test_saliency_one_dimensional_output(self):
        lin = linear_model()
        sal = FeatureVisualizer(Flat0(lin)).saliency_map(torch.randn(1, 4), target_class=2)
        np.testing.assert_allclose(sal, np.abs(lin.weight[2].detach().numpy()))

    def test_integrated_gradients_linear_is_exact(self):
        lin = linear_model()
        vis = FeatureVisualizer(lin)
        x = torch.tensor([[1.0, -2.0, 0.5, 3.0]])
        ig = vis.integrated_gradients(x, target_class=2, steps=8)
        expected = (x[0] * lin.weight[2]).detach().numpy()
        np.testing.assert_allclose(ig, expected, rtol=1e-5)
        # Completeness: attributions sum to f(x) - f(baseline).
        with torch.no_grad():
            delta = lin(x)[0, 2] - lin(torch.zeros_like(x))[0, 2]
        assert ig.sum() == pytest.approx(delta.item(), rel=1e-5)

    def test_integrated_gradients_custom_baseline_and_1d_output(self):
        lin = linear_model()
        x = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        baseline = torch.full_like(x, 0.5)
        ig = FeatureVisualizer(Flat0(lin)).integrated_gradients(
            x, target_class=0, baseline=baseline, steps=4
        )
        expected = ((x - baseline)[0] * lin.weight[0]).detach().numpy()
        np.testing.assert_allclose(ig, expected, rtol=1e-5)

    def test_layer_conductance_linear(self):
        fc1 = nn.Linear(4, 3)
        fc2 = nn.Linear(3, 2)
        model = nn.Sequential(fc1, fc2)
        x = torch.randn(2, 4)
        cond = FeatureVisualizer(model).layer_conductance(x, target_class=1, target_layer=fc1)
        # Score uses sample 0 only, so only its activations get gradient.
        with torch.no_grad():
            expected = (fc1(x)[0] * fc2.weight[1]).numpy()
        np.testing.assert_allclose(cond, expected, rtol=1e-5, atol=1e-6)
        assert len(fc1._forward_hooks) == 0

    def test_layer_conductance_one_dimensional_output(self):
        fc1 = nn.Linear(4, 3)
        fc2 = nn.Linear(3, 2)
        model = Flat0(nn.Sequential(fc1, fc2))
        x = torch.randn(1, 4)
        cond = FeatureVisualizer(model).layer_conductance(x, target_class=0, target_layer=fc1)
        with torch.no_grad():
            expected = (fc1(x)[0] * fc2.weight[0]).numpy()
        np.testing.assert_allclose(cond, expected, rtol=1e-5, atol=1e-6)


class TestOtherVisualizations:
    def test_attention_visualization(self):
        attn = torch.rand(1, 2, 3, 3)
        vis = FeatureVisualizer(nn.Identity())
        np.testing.assert_allclose(vis.attention_visualization(attn, head_idx=1), attn[0, 1])
        np.testing.assert_allclose(vis.attention_visualization(attn), attn[0].mean(dim=0))

    def test_feature_maps_top_k(self):
        act = torch.zeros(1, 4, 2, 2)
        act[0, 0] = torch.tensor([[0.0, 1.0], [2.0, 3.0]])  # mean |a| = 1.5
        act[0, 1] = 10.0 * torch.tensor([[0.0, 1.0], [0.0, 1.0]])  # 5.0
        act[0, 2] = -0.1  # 0.1
        act[0, 3] = torch.tensor([[4.0, 0.0], [0.0, 0.0]])  # 1.0
        vis = FeatureVisualizer(nn.Identity())
        maps = vis.feature_maps(act, top_k=2)
        assert len(maps) == 2
        np.testing.assert_allclose(maps[0], [[0.0, 1.0], [0.0, 1.0]], atol=1e-6)
        np.testing.assert_allclose(maps[1], [[0.0, 1 / 3], [2 / 3, 1.0]], atol=1e-6)
        assert len(vis.feature_maps(act, top_k=16)) == 4

    def test_activation_maximization_linear_is_exact(self):
        lin = linear_model()
        vis = FeatureVisualizer(lin)
        torch.manual_seed(123)
        start = torch.randn(1, 4)
        torch.manual_seed(123)
        result = vis.neuron_activation_maximization(lin, neuron_idx=1, input_shape=(4,), steps=5)
        # Gradient of a linear unit is its weight row, so each step adds lr * w.
        expected = start + 5 * 0.1 * lin.weight[1].detach()
        torch.testing.assert_close(result, expected)
        assert not result.requires_grad
        assert len(lin._forward_hooks) == 0

    def test_activation_maximization_conv_increases_activation(self):
        conv = nn.Conv2d(1, 2, kernel_size=3, padding=1)
        model = nn.Sequential(conv, nn.Flatten())
        vis = FeatureVisualizer(model)
        torch.manual_seed(7)
        start = torch.randn(1, 1, 4, 4)
        torch.manual_seed(7)
        result = vis.neuron_activation_maximization(
            conv, neuron_idx=1, input_shape=(1, 4, 4), steps=10, lr=0.05
        )
        with torch.no_grad():
            before = conv(start)[0, 1].mean()
            after = conv(result)[0, 1].mean()
        assert after > before

    @pytest.mark.filterwarnings(FULL_HOOK_WARNING)
    def test_compare_features(self, tmp_path):
        model, conv = tiny_cnn()
        before = tmp_path / "before.pt"
        after = tmp_path / "after.pt"
        torch.save({"model_state_dict": model.state_dict()}, before)
        flipped = {k: v.clone() for k, v in model.state_dict().items()}
        flipped["4.weight"] = -flipped["4.weight"]
        torch.save({"model_state_dict": flipped}, after)

        x = torch.zeros(1, 1, 4, 4)
        x[0, 0, 1, 1] = 1.0
        vis = FeatureVisualizer(model)

        same = vis.compare_features(x, 0, conv, str(before), str(before))
        np.testing.assert_allclose(same["cam_diff"], 0.0)
        assert same["mean_change"] == 0.0

        diff = vis.compare_features(x, 0, conv, str(before), str(after))
        np.testing.assert_allclose(diff["cam_before"], x[0, 0].numpy(), atol=1e-6)
        np.testing.assert_allclose(diff["cam_after"], 0.0)
        np.testing.assert_allclose(diff["cam_diff"], -x[0, 0].numpy(), atol=1e-6)
        assert diff["mean_change"] == pytest.approx(1 / 16)
