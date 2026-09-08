""" unit tests — prototype construction on synthetic features.

Related: tools/build_motion_prototypes.py
"""
from __future__ import annotations

import torch

from tools.build_motion_prototypes import (
    NUM_CLASS,
    _build_class_prototypes,
    _build_performer_prototypes,
    _compute_deviation_table,
)


def test_class_prototypes_centroid_is_mean() -> None:
    # 4 samples of class 0, 2 of class 1
    features = torch.tensor([
        [1.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [2.0, -1.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 12.0, 0.0],
    ])
    labels = torch.tensor([0, 0, 0, 0, 1, 1])
    mean, medoid = _build_class_prototypes(features, labels)
    assert mean.shape == (NUM_CLASS, 3)
    # Class 0 centroid = (2, 0, 0)
    assert torch.allclose(mean[0], torch.tensor([2.0, 0.0, 0.0]))
    # Class 1 centroid = (0, 11, 0)
    assert torch.allclose(mean[1], torch.tensor([0.0, 11.0, 0.0]))


def test_class_medoid_is_closest_real_sample() -> None:
    # Two class-0 samples, centroid = (2, 0); samples (1,0) and (3,0) both at
    # distance 1.0 — the first one in the list wins argmin (stable).
    features = torch.tensor([[1.0, 0.0], [3.0, 0.0], [10.0, 10.0]])
    labels = torch.tensor([0, 0, 1])
    mean, medoid = _build_class_prototypes(features, labels)
    # Class 0 has absolute idx 0 or 1; the rows 0 and 1 are equidistant (both
    # at distance 1.0 from the centroid (2, 0)), so argmin → 0.
    assert medoid[0].item() == 0
    # Class 1 has only sample index 2.
    assert medoid[1].item() == 2
    # Classes 2..11 have no samples: medoid = -1
    for c in range(2, NUM_CLASS):
        assert medoid[c].item() == -1


def test_performer_prototypes_group_by_id() -> None:
    features = torch.tensor([
        [1.0, 0.0], [3.0, 0.0],   # JP_06 average = (2, 0)
        [0.0, 4.0], [0.0, 6.0],   # JP_07 average = (0, 5)
    ])
    filenames = [
        "JP_06_joy_1_L",
        "JP_06_joy_1_M",
        "JP_07_anger_2_L",
        "JP_07_anger_2_H",
    ]
    perf_mean, perf_id_to_idx = _build_performer_prototypes(features, filenames)
    assert set(perf_id_to_idx.keys()) == {"JP_06", "JP_07"}
    jp06 = perf_mean[perf_id_to_idx["JP_06"]]
    jp07 = perf_mean[perf_id_to_idx["JP_07"]]
    assert torch.allclose(jp06, torch.tensor([2.0, 0.0]))
    assert torch.allclose(jp07, torch.tensor([0.0, 5.0]))


def test_performer_prototypes_skip_unparseable_names() -> None:
    features = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
    filenames = ["JP_06_joy_1_L", "not-a-diema-filename"]
    perf_mean, perf_id_to_idx = _build_performer_prototypes(features, filenames)
    # Only JP_06 survived; its prototype = the one sample we could parse.
    assert list(perf_id_to_idx.keys()) == ["JP_06"]
    assert torch.allclose(perf_mean[0], torch.tensor([1.0, 0.0]))


def test_deviation_table_shapes() -> None:
    torch.manual_seed(0)
    N, D = 10, 8
    features = torch.randn(N, D)
    labels = torch.randint(0, NUM_CLASS, (N,))
    filenames = [f"JP_0{i // 5}_joy_1_L" for i in range(N)]  # 2 performers
    class_mean, _ = _build_class_prototypes(features, labels)
    perf_mean, perf_id_to_idx = _build_performer_prototypes(features, filenames)
    dev = _compute_deviation_table(
        features, labels, filenames, class_mean, perf_mean, perf_id_to_idx,
    )
    assert dev["class_distance"].shape == (N,)
    assert dev["class_distances_all"].shape == (N, NUM_CLASS)
    assert dev["performer_distance"].shape == (N,)
    assert dev["residual_norm"].shape == (N,)
    assert dev["own_perf_idx"].shape == (N,)
    # Every sample has a valid performer (no unparseable names in this fixture).
    assert (dev["own_perf_idx"] >= 0).all()


def test_deviation_distance_to_own_class_is_nonnegative() -> None:
    torch.manual_seed(1)
    N, D = 6, 4
    features = torch.randn(N, D)
    labels = torch.arange(N) % NUM_CLASS
    filenames = [f"JP_0{i % 3}_joy_1_L" for i in range(N)]
    class_mean, _ = _build_class_prototypes(features, labels)
    perf_mean, perf_id_to_idx = _build_performer_prototypes(features, filenames)
    dev = _compute_deviation_table(
        features, labels, filenames, class_mean, perf_mean, perf_id_to_idx,
    )
    assert (dev["class_distance"] >= 0).all()
    assert (dev["performer_distance"] >= 0).all()
    assert (dev["residual_norm"] >= 0).all()
