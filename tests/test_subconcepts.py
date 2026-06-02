import torch
import pytest

from fannar.concepts import (
    collect_subconcept_records,
    gram_metric_usage,
    metric_score_and_coords,
    normalized_delta,
    top_metric_patches,
)


def test_collect_subconcept_records_from_metric_scores():
    activations = torch.ones(2, 3, 4, 4)
    activations[:, 0] = torch.arange(16).reshape(4, 4)
    metric = torch.eye(3)

    records = collect_subconcept_records(activations, metric)

    assert records
    assert all(len(record) == 5 for record in records)


def test_top_metric_patches_filters_global_high_scores():
    images = torch.rand(3, 3, 16, 16)
    activations = torch.rand(3, 3, 4, 4)
    activations[0, :, 2, 2] += 5.0
    metric = torch.eye(3)

    patches = top_metric_patches(
        images,
        activations,
        metric,
        max_patches=4,
        crop_size=8,
        min_quantile=0.85,
        max_per_image=2,
    )

    assert 0 < len(patches) <= 4
    assert all(patch.shape == (3, 8, 8) for patch in patches)


def test_metric_energy_aligns_dtype():
    gram = torch.eye(3, dtype=torch.float64)
    field = torch.rand(16, 3, dtype=torch.float32)

    usage = gram_metric_usage(gram, field, (4, 4))
    score, coords = metric_score_and_coords(gram, field, (4, 4))
    delta = normalized_delta(usage.used_similarity, usage.used_similarity.double())

    assert usage.surface.field.dtype == field.dtype
    assert score.dtype == field.dtype
    assert coords.dtype == field.dtype
    assert delta.dtype == usage.used_similarity.dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sem CUDA")
def test_metric_energy_aligns_device_from_mixed_inputs():
    gram = torch.eye(3, device="cuda")
    field = torch.rand(16, 3)

    usage = gram_metric_usage(gram, field, (4, 4))
    score, coords = metric_score_and_coords(gram, field, (4, 4))
    assert usage.surface.field.device == field.device
    assert usage.used_similarity.device == field.device
    assert score.device == field.device
    assert coords.device == field.device
