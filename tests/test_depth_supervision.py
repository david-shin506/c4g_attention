import torch

from src.loss.depth_supervision import (
    StrictDepthLoss,
    build_pose_aligned_depth_labels,
    estimate_pose_scale,
)


def _identity_vggt_pose_encoding(camera_centers: torch.Tensor) -> torch.Tensor:
    """Build identity-rotation W2C pose encodings from camera centers."""

    pose = torch.zeros(*camera_centers.shape[:-1], 9)
    pose[..., :3] = -camera_centers
    pose[..., 6] = 1.0  # XYZW identity quaternion.
    pose[..., 7:] = 1.0
    return pose


def test_pose_aligned_labels_match_renderer_units() -> None:
    predicted_centers = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]]
    )
    dataset_centers = 3.0 * predicted_centers + torch.tensor([[[4.0, -2.0, 1.0]]])
    dataset_c2w = torch.eye(4).expand(1, 3, 4, 4).clone()
    dataset_c2w[..., :3, 3] = dataset_centers

    labels = build_pose_aligned_depth_labels(
        vggt_depth=torch.full((1, 3, 2, 2, 1), 2.0),
        vggt_depth_confidence=torch.ones(1, 3, 2, 2),
        vggt_pose_encoding=_identity_vggt_pose_encoding(predicted_centers),
        dataset_c2w=dataset_c2w,
        near=torch.full((1, 3), 0.5),
        far=torch.full((1, 3), 100.0),
        renderer_scale_invariant=True,
        confidence_quantile=0.0,
    )

    torch.testing.assert_close(labels.scale, torch.tensor([3.0]))
    assert labels.alignment_valid.tolist() == [True]
    assert labels.mask.all()
    torch.testing.assert_close(labels.depth, torch.full((1, 3, 2, 2), 12.0))


def test_static_camera_alignment_is_invalid() -> None:
    centers = torch.zeros(1, 3, 3)
    scale, error, valid = estimate_pose_scale(centers, centers)
    torch.testing.assert_close(scale, torch.ones(1))
    assert torch.isinf(error).all()
    assert valid.tolist() == [False]


def test_pose_scale_uses_well_separated_pairs() -> None:
    predicted = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [1.0, 0.0, 0.0]]]
    )
    dataset = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.10, 0.0, 0.0], [2.0, 0.0, 0.0]]]
    )
    scale, error, valid = estimate_pose_scale(predicted, dataset)

    assert valid.tolist() == [True]
    assert 1.8 < scale.item() < 2.1
    assert error.item() < 0.1


def test_strict_loss_does_not_refit_global_scale() -> None:
    target = torch.full((2, 4, 4), 5.0)
    mask = torch.ones_like(target, dtype=torch.bool)
    loss_fn = StrictDepthLoss("log_l1")

    matching_loss, _ = loss_fn(target.clone().requires_grad_(), target, mask)
    scaled_loss, diagnostics = loss_fn((target * 2).requires_grad_(), target, mask)

    torch.testing.assert_close(matching_loss, torch.tensor(0.0))
    torch.testing.assert_close(scaled_loss, torch.tensor(2.0).log())
    torch.testing.assert_close(
        diagnostics["median_prediction_ratio"], torch.tensor(2.0)
    )


def test_zero_prediction_is_penalized() -> None:
    target = torch.full((1, 2, 2), 5.0)
    prediction = torch.zeros_like(target, requires_grad=True)
    loss, diagnostics = StrictDepthLoss("log_l1")(
        prediction, target, torch.ones_like(target, dtype=torch.bool)
    )

    assert loss > 0
    loss.backward()
    assert prediction.grad is not None and prediction.grad.abs().sum() > 0
    torch.testing.assert_close(diagnostics["valid_fraction"], torch.tensor(1.0))
