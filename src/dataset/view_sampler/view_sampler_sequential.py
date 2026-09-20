from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float, Int64
from torch import Tensor

from ...misc.step_tracker import StepTracker
from ..types import Stage
from .view_sampler import ViewSampler


@dataclass
class ViewSamplerSequentialCfg:
    name: Literal["sequential"]
    num_context_views: int
    gap: int = 6


class ViewSamplerSequential(ViewSampler[ViewSamplerSequentialCfg]):
    """Sequential view sampler for Spring test.

    Samples context views starting from index 0 with a fixed gap.
    Target views are placed at the midpoints between consecutive context views.

    Example (num_context_views=8, gap=6):
      context: [0, 6, 12, 18, 24, 30, 36, 42]
      target:  [3, 9, 15, 21, 27, 33, 39]
    """

    def __init__(
        self,
        cfg: ViewSamplerSequentialCfg,
        stage: Stage,
        is_overfitting: bool,
        cameras_are_circular: bool,
        step_tracker: StepTracker | None,
    ) -> None:
        super().__init__(cfg, stage, is_overfitting, cameras_are_circular, step_tracker)

    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
    ) -> tuple[
        Int64[Tensor, " context_view"],
        Int64[Tensor, " target_view"],
        Float[Tensor, " overlap"],
    ]:
        num_views = extrinsics.shape[0]
        gap = self.cfg.gap

        # Context: start from 0, step by gap
        index_context = [i * gap for i in range(self.cfg.num_context_views)]

        # Check bounds
        if index_context[-1] >= num_views:
            raise ValueError(
                f"Not enough frames: need index {index_context[-1]} but scene has {num_views} frames."
            )

        # Target: midpoints between consecutive context frames
        index_target = [index_context[i] + gap // 2 for i in range(len(index_context) - 1)]

        # Filter out-of-bounds targets
        index_target = [t for t in index_target if t < num_views]

        overlap = torch.tensor([0.5], device=device)
        return (
            torch.tensor(index_context, dtype=torch.int64, device=device),
            torch.tensor(index_target, dtype=torch.int64, device=device),
            overlap,
        )

    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_context_views - 1
