from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from .croco.misc import freeze_all_params
from .vggt_omega.models.aggregator import Aggregator
from .vggt_omega.models.heads.dense_head import DenseHead


@dataclass
class BackboneVGGTOmegaCfg:
    name: Literal["vggt_omega_multi"]


class BackboneVGGTOmega(nn.Module):
    """Expose VGGT-Omega through the token interface used by EncoderVGGT."""

    def __init__(
        self,
        cfg: BackboneVGGTOmegaCfg,
        d_in: int,
        gradient_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        if d_in != 3:
            raise ValueError(f"VGGT-Omega expects RGB input, got {d_in} channels")

        self.cfg = cfg
        self.aggregator = Aggregator(
            patch_size=16,
            embed_dim=1024,
            gradient_checkpoint=gradient_checkpoint,
        )
        self.dense_head = DenseHead(dim_in=2048, patch_size=16)

        self.dec_depth = self.aggregator.depth - 1
        self.enc_embed_dim = 2048
        self.dec_embed_dim = 2048

    def set_freeze(self, freeze: str) -> None:
        if freeze not in {"none", "encoder"}:
            raise ValueError(f"Unsupported VGGT-Omega freeze mode: {freeze}")
        if freeze == "encoder":
            freeze_all_params([self.aggregator])

    def forward(
        self,
        context: dict,
        symmetrize_batch: bool = False,
        return_views: bool = False,
    ):
        del symmetrize_batch, return_views
        images = context["image"]
        b, v, _, h, w = images.shape
        if h % self.patch_size or w % self.patch_size:
            raise ValueError(
                "VGGT-Omega input dimensions must be divisible by "
                f"{self.patch_size}, got {(h, w)}"
            )

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        shape = torch.tensor((h, w), device=images.device).expand(b, v, 2)
        return aggregated_tokens_list, shape, patch_start_idx

    def predict_depth(self, tokens, images, patch_start_idx: int):
        return self.dense_head(tokens, images, patch_start_idx)

    @property
    def patch_size(self) -> int:
        return self.aggregator.patch_size

    @property
    def d_out(self) -> int:
        return 2048
