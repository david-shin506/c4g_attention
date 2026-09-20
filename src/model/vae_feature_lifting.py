"""Train a feature value stream on frozen C4G attention (HG Instill architecture).

The original Gaussian decoder computes exactly the same x stream. Only the y
value projections, y FFNs, y normalization and 16-channel head are optimized.
"""
from contextlib import nullcontext
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint
from .encoder.common.gmae import FeedForward


class FeatureValueBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.to_anotherv = nn.Linear(dim, dim, bias=False)
        self.to_yout = nn.Linear(dim, dim)
        self.ff = FeedForward(dim, dim * 2)

    def forward(self, y, q, k):
        v = rearrange(self.to_anotherv(y), 'b n (h d) -> b h n d', h=q.shape[1])
        z = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        y = y + self.to_yout(rearrange(z, 'b h n d -> b n (h d)'))
        return y + self.ff(y)


class FrozenGeometryFeatureDecoder(nn.Module):
    def __init__(self, geometry, dim=2048, channels=16, gaussians_per_token=1):
        super().__init__()
        self.geometry = geometry.requires_grad_(False).eval()
        self.feature_blocks = nn.ModuleList([FeatureValueBlock(dim) for _ in geometry.layers])
        self.feature_norm = nn.LayerNorm(dim)
        self.feature_head = nn.Linear(dim, channels * gaussians_per_token)
        self.channels = channels
        self.gaussians_per_token = gaussians_per_token
        self.context_feature = None
        self.gaussian_slots = None
        self.features = []
        self.geometry_traces = []
        self.capture_geometry = False

    def forward(self, x, mask=None, context_feature=None, xpos=None):
        if mask is not None or xpos is not None:
            raise NotImplementedError('This feature path uses the frozen sinusoidal C4G decoder.')
        y = torch.cat([self.context_feature, self.gaussian_slots], dim=1)
        if x.shape != y.shape:
            raise ValueError(f'Patch/feature sequence mismatch: {x.shape} vs {y.shape}')
        for (attn, ff), feature_block in zip(self.geometry.layers, self.feature_blocks):
            with torch.no_grad():
                q, k, v = [rearrange(t, 'b n (h d) -> b h n d', h=attn.heads)
                           for t in attn.to_qkv(attn.norm(x)).chunk(3, dim=-1)]
                if attn.rope is not None:
                    raise NotImplementedError('Expected no RoPE in C4G geometry decoder.')
                z = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
                x = x + attn.to_out(rearrange(z, 'b h n d -> b n (h d)'))
                x = x + ff(x)
            if torch.is_grad_enabled():
                y = checkpoint(feature_block, y, q.detach(), k.detach(), use_reentrant=False)
            else:
                y = feature_block(y, q, k)
        raw = self.feature_head(self.feature_norm(y)[:, -self.gaussian_slots.shape[1]:])
        self.features.append(rearrange(raw, 'b n (g c) -> b (n g) c', g=self.gaussians_per_token, c=self.channels))
        if self.capture_geometry:
            self.geometry_traces.append(self.geometry.norm(x).detach())
        return self.geometry.norm(x)

    def trainable_parameters(self):
        yield from self.feature_blocks.parameters()
        yield from self.feature_norm.parameters()
        yield from self.feature_head.parameters()

    def feature_state_dict(self):
        return {k:v for k,v in self.state_dict().items() if not k.startswith('geometry.')}

    def load_feature_state_dict(self, state):
        missing, unexpected = self.load_state_dict(state, strict=False)
        if unexpected or any(not k.startswith('geometry.') for k in missing):
            raise ValueError(f'Feature checkpoint mismatch: {missing}, {unexpected}')


class VAEFeatureLifter(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder.requires_grad_(False).eval()
        self.decoder = FrozenGeometryFeatureDecoder(encoder.gmae_decoder,
            gaussians_per_token=encoder.cfg.gaussians_per_token)
        encoder.gmae_decoder = self.decoder

    def forward(self, context, context_latents, timestamps, step=45000):
        # 480x480 VAE latents are 60x60; match the 224/14=16 patch grid.
        b, v, c, _, _ = context_latents.shape
        h, w = context['image'].shape[-2:]
        patch_h, patch_w = h // self.encoder.patch_size, w // self.encoder.patch_size
        small = F.interpolate(context_latents.flatten(0, 1).float(), size=(patch_h, patch_w),
                              mode='bilinear', align_corners=False)
        tokens = rearrange(small, '(b v) c h w -> b (v h w) c', b=b, v=v)
        self.decoder.context_feature = F.pad(tokens, (0, 2048-c))
        self.decoder.gaussian_slots = self.encoder.gaussian_tokens.detach().unsqueeze(0).expand(b,-1,-1)
        self.decoder.features = []
        self.decoder.geometry_traces = []
        # Keep original encoder eval branch (no RGB training/checkpointing).
        self.encoder.eval()
        try:
            gaussians = self.encoder(context, step, target_timestamps=timestamps, visualization_dump=None)
            features = dict(zip(timestamps.tolist(), self.decoder.features))
            return gaussians, features
        finally:
            self.decoder.features = []
            self.decoder.context_feature = None
            self.decoder.gaussian_slots = None
