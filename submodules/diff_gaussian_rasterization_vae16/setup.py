"""Isolated 16-channel build; does not replace the RGB or tracking extensions."""
from pathlib import Path
from torch.utils.cpp_extension import load
root = Path(__file__).resolve().parent
_C = load(name="c4g_vae_feature16", sources=[str(root / p) for p in [
    "cuda_rasterizer/rasterizer_impl.cu", "cuda_rasterizer/forward.cu",
    "cuda_rasterizer/backward.cu", "rasterize_points.cu", "ext.cpp"]],
    extra_cuda_cflags=["-I" + str(root / "third_party/glm")])
