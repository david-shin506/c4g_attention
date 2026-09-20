# CamxTime export decoder compatibility

`decoder_splatting_cuda.py` is a byte-for-byte copy of the decoder used to create the existing CamxTime_480x480 dataset. SHA256: `ac300cd5a49e0356698a3204703771be2fb9f69a874219f8966d7ff748ad9c70`.

The shared decoder acquired a different low-pass schedule after the export began. The export resume guard correctly rejected the changed source. `export_camxtime_vace.py` now uses this isolated decoder, preserving the original dataset specification without reverting shared training changes. Its relative imports resolve under `src.model.decoder` via importlib.

Restart verification (2026-09-15): saved Gaussians and aligned poses at t22, t38, t54 reproduced original target PNGs with maximum uint8 difference 1 and mean difference at most 0.000025. See `outputs/dataset_generation_ctx17/restart_decoder_validation.json`. The checkpoint/config/encoder/camera-update/alignment hashes remained unchanged. The CUDA rasterizer remains shared and was exercised by this verification.
