"""Do the per-timestamp Gaussians actually move, and do they follow the GT scene flow?

For a fixed set of kubric / spring training batches (drawn once, shared by every checkpoint), decode the
Gaussians at every timestamp and compare token k's displacement between consecutive timestamps with the
GT 3D scene flow of the pixel it lands on, using the exact projection / visibility / scale conventions of
src/loss/loss_motion.py. Tokens are split into
    dynamic: visible on a (dilated) dynamic pixel in some view
    static:  the static-anchor set of the motion loss (seen, never visible on dynamic/unknown pixels)
and flow pairs into GT-moving (|f_gt| > MOVE_REL * scene scale) vs GT-still.

Usage (one GPU):
    python scripts/diag_gaussian_motion.py --out run_logs/diag_gaussian_motion.json \
        --ckpt name=path=training_cfg [...]
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize
from torch.utils.data import default_collate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_typed_root_config  # noqa: E402
from src.dataset import get_dataset  # noqa: E402
from dataclasses import replace  # noqa: E402

from src.loss.loss_motion import MotionLoss, MotionLossCfg, _dilate, _project, _sample, _unproject  # noqa: E402
from src.misc.step_tracker import StepTracker  # noqa: E402
from src.model.encoder import get_encoder  # noqa: E402

MOVE_REL = 0.005  # GT flow counts as motion above this fraction of the scene scale


def load_cfg(training_cfg: str):
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name="main", overrides=[f"+training={training_cfg}", "data_loader.train.num_workers=0"])
    return load_typed_root_config(cfg)


def collect_batches(cfg, per_dataset: int, step: int, seed: int) -> list[dict]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tracker = StepTracker()
    tracker.set_step(step)
    batches = []
    for ds in get_dataset(cfg.dataset, "train", tracker):
        name = getattr(ds, "cfg").name
        if name not in ("kubric", "spring"):
            continue
        got, tries = 0, 0
        while got < per_dataset and tries < per_dataset * 20:
            tries += 1
            try:
                ex = ds[random.randrange(len(ds))]
            except Exception as exc:  # spring skips scenes with a degenerate baseline
                print(f"  skip {name}: {exc}", flush=True)
                continue
            if "motion_flow" not in ex["context"]:
                continue
            batches.append(default_collate([ex]))
            got += 1
        print(f"collected {got} {name} batches", flush=True)
    return batches


def to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    return x


@torch.no_grad()
def analyse(means: dict[int, torch.Tensor], views: dict, mcfg: MotionLossCfg) -> dict | None:
    c2w, K, index, camera = views["extrinsics"], views["intrinsics"], views["index"], views["camera"]
    V = c2w.shape[0]
    H, W = views["depth"].shape[-2:]
    timestamps = sorted(int(t) for t in torch.unique(index).tolist())
    if len(timestamps) < 2:
        return None
    t0 = timestamps[0]
    nxt = {t: timestamps[i + 1] for i, t in enumerate(timestamps[:-1])}
    next_view = [-1] * V
    for v in range(V):
        t = int(index[v])
        if t in nxt:
            hit = ((index == nxt[t]) & (camera == camera[v])).nonzero()
            if hit.numel():
                next_view[v] = int(hit[0, 0])

    proj = []
    for v in range(V):
        p, z, inb = _project(means[int(index[v])], c2w[v], K[v], (H, W))
        d = _sample(views["depth"][v], p)
        proj.append({"p": p, "z": z, "d": d, "valid": inb & (d > 0)})
    ratios = torch.cat([(e["z"][e["valid"]] / e["d"][e["valid"]]) for e in proj])
    if ratios.numel() < mcfg.min_tokens:
        return None
    scale = float(ratios.median().clamp(1e-3, 1e3))
    vis = [e["valid"] & ((e["z"] - e["d"] * scale).abs() <= mcfg.visibility_rel_tol * e["d"] * scale) for e in proj]
    ref = torch.cat([e["d"][x] for e, x in zip(proj, vis)]) * scale
    if ref.numel() < mcfg.min_tokens:
        return None
    scene_scale = float(ref.median().clamp(min=1e-4))

    N = means[t0].shape[0]
    seen = torch.zeros(N, dtype=torch.bool, device=c2w.device)
    static_ok = torch.ones_like(seen)
    dynamic = torch.zeros_like(seen)
    for v in range(V):
        m = views["motion_mask"][v]
        dyn = _dilate((m > 0.5).float(), mcfg.mask_dilation) > 0.5
        hit_bad = _sample((dyn | (m < 0)).float(), proj[v]["p"]) > 0.5
        hit_dyn = _sample(dyn.float(), proj[v]["p"]) > 0.5
        seen |= vis[v]
        static_ok &= ~(vis[v] & hit_bad)
        dynamic |= vis[v] & hit_dyn
    static = seen & static_ok

    # Largest excursion from the first-timestamp position, per token.
    stack = torch.stack([means[t] for t in timestamps])  # [T, N, 3]
    excursion = (stack - stack[0:1]).norm(dim=-1).amax(0) / scene_scale

    moving_ratio, moving_cos, moving_gt, still_pred = [], [], [], []
    for v in range(V):
        vn = next_view[v]
        if vn < 0:
            continue
        sel = vis[v] & (_sample(views["motion_valid"][v], proj[v]["p"]) > 0.5)
        if sel.sum() < mcfg.min_tokens:
            continue
        t, tn = int(index[v]), int(index[vn])
        p = proj[v]["p"][sel]
        f2 = _sample(views["motion_flow"][v].permute(2, 0, 1), p)
        d = proj[v]["d"][sel] * scale
        dn = _sample(views["motion_depth_next"][v], p) * scale
        ok = dn > 0
        X = _unproject(p, d, c2w[v], K[v], (H, W))
        Xn = _unproject(p + f2, dn, c2w[vn], K[vn], (H, W))
        f_gt = ((Xn - X) / scene_scale)[ok]
        f_pred = ((means[tn][sel] - means[t][sel]) / scene_scale)[ok]
        gt_mag = f_gt.norm(dim=-1)
        pred_mag = f_pred.norm(dim=-1)
        mv = gt_mag > MOVE_REL
        if mv.any():
            moving_gt.append(gt_mag[mv])
            moving_ratio.append(pred_mag[mv] / gt_mag[mv])
            moving_cos.append(torch.nn.functional.cosine_similarity(f_pred[mv], f_gt[mv], dim=-1))
        if (~mv).any():
            still_pred.append(pred_mag[~mv])

    def med(xs):
        return float(torch.cat(xs).median()) if xs else float("nan")

    return {
        "n_dynamic_tokens": int(dynamic.sum()),
        "n_static_tokens": int(static.sum()),
        "excursion_dynamic_med": float(excursion[dynamic].median()) if dynamic.any() else float("nan"),
        "excursion_static_med": float(excursion[static].median()) if static.any() else float("nan"),
        "excursion_all_med": float(excursion.median()),
        "n_moving_pairs": int(sum(x.numel() for x in moving_gt)),
        "moving_gt_mag_med": med(moving_gt),
        "moving_pred_over_gt_med": med(moving_ratio),
        "moving_cos_med": med(moving_cos),
        "still_pred_mag_med": med(still_pred),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", action="append", required=True, help="name=path=training_cfg")
    parser.add_argument("--per-dataset", type=int, default=12)
    parser.add_argument("--step", type=int, default=6000, help="step tracker value for the view-sampler warm-up")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    device = torch.device("cuda")

    specs = [c.split("=", 2) for c in args.ckpt]
    base_cfg = load_cfg(specs[0][2])
    batches = collect_batches(base_cfg, args.per_dataset, args.step, args.seed)
    mcfg = base_cfg.train.motion if hasattr(base_cfg.train, "motion") else MotionLossCfg()

    results = {}
    for name, path, training_cfg in specs:
        cfg = load_cfg(training_cfg)
        encoder, _ = get_encoder(cfg.model.encoder)
        sd = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        step = int(sd.get("global_step", -1))
        state = {k[len("encoder."):]: v for k, v in sd["state_dict"].items() if k.startswith("encoder.")}
        missing, unexpected = encoder.load_state_dict(state, strict=False)
        print(f"[{name}] step {step} ({cfg.model.encoder.normalize_type}/p{cfg.model.encoder.sinusoidal_period}) "
              f"missing {len(missing)} unexpected {len(unexpected)}", flush=True)
        del sd, state
        encoder = encoder.to(device).eval()
        shim = encoder.get_data_shim()
        per_ds: dict[str, list[dict]] = {}
        for batch in batches:
            batch = to_device(shim({k: (dict(v) if isinstance(v, dict) else v) for k, v in batch.items()}), device)
            ds = batch["dataset_name"][0]
            ts = torch.unique(torch.cat([batch["context"]["index"], batch["target"]["index"]], dim=1))
            gaussians = encoder(batch["context"], step, target_timestamps=ts)
            means = {t: g.means[0].float() for t, g in gaussians.items()}
            views = {
                k: torch.cat([batch["context"][k][0], batch["target"][k][0]])
                for k in ("extrinsics", "intrinsics", "index", "camera", "depth", "motion_flow",
                          "motion_depth_next", "motion_mask", "motion_valid")
            }
            views = {k: v.float() if v.is_floating_point() else v for k, v in views.items()}
            r = analyse(means, views, mcfg)
            if r is not None:
                # v2 loss terms as the training code computes them (unit weights, all pairs).
                v2 = MotionLoss(replace(mcfg, flow3d_weight=0.0, flow2d_weight=0.0, static_weight=0.0,
                                        flow3d_move_weight=1.0, flow3d_still_weight=1.0))
                views_l = {**views, "image": torch.cat([batch["context"]["image"][0], batch["target"]["image"][0]])}
                losses, stats = v2(means, views_l, ds, 10**6)
                r["v2_flow3d_move"] = float(losses["flow3d_move"]) if "flow3d_move" in losses else float("nan")
                r["v2_flow3d_still"] = float(losses["flow3d_still"]) if "flow3d_still" in losses else float("nan")
                r["v2_move_ratio"] = stats.get("flow3d_move_ratio", float("nan"))
                r["v2_move_cos"] = stats.get("flow3d_move_cos", float("nan"))
                per_ds.setdefault(ds, []).append(r)
        summary = {"step": step}
        for ds, rows in per_ds.items():
            summary[ds] = {k: float(np.nanmedian([row[k] for row in rows])) for k in rows[0]}
            summary[ds]["n_batches"] = len(rows)
        results[name] = summary
        print(json.dumps({name: summary}, indent=1), flush=True)
        del encoder
        torch.cuda.empty_cache()

    Path(args.out).write_text(json.dumps(results, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
