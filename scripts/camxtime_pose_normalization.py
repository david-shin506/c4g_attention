"""Camera normalization with an explicit stationary-center policy."""
import numpy as np
import torch

POLICY = {
    'version': 2,
    'moving': 'Normalize translations and near/far by maximum context baseline.',
    'stationary_centers': 'Use scale=1 when all requested camera centers have sub-threshold displacement; preserve all relative rotations and translations.',
    'invalid': 'Reject nonfinite poses, excessive baseline, or unseen translation within the dense requested span.',
}

def normalize_camxtime_poses(source_c2w, context_indices, requested_camera_indices, baseline_min, baseline_max):
    poses = torch.tensor(np.asarray(source_c2w) @ np.diag([1,-1,-1,1]), dtype=torch.float32)
    if not torch.isfinite(poses).all():
        raise ValueError('Nonfinite source camera poses')
    context_indices = list(context_indices)
    first = context_indices[0]
    baseline = (poses[context_indices[1:], :3, 3] - poses[first, :3, 3]).norm(dim=1).max()
    dense_baseline = (poses[list(requested_camera_indices), :3, 3] - poses[first, :3, 3]).norm(dim=1).max()
    if baseline < baseline_min:
        if dense_baseline >= baseline_min:
            raise ValueError(f'Input cameras have no baseline but requested cameras move: {dense_baseline}')
        scale = torch.tensor(1., dtype=poses.dtype)
        mode = 'stationary_centers_unit_scale'
    elif baseline <= baseline_max:
        scale = baseline
        mode = 'context_baseline'
    else:
        raise ValueError(f'Context baseline exceeds maximum: {baseline}')
    poses[:, :3, 3] /= scale
    poses = torch.linalg.inv(poses[first:first+1]) @ poses
    return poses, scale, {'mode': mode, 'original_context_baseline': float(baseline),
                          'original_requested_baseline': float(dense_baseline), 'applied_scale': float(scale)}
