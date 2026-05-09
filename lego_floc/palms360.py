import math
import numpy as np
import torch
import torch.nn.functional as F

def wrap_to_pi(theta):
    return (theta + math.pi) % (2 * math.pi) - math.pi


def depth_profiles_for_map_matching(profiles, meta):
    profiles = np.asarray(profiles, dtype=np.float32)
    if meta.get('oracle_depth40_order') == 'image_left_to_right':
        return np.flip(profiles, axis=-1).copy()
    return profiles


def localize_palms_frame(desdf, rays, orn_slice=36, return_np=True, lambd=40.0, flip_query=False):
    query = torch.as_tensor(rays, dtype=torch.float32)
    if flip_query:
        query = torch.flip(query, [0])

    O = desdf.shape[2]
    V = query.shape[0]
    query = query.reshape((1, 1, -1))
    pad_front = V // 2
    pad_back = V - pad_front
    pad_desdf = F.pad(torch.as_tensor(desdf, dtype=torch.float32), [pad_front, pad_back], mode='circular')

    prob_vol = torch.stack(
        [-torch.norm(pad_desdf[:, :, i:i + V] - query, p=1.0, dim=2) for i in range(O)],
        dim=2,
    )
    prob_vol = torch.exp(prob_vol / float(lambd))
    prob_dist, orientations = torch.max(prob_vol, dim=2)
    pred_y, pred_x = torch.where(prob_dist == prob_dist.max())
    pred_y = pred_y[0:1]
    pred_x = pred_x[0:1]
    orn = orientations[pred_y, pred_x] / orn_slice * 2 * torch.pi
    pred = torch.cat((pred_x.float(), pred_y.float(), orn.float()))

    if return_np:
        return (
            prob_vol.detach().cpu().numpy(),
            prob_dist.detach().cpu().numpy(),
            orientations.detach().cpu().numpy(),
            pred.detach().cpu().numpy(),
        )
    return (
        prob_vol.to(torch.float32).detach().cpu(),
        prob_dist.to(torch.float32).detach().cpu(),
        orientations.to(torch.float32).detach().cpu(),
        pred.to(torch.float32).detach().cpu(),
    )


def localize_360(desdf, descriptor, lambd=40.0, flip_query=False):
    query = torch.tensor(descriptor, dtype=torch.float32)
    if flip_query:
        query = torch.flip(query, [0])
    valid = torch.isfinite(query)
    if not torch.any(valid):
        raise RuntimeError('No valid bins in session descriptor')
    query = query.reshape(1, 1, -1)

    V = query.shape[2]
    O = desdf.shape[2]
    pad_front = V // 2
    pad_back = V - pad_front
    pad = F.pad(torch.tensor(desdf), [pad_front, pad_back], mode='circular')
    prob = torch.stack(
        [-torch.norm(pad[:, :, i:i + V][:, :, valid] - query[:, :, valid], p=1.0, dim=2) for i in range(O)],
        dim=2,
    )
    prob = torch.exp(prob / lambd)
    prob_dist, orns = torch.max(prob, dim=2)
    py, px = torch.where(prob_dist == prob_dist.max())
    py = py[0]
    px = px[0]
    orn = orns[py, px] / 36 * 2 * math.pi
    return np.array([float(px), float(py), float(orn)])


def build_descriptor_360(profiles, centers, fwidths, fill_missing=True):
    descriptor = np.full((36,), np.nan, dtype=np.float32)
    for j in range(36):
        angle = math.radians(j * 10.0)
        samples = []
        for center, profile, f_w in zip(centers, profiles, fwidths):
            rel = wrap_to_pi(angle - center)
            half_fov = math.atan(1.0 / (2.0 * f_w))
            if abs(rel) <= half_fov:
                w = math.tan(rel) * len(profile) * f_w + (len(profile) - 1) / 2.0
                if 0 <= w <= len(profile) - 1:
                    x0 = int(math.floor(w))
                    x1 = min(x0 + 1, len(profile) - 1)
                    t = w - x0
                    depth = (1.0 - t) * profile[x0] + t * profile[x1]
                    ray = depth / max(math.cos(rel), 1e-6)
                    samples.append(ray)
        if samples:
            descriptor[j] = float(np.median(samples))

    valid = np.isfinite(descriptor)
    if not np.any(valid):
        raise RuntimeError('No valid bins in session descriptor')
    if not fill_missing:
        return descriptor

    valid_idx = np.where(valid)[0]
    for j in range(36):
        if not valid[j]:
            nearest = min(valid_idx, key=lambda idx: min((idx - j) % 36, (j - idx) % 36))
            descriptor[j] = descriptor[nearest]
    return descriptor


def estimate_angular_coverage(centers, fwidths):
    intervals = []
    for center, f_w in zip(centers, fwidths):
        half_fov = math.atan(1.0 / (2.0 * float(f_w)))
        start = (float(center) - half_fov) % (2.0 * math.pi)
        end = (float(center) + half_fov) % (2.0 * math.pi)
        if start <= end:
            intervals.append((start, end))
        else:
            intervals.append((start, 2.0 * math.pi))
            intervals.append((0.0, end))

    if not intervals:
        return 0.0
    intervals.sort()
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return float(sum(end - start for start, end in merged))


def select_palms_360_subset_indices(centers, fwidths, start_idx=0, coverage_eps=1e-5):
    """PALMS-style greedy subset selection for continuous 360-degree coverage."""
    centers = np.asarray(centers, dtype=np.float32)
    fwidths = np.asarray(fwidths, dtype=np.float32)
    if centers.size == 0:
        return np.zeros((0,), dtype=np.int64), False, 0.0
    if not 0 <= int(start_idx) < centers.size:
        raise ValueError(f'start_idx out of range: {start_idx}')

    yaws = centers % (2.0 * math.pi)
    start_yaw = float(yaws[int(start_idx)])
    normalized_yaws = (yaws - start_yaw) % (2.0 * math.pi)
    sorted_indices = np.argsort(normalized_yaws)
    sorted_yaws = normalized_yaws[sorted_indices]
    half_fovs = np.array([math.atan(1.0 / (2.0 * float(f_w))) for f_w in fwidths], dtype=np.float32)
    sorted_half_fovs = half_fovs[sorted_indices]

    selected = []
    covered_until = -float(coverage_eps)
    i = 0
    target = 2.0 * math.pi - float(coverage_eps)
    while covered_until < target:
        best_i = None
        best_cover = -float('inf')
        for j in range(i, len(sorted_yaws)):
            yaw = float(sorted_yaws[j])
            half_fov = float(sorted_half_fovs[j])
            start = yaw - half_fov
            end = yaw + half_fov
            if start > covered_until + float(coverage_eps):
                break
            if end > best_cover:
                best_cover = end
                best_i = j

        if best_i is None:
            break
        selected.append(int(sorted_indices[best_i]))
        covered_until = best_cover
        i = best_i + 1

    coverage_complete = bool(covered_until >= target)
    return np.asarray(selected, dtype=np.int64), coverage_complete, float(max(0.0, min(covered_until, 2.0 * math.pi)))


def select_partial_uniform_indices(centers, num_frames=3, start_idx=0):
    centers = np.asarray(centers, dtype=np.float32)
    if centers.size == 0:
        return np.zeros((0,), dtype=np.int64)
    if centers.size <= num_frames:
        return np.arange(centers.size, dtype=np.int64)
    if not 0 <= int(start_idx) < centers.size:
        raise ValueError(f'start_idx out of range: {start_idx}')

    yaws = centers % (2.0 * math.pi)
    start_yaw = float(yaws[int(start_idx)])
    normalized_yaws = (yaws - start_yaw) % (2.0 * math.pi)
    targets = np.linspace(0.0, 2.0 * math.pi, num_frames, endpoint=False)

    selected = []
    for target in targets:
        best_idx = None
        best_dist = float('inf')
        for idx, yaw in enumerate(normalized_yaws):
            if idx in selected:
                continue
            dist = abs(wrap_to_pi(float(yaw) - float(target)))
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        selected.append(int(best_idx))
    return np.asarray(selected, dtype=np.int64)


def parse_partial_uniform_policy(frame_policy):
    if not frame_policy.startswith('partial_') or not frame_policy.endswith('_uniform'):
        return None
    parts = frame_policy.split('_')
    if len(parts) != 3:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def select_session360_frame_data(profiles, centers, fwidths, frame_policy='all', subset_fallback='all'):
    profiles = np.asarray(profiles, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    fwidths = np.asarray(fwidths, dtype=np.float32)
    total_frames = int(len(centers))
    all_indices = np.arange(total_frames, dtype=np.int64)
    if frame_policy == 'all':
        return profiles, centers, fwidths, all_indices, {
            'frame_policy': 'all',
            'coverage_complete': True,
            'coverage_radians': float(2.0 * math.pi),
            'frames_used': total_frames,
            'total_frames': total_frames,
            'frame_indices': all_indices.tolist(),
        }
    partial_num_frames = parse_partial_uniform_policy(frame_policy)
    if partial_num_frames is not None:
        indices = select_partial_uniform_indices(centers, num_frames=partial_num_frames)
        coverage_radians = estimate_angular_coverage(centers[indices], fwidths[indices])
        return profiles[indices], centers[indices], fwidths[indices], indices, {
            'frame_policy': frame_policy,
            'coverage_complete': False,
            'coverage_radians': coverage_radians,
            'frames_used': int(len(indices)),
            'total_frames': total_frames,
            'frame_indices': indices.tolist(),
        }
    if frame_policy != 'palms_subset':
        raise ValueError(f'Unknown session360 frame_policy: {frame_policy}')

    indices, coverage_complete, coverage_radians = select_palms_360_subset_indices(centers, fwidths)
    if not coverage_complete:
        if subset_fallback == 'all':
            return profiles, centers, fwidths, all_indices, {
                'frame_policy': 'palms_subset_fallback_all',
                'coverage_complete': False,
                'coverage_radians': coverage_radians,
                'frames_used': total_frames,
                'total_frames': total_frames,
                'frame_indices': all_indices.tolist(),
                'subset_frame_indices': indices.tolist(),
            }
        raise RuntimeError(
            f'PALMS subset failed to cover 360 degrees: '
            f'{math.degrees(coverage_radians):.2f} deg covered with {len(indices)} frames'
        )

    return profiles[indices], centers[indices], fwidths[indices], indices, {
        'frame_policy': 'palms_subset',
        'coverage_complete': True,
        'coverage_radians': coverage_radians,
        'frames_used': int(len(indices)),
        'total_frames': total_frames,
        'frame_indices': indices.tolist(),
    }


