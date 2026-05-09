import numpy as np
import torch
import torch.nn.functional as F


def get_ray_from_depth(depth, V=11, dv=10, a0=None, F_W=3 / 8):
    depth = np.asarray(depth, dtype=np.float32)
    width = depth.shape[0]
    angles = (np.arange(V, dtype=np.float32) - (V - 1) / 2.0) * float(dv) / 180.0 * np.pi
    center = (width - 1) / 2.0 if a0 is None else float(a0)
    xs = np.arange(width, dtype=np.float32)
    query = np.tan(angles) * width * float(F_W) + center
    sampled = np.interp(query, xs, depth, left=depth[0], right=depth[-1]).astype(np.float32)
    return sampled / np.cos(angles)


def localize(desdf, rays, orn_slice=36, return_np=True, lambd=40.0):
    if not torch.is_tensor(desdf):
        desdf = torch.as_tensor(desdf, dtype=torch.float32)
    if not torch.is_tensor(rays):
        rays = torch.as_tensor(rays, dtype=torch.float32, device=desdf.device)
    else:
        rays = rays.to(device=desdf.device, dtype=torch.float32)
    rays = torch.flip(rays, [0]).reshape(1, 1, -1)
    orientation_count = desdf.shape[2]
    ray_count = rays.shape[2]
    pad_front = ray_count // 2
    pad_back = ray_count - pad_front
    pad_desdf = F.pad(desdf.to(torch.float32), [pad_front, pad_back], mode='circular')
    windows = pad_desdf.unfold(2, ray_count, 1)[:, :, :orientation_count, :]
    prob_vol = torch.exp(-(windows - rays.unsqueeze(2)).abs().sum(dim=3) / float(lambd))
    prob_dist, orientations = torch.max(prob_vol, dim=2)
    pred_y, pred_x = torch.where(prob_dist == prob_dist.max())
    pred_y = pred_y[0:1]
    pred_x = pred_x[0:1]
    orn = orientations[pred_y, pred_x] / float(orn_slice) * 2.0 * torch.pi
    pred = torch.cat((pred_x.float(), pred_y.float(), orn.float()))
    if return_np:
        return tuple(x.detach().cpu().numpy() for x in (prob_vol, prob_dist, orientations, pred))
    return prob_vol, prob_dist, orientations, pred
