"""
Self-P2IR's PointNet++ operations in plain PyTorch, so the real-frame network runs without compiling
their CUDA extension (pointnet2_cuda, written for PyTorch 1.10).

Drop-in for models/pointnet/pointnet2/pointnet2_utils.py. Each function follows the CUDA kernel in
models/pointnet/pointnet2/src/ line by line:
  furthest_point_sample  starts at point 0; each next point is the one farthest from those chosen
  ball_query             the first `nsample` points (in index order) closer than `radius`; a ball with
                         fewer points is filled with its first point; an empty ball is all zeros
  three_nn               the 3 nearest points and their distances (not squared)
gather / grouping / three_interpolate are index look-ups.
"""
import torch
import torch.nn as nn


def furthest_point_sample(xyz, npoint):
    """xyz (B, N, 3) -> indices (B, npoint)."""
    B, N, _ = xyz.shape
    idx = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    dist = torch.full((B, N), 1e10, device=xyz.device)
    last = torch.zeros(B, dtype=torch.long, device=xyz.device)
    rows = torch.arange(B, device=xyz.device)
    for j in range(1, npoint):
        dist = torch.minimum(dist, ((xyz - xyz[rows, last][:, None]) ** 2).sum(-1))
        last = dist.argmax(dim=1)
        idx[:, j] = last
    return idx


def gather_operation(features, idx):
    """features (B, C, N), idx (B, npoint) -> (B, C, npoint)."""
    return torch.gather(features, 2, idx.long()[:, None].expand(-1, features.shape[1], -1))


def grouping_operation(features, idx):
    """features (B, C, N), idx (B, npoint, nsample) -> (B, C, npoint, nsample)."""
    B, npoint, nsample = idx.shape
    return gather_operation(features, idx.reshape(B, -1)).reshape(B, -1, npoint, nsample)


def ball_query(radius, nsample, xyz, new_xyz):
    """xyz (B, N, 3), centres new_xyz (B, npoint, 3) -> indices (B, npoint, nsample)."""
    N = xyz.shape[1]
    inside = torch.cdist(new_xyz, xyz) ** 2 < radius * radius
    order = torch.where(inside, torch.arange(N, device=xyz.device), N).sort(dim=2).values[:, :, :nsample]
    first = order[:, :, :1]
    idx = torch.where(order < N, order, first)       # short ball: repeat its first point
    return torch.where(first < N, idx, 0)            # empty ball: zeros


def three_nn(unknown, known):
    """unknown (B, n, 3), known (B, m, 3) -> distances (B, n, 3), indices (B, n, 3)."""
    d2, idx = (torch.cdist(unknown, known) ** 2).topk(3, dim=2, largest=False)
    return torch.sqrt(d2), idx


def three_interpolate(features, idx, weight):
    """features (B, C, m), idx and weight (B, n, 3) -> (B, C, n)."""
    return (grouping_operation(features, idx) * weight[:, None]).sum(-1)


class QueryAndGroup(nn.Module):
    def __init__(self, radius, nsample, use_xyz=True):
        super().__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz, new_xyz, features=None):
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        grouped = grouping_operation(xyz.transpose(1, 2).contiguous(), idx) - new_xyz.transpose(1, 2).unsqueeze(-1)
        if features is None:
            return grouped
        grouped_features = grouping_operation(features, idx)
        return torch.cat([grouped, grouped_features], dim=1) if self.use_xyz else grouped_features


class GroupAll(nn.Module):
    def __init__(self, use_xyz=True):
        super().__init__()
        self.use_xyz = use_xyz

    def forward(self, xyz, new_xyz, features=None):
        grouped = xyz.transpose(1, 2).unsqueeze(2)
        if features is None:
            return grouped
        return torch.cat([grouped, features.unsqueeze(2)], dim=1) if self.use_xyz else features.unsqueeze(2)
