"""Four coordinate-only set encoders; internal pooling queries are not ball queries."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class PoolAttention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        self.null = nn.Parameter(torch.zeros(1, 1, dim))
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, x: Tensor, valid: Tensor) -> Tensor:
        n = x.shape[0]
        keys = torch.cat((self.null.expand(n, -1, -1), x), 1)
        mask = torch.cat((valid.new_ones(n, 1), valid), 1)
        result, _ = self.attention(self.query.expand(n, -1, -1), keys, keys,
                                   key_padding_mask=~mask, need_weights=False)
        return result[:, 0]


def masked_mean(x: Tensor, valid: Tensor) -> Tensor:
    return (x * valid[..., None]).sum(-2) / valid.sum(-1, keepdim=True).clamp_min(1)


class PoseTokenizer(nn.Module):
    """B,T,N,17,2 coordinates and validity -> one pose token per frame."""

    def __init__(self, method: str, dim: int, heads: int) -> None:
        super().__init__()
        self.method = method
        self.null = nn.Parameter(torch.zeros(dim))
        if method in {"deepsets", "attention"}:
            self.player = nn.Sequential(nn.Linear(34, dim), nn.GELU(), nn.Linear(dim, dim))
        else:
            self.joint = nn.Linear(2, dim)
            self.joint_identity = nn.Parameter(torch.randn(17, dim) * .02)
        if method == "deepsets":
            self.output = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        elif method == "attention":
            self.pool = PoolAttention(dim, heads)
        elif method == "hierarchical":
            self.joint_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
            self.joint_pool = PoolAttention(dim, heads)
            self.player_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
            self.pool = PoolAttention(dim, heads)
        elif method == "gnn":
            edges = ((0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 6),
                     (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
                     (11, 12), (11, 13), (13, 15), (12, 14), (14, 16))
            graph = torch.eye(17)
            for a, b in edges:
                graph[a, b] = graph[b, a] = 1
            self.register_buffer("adjacency", graph)
            self.messages = nn.ModuleList((nn.Linear(dim, dim), nn.Linear(dim, dim)))
            self.pool = PoolAttention(dim, heads)
        else:
            raise ValueError("Unsupported pose pooling")

    def forward(self, coordinates: Tensor, valid: Tensor) -> Tensor:
        b, t, people, joints, _ = coordinates.shape
        if people == 0:
            return self.null.expand(b, t, -1)
        v = valid.reshape(b * t, people, joints)
        xy = coordinates.masked_fill(~valid[..., None], 0).reshape(b * t, people, joints, 2)
        person_valid = v.any(-1)
        if self.method in {"deepsets", "attention"}:
            h = self.player(xy.flatten(-2))
        else:
            h = self.joint(xy) + self.joint_identity
            if self.method == "gnn":
                edges = self.adjacency[None, None] * v[..., None, :]
                edges = edges / edges.sum(-1, keepdim=True).clamp_min(1)
                for layer in self.messages:
                    h = torch.nn.functional.gelu(layer(edges @ h)) * v[..., None]
                h = masked_mean(h, v)
            else:
                flat = h.reshape(b * t * people, joints, -1)
                mask = v.reshape(b * t * people, joints)
                # A learned null key prevents all-masked attention for absent people.
                keys = torch.cat((self.null.expand(len(flat), 1, -1), flat), 1)
                key_valid = torch.cat((mask.new_ones(len(flat), 1), mask), 1)
                h, _ = self.joint_attention(flat, keys, keys, key_padding_mask=~key_valid, need_weights=False)
                h = self.joint_pool(h, mask).reshape(b * t, people, -1)
                keys = torch.cat((self.null.expand(b * t, 1, -1), h), 1)
                pv = torch.cat((person_valid.new_ones(b * t, 1), person_valid), 1)
                delta, _ = self.player_attention(h, keys, keys, key_padding_mask=~pv, need_weights=False)
                h = h + delta
        if self.method == "deepsets":
            result = self.output(masked_mean(h, person_valid))
        else:
            result = self.pool(h, person_valid)
        result = torch.where(person_valid.any(-1, keepdim=True), result, self.null)
        return result.reshape(b, t, -1)
