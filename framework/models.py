"""Torch edge-list implementation of STAGATE and controlled multi-relation extensions.

The reference weight-copy semantics (independent decoder Parameters sharing transposed
storage) are intentionally preserved. They are NOT replaced by autograd weight tying.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


def edge_softmax(scores: torch.Tensor, destination: torch.Tensor, n: int) -> torch.Tensor:
    maxima = scores.new_full((n,), -torch.inf)
    maxima.scatter_reduce_(0, destination, scores.detach(), reduce="amax", include_self=True)
    exponents = (scores - maxima[destination]).exp()
    denominator = scores.new_zeros(n).index_add_(0, destination, exponents)
    return exponents / denominator[destination].clamp_min(1e-12)


def aggregate(projected: torch.Tensor, edges: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(projected).index_add_(0, edges[1], projected[edges[0]] * attention[:, None])


class NativeGAT(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin_src = nn.Parameter(torch.zeros(in_dim, out_dim))
        nn.init.xavier_normal_(self.lin_src.data, gain=1.414)
        self.lin_dst = self.lin_src
        self.att_src = nn.Parameter(torch.empty(1, 1, out_dim))
        self.att_dst = nn.Parameter(torch.empty(1, 1, out_dim))
        nn.init.xavier_normal_(self.att_src.data, gain=1.414)
        nn.init.xavier_normal_(self.att_dst.data, gain=1.414)

    def attention(self, projected: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        source = (projected * self.att_src[0]).sum(-1)
        target = (projected * self.att_dst[0]).sum(-1)
        scores = torch.sigmoid(source[edges[0]] + target[edges[1]])
        return edge_softmax(scores, edges[1], len(projected))


class RelationHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.att_src = nn.Parameter(torch.empty(1, 1, dim))
        self.att_dst = nn.Parameter(torch.empty(1, 1, dim))
        nn.init.xavier_normal_(self.att_src.data, gain=1.414)
        nn.init.xavier_normal_(self.att_dst.data, gain=1.414)

    attention = NativeGAT.attention


@dataclass
class ModelOutput:
    embedding: torch.Tensor
    reconstruction: torch.Tensor
    relation_weights: torch.Tensor
    attention: dict[str, torch.Tensor]


class SpatialModel(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 512, latent_dim: int = 30, fusion: str = "single",
                 expression_weight: float = .5, expression_cap: float = .2):
        super().__init__()
        if fusion not in ("single", "fixed", "gate", "bounded"):
            raise ValueError(f"Unknown fusion: {fusion}")
        self.fusion = fusion
        self.expression_weight = expression_weight
        self.expression_cap = expression_cap
        # Initialization order and unused Parameters match the official four GAT layers.
        self.conv1 = NativeGAT(in_dim, hidden_dim)
        self.conv2 = NativeGAT(hidden_dim, latent_dim)
        self.conv3 = NativeGAT(latent_dim, hidden_dim)
        self.conv4 = NativeGAT(hidden_dim, in_dim)
        if fusion != "single":
            self.expression_head = RelationHead(hidden_dim)
        if fusion in ("gate", "bounded"):
            self.gates = nn.ModuleList([nn.Linear(2 * hidden_dim, 1) for _ in range(2)])
            for gate in self.gates:
                nn.init.zeros_(gate.weight)
                nn.init.zeros_(gate.bias)
            if fusion == "bounded":
                if not 0 < expression_weight < expression_cap:
                    raise ValueError("Bounded gate requires 0 < expression_weight < expression_cap")
                nn.init.constant_(self.gates[1].bias, math.log(expression_weight / (expression_cap - expression_weight)))

    def forward(self, features: torch.Tensor, graphs: dict[str, torch.Tensor]) -> ModelOutput:
        projected = features @ self.conv1.lin_src
        alpha = {"spatial": self.conv1.attention(projected, graphs["spatial"])}
        messages = [aggregate(projected, graphs["spatial"], alpha["spatial"])]
        relations = ["spatial"]
        if self.fusion != "single":
            relations.append("expression")
            alpha["expression"] = self.expression_head.attention(projected, graphs["expression"])
            messages.append(aggregate(projected, graphs["expression"], alpha["expression"]))
        if self.fusion in ("gate", "bounded"):
            scores = torch.cat([gate(torch.cat((projected, message), dim=1))
                                for gate, message in zip(self.gates, messages)], dim=1)
            if self.fusion == "bounded":
                rho = self.expression_cap * torch.sigmoid(scores[:, 1:2] - scores[:, 0:1])
                weights = torch.cat((1 - rho, rho), dim=1)
            else:
                weights = scores.softmax(dim=1)
        else:
            weights = features.new_full((len(features), len(relations)), 1.0 / len(relations))
            if len(relations) == 2:
                weights[:, 0] = 1 - self.expression_weight
                weights[:, 1] = self.expression_weight
        hidden = F.elu(sum(weights[:, i:i + 1] * message for i, message in enumerate(messages)))
        z = hidden @ self.conv2.lin_src
        # Same semantics as official STAGATE.forward, including decoder optimizer state.
        self.conv3.lin_src.data = self.conv2.lin_src.T
        self.conv3.lin_dst.data = self.conv2.lin_dst.T
        self.conv4.lin_src.data = self.conv1.lin_src.T
        self.conv4.lin_dst.data = self.conv1.lin_dst.T
        decoded = z @ self.conv3.lin_src
        reconstructed_hidden = F.elu(sum(weights[:, i:i + 1] * aggregate(decoded, graphs[name], alpha[name])
                                         for i, name in enumerate(relations)))
        reconstructed = reconstructed_hidden @ self.conv4.lin_src
        return ModelOutput(z, reconstructed, weights, alpha)


class ReferenceAdapter(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        try:
            from ._reference.STAGATE import STAGATE
        except ImportError as exc:
            raise RuntimeError("Official reference requires torch-geometric and matching torch-sparse wheels; see README.md") from exc
        self.network = STAGATE([in_dim, hidden_dim, latent_dim])

    def forward(self, features: torch.Tensor, graphs: dict[str, torch.Tensor]) -> ModelOutput:
        z, reconstruction = self.network(features, graphs["spatial"])
        return ModelOutput(z, reconstruction, features.new_ones((len(features), 1)), {})


def build_model(variant: str, in_dim: int, hidden_dim: int, latent_dim: int,
                expression_weight: float = .5, expression_cap: float = .2) -> nn.Module:
    if variant == "reference":
        return ReferenceAdapter(in_dim, hidden_dim, latent_dim)
    fusion = ("bounded" if variant == "dual_bounded_gate" else "gate" if variant.startswith("dual_gate")
              else "fixed" if variant in ("dual_fixed", "dual_weak") else "single")
    weight = .5 if variant == "dual_fixed" else expression_weight
    return SpatialModel(in_dim, hidden_dim, latent_dim, fusion, weight, expression_cap)
