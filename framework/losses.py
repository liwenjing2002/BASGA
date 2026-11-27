import torch
from torch.nn import functional as F


def structure_losses(z: torch.Tensor, edges: torch.Tensor, boundary: torch.Tensor, margin: float = 1.0):
    if torch.any(edges[0] == edges[1]):
        raise ValueError("Structural losses must not contain self loops")
    normalized = F.normalize(z, p=2, dim=1, eps=1e-8)
    distance = torch.linalg.vector_norm(normalized[edges[0]] - normalized[edges[1]], dim=1)
    boundary = boundary.detach()
    intra = ((1 - boundary) * distance.square()).mean()
    inter = (boundary * (margin - distance).clamp_min(0).square()).mean()
    return intra, inter


def loss_weights(variant: str, boundary_cfg: dict) -> tuple[float, float]:
    if variant in ("reference", "baseline", "dual_fixed", "dual_gate", "dual_weak", "dual_bounded_gate"):
        return 0.0, 0.0
    intra, inter = boundary_cfg["intra_weight"], boundary_cfg["inter_weight"]
    if variant == "dual_gate_intra":
        inter = 0.0
    if variant == "dual_gate_inter":
        intra = 0.0
    return intra, inter


def ramp_factor(epoch: int, warmup: int, ramp: int) -> float:
    if epoch <= warmup:
        return 0.0
    return min(1.0, (epoch - warmup) / max(1, ramp))
