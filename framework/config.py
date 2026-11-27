from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import tomllib

VARIANTS = (
    "reference", "baseline", "dual_fixed", "dual_gate", "baseline_boundary",
    "dual_gate_boundary", "dual_gate_intra", "dual_gate_inter",
    "dual_weak", "dual_bounded_gate",
)
DEFAULT_VARIANTS = VARIANTS[:8]
FORMAL_SEEDS = (0, 1, 2, 3, 4)
DEV_SEEDS = (10, 11)


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()[:12]


def load_config(path: str | Path, data_dir: str | None = None, output_dir: str | None = None) -> dict:
    path = Path(path).expanduser().resolve()
    with path.open("rb") as stream:
        cfg = tomllib.load(stream)
    cfg = complete_config(cfg)
    for section, key in (("data", "directory"), ("data", "cache_dir"), ("output", "directory")):
        raw = Path(os.path.expandvars(cfg[section][key])).expanduser()
        cfg[section][key] = str((path.parent / raw).resolve() if not raw.is_absolute() else raw.resolve())
    if data_dir or os.environ.get("SPATIAL_DATA_DIR"):
        cfg["data"]["directory"] = str(Path(data_dir or os.environ["SPATIAL_DATA_DIR"]).expanduser().resolve())
    if output_dir:
        cfg["output"]["directory"] = str(Path(output_dir).expanduser().resolve())
    validate_config(cfg)
    return cfg


def complete_config(cfg: dict) -> dict:
    """Add backward-compatible defaults when reading v0.1 configurations/checkpoints."""
    cfg = copy.deepcopy(cfg)
    defaults = {
        "training": {"adam_foreach": "auto", "adam_fused": "auto", "strict_determinism": False,
                     "gate_lr_factor": 1.0, "gate_warmup": 0, "snapshot_epochs": []},
        "graph": {"expression_scope": "global"},
        "model": {"expression_weight": 0.5, "expression_cap": 0.2},
        "boundary": {"mode": "expression", "shuffle_seed": 17},
    }
    for section, values in defaults.items():
        for key, value in values.items():
            cfg[section].setdefault(key, value)
    return cfg


def validate_config(cfg: dict) -> None:
    for section, keys in {"data": ("n_hvg", "pca_dim", "target_sum"), "graph": ("radius", "expression_k"),
                          "model": ("hidden_dim", "latent_dim"), "training": ("epochs", "lr", "checkpoint_every", "threads"),
                          "evaluation": ("neighbors",)}.items():
        for key in keys:
            if cfg[section][key] <= 0:
                raise ValueError(f"{section}.{key} must be positive")
    if cfg['evaluation'].get('method','mclust')=='mclust' and cfg['evaluation']['n_clusters']<=0:
        raise ValueError('evaluation.n_clusters must be positive for mclust')
    if cfg["data"]["pca_dim"] >= cfg["data"]["n_hvg"]:
        raise ValueError("pca_dim must be smaller than n_hvg")
    for key in ("intra_weight", "inter_weight", "margin", "slope"):
        if cfg["boundary"][key] < 0:
            raise ValueError(f"boundary.{key} cannot be negative")
    if cfg["training"]["warmup"] < 0 or cfg["training"]["ramp"] < 0:
        raise ValueError("warmup and ramp cannot be negative")
    for key in ("adam_foreach", "adam_fused"):
        value = cfg["training"].get(key, "auto")
        if not (type(value) is bool or value == "auto"):
            raise ValueError(f"training.{key} must be a boolean or 'auto'")
    if cfg["training"].get("adam_foreach") is True and cfg["training"].get("adam_fused") is True:
        raise ValueError("foreach and fused cannot both be true")
    if cfg["training"].get("gate_lr_factor", 1) <= 0 or cfg["training"].get("gate_warmup", 0) < 0:
        raise ValueError("gate_lr_factor must be positive; gate_warmup cannot be negative")
    if cfg["graph"].get("expression_scope", "global") not in ("global", "spatial_2hop"):
        raise ValueError("Unknown expression graph scope")
    if cfg["boundary"].get("mode", "expression") not in ("expression", "constant", "shuffled"):
        raise ValueError("Unknown boundary prior mode")
    weight = cfg["model"].get("expression_weight", .5)
    cap = cfg["model"].get("expression_cap", .2)
    if not 0 <= weight <= 1 or not 0 < cap < 1:
        raise ValueError("Invalid expression_weight or expression_cap")
    if cfg.get("variant") == "dual_bounded_gate" and not 0 < weight < cap:
        raise ValueError("Bounded gate initialization needs 0 < expression_weight < expression_cap")


def effective_config(cfg: dict, variant: str, smoke: bool = False, **overrides) -> dict:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    out = complete_config(cfg)
    for key, section in (("epochs", "training"), ("expression_k", "graph"),
                         ("intra_weight", "boundary"), ("inter_weight", "boundary"), ("rscript", "evaluation"),
                         ("lr", "training"), ("adam_foreach", "training"), ("adam_fused", "training"),
                         ("expression_weight", "model"), ("expression_cap", "model"), ("expression_scope", "graph"),
                         ("boundary_mode", "boundary")):
        if overrides.get(key) is not None:
            out[section]["mode" if key == "boundary_mode" else key] = overrides[key]
    if smoke:
        out["training"].update(epochs=3, warmup=0, ramp=1, checkpoint_every=1)
        out["model"].update(hidden_dim=64, latent_dim=8)
        out["training"]["snapshot_epochs"] = [1, 3]
    out["variant"] = variant
    out["smoke"] = smoke
    validate_config(out)
    return out


def scientific_config(cfg: dict) -> dict:
    """Exclude machine-specific paths; retain every scientific parameter."""
    out = copy.deepcopy(cfg)
    for key in ("directory", "cache_dir"):
        out["data"].pop(key, None)
    out.pop("output", None)
    out["evaluation"].pop("rscript", None)
    return out
