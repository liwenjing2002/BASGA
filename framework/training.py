from __future__ import annotations

import os
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .config import digest, scientific_config, complete_config
from .evaluation import cluster, clustering_metrics, compute_visualizations, backend_environment
from .graphs import spatial_edges, expression_edges, local_expression_edges, structural_edges, boundary_prior, apply_boundary_control, graph_summary
from .io import write_json, read_json, environment_info, capture_freeze, source_hashes
from .losses import structure_losses, loss_weights, ramp_factor
from .models import build_model


def set_seed(seed: int, threads: int = 4, deterministic: bool = True, strict: bool = False):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    # Strict round-2 audits stop on unsupported nondeterministic CUDA operations.
    # Legacy configs retain warn-only behavior for compatibility.
    torch.use_deterministic_algorithms(deterministic, warn_only=not strict)


def resolve_device(request: str) -> torch.device:
    if request == "auto":
        request = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(request)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Install CUDA-enabled PyTorch and check torch.cuda.is_available()")
    return device


def make_graphs(adata,cfg):
    if cfg['experiment']['kind']=='hippo':
        from .multisection_graph import graph_for_cfg
        return {'spatial':graph_for_cfg(adata,cfg)}
    return {'spatial':spatial_edges(adata.obsm['spatial'],cfg['graph']['radius'])}


def build_optimizer(model, training: dict):
    def option(name):
        value = training.get(name, "auto")
        return None if value == "auto" else value
    factor = training.get("gate_lr_factor", 1.0)
    parameters = list(model.parameters())
    if hasattr(model, "gates") and factor != 1.0:
        gate_params = list(model.gates.parameters())
        gate_ids = {id(p) for p in gate_params}
        parameters = [{"params": [p for p in parameters if id(p) not in gate_ids], "lr": training["lr"]},
                      {"params": gate_params, "lr": training["lr"] * factor}]
    return torch.optim.Adam(parameters, lr=training["lr"], weight_decay=training["weight_decay"],
                            foreach=option("adam_foreach"), fused=option("adam_fused"))


def run_experiment(cfg: dict, seed: int, *, device_request="auto", tag="default", split="formal",
                   skip_completed=False, resume=False, prepare_fn=None, data_code_signature=None) -> Path:
    cfg = complete_config(cfg)
    if not tag or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in tag):
        raise ValueError("tag must contain only letters, digits, underscores or hyphens")
    if cfg["smoke"]:
        split = "smoke"
    elif split == "formal" and seed not in range(5):
        raise ValueError("Formal seeds are fixed to 0..4; use --split development for screening")
    elif split == "development" and seed not in (10, 11):
        raise ValueError("Development screening seeds are fixed to 10 and 11")
    if split not in ("smoke", "formal", "development", "audit") or seed < 0:
        raise ValueError("Invalid experiment split or seed")
    device = resolve_device(device_request)
    set_seed(seed, cfg["training"]["threads"], cfg["training"]["deterministic"], cfg["training"]["strict_determinism"])
    adata, integrity = prepare_fn(cfg)
    scientific = scientific_config(cfg)
    code = dict(source_hashes(), experiment_data_ast=data_code_signature)
    environment = environment_info()
    environment["clustering_backend"] = backend_environment(cfg)
    environment_key = {key: environment[key] for key in ("packages", "cuda_runtime", "gpu", "clustering_backend")}
    environment_key["execution_device"] = str(device)
    environment["execution_device"] = str(device)
    config_id = digest({"config": scientific, "data": integrity["cache_id"], "code": code, "environment": environment_key})
    # A comparison protocol allows the intended graph/structural ablations, but never
    # mixes different data, epoch counts, model dimensions or evaluation settings.
    protocol = {key: scientific[key] for key in ("data", "training", "evaluation", "smoke")}
    protocol["model"] = {key: scientific["model"][key] for key in ("hidden_dim", "latent_dim")}
    protocol["radius"] = cfg["graph"]["radius"]
    protocol_id = digest({"protocol": protocol, "data": integrity["cache_id"], "code": code, "environment": environment_key})
    directory = Path(cfg["output"]["directory"]) / split / tag / cfg["variant"] / f"{config_id}_seed{seed}"
    metrics_path = directory / "metrics.json"
    if metrics_path.is_file():
        previous = read_json(metrics_path)
        if previous.get("status") == "complete":
            required = ["result.h5ad", "checkpoint.pt", "history.csv", "embedding.npy"]
            if not all((directory / name).is_file() for name in required):
                raise RuntimeError(f"Completed run is missing artifacts: {directory}; use plot to regenerate figures")
            if skip_completed:
                print(f"SKIP completed {cfg['variant']} seed {seed}: {directory}", flush=True)
                return directory
            raise FileExistsError(f"Run exists: {directory}; use --skip-completed or choose a new --tag")
    if directory.exists() and any(directory.iterdir()) and not resume:
        raise FileExistsError(f"Incomplete run exists: {directory}; use --resume or choose a new --tag")
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "config.json", cfg)
    write_json(directory / "integrity.json", integrity)
    write_json(directory / "code_hashes.json", code)
    write_json(directory / "environment.json", environment)
    capture_freeze(directory)
    write_json(directory / "status.json", {"status": "running", "seed": seed, "variant": cfg["variant"], "split": split})
    graphs_numpy = make_graphs(adata, cfg)
    write_json(directory / "graphs.json", {name: graph_summary(edges, adata.n_obs) for name, edges in graphs_numpy.items()})
    np.savez_compressed(directory / "graphs.npz", **graphs_numpy)
    graphs = {name: torch.as_tensor(edges, dtype=torch.long, device=device) for name, edges in graphs_numpy.items()}
    x = torch.as_tensor(adata.X.toarray(), dtype=torch.float32, device=device)
    # Preprocessing is seeded independently; reset before model init for exact reference parity.
    set_seed(seed, cfg["training"]["threads"], cfg["training"]["deterministic"], cfg["training"]["strict_determinism"])
    model = build_model(cfg["variant"], adata.n_vars, **cfg["model"]).to(device)
    optimizer = build_optimizer(model, cfg["training"])
    struct_numpy = structural_edges(graphs_numpy["spatial"])
    prior_numpy = boundary_prior(adata.obsm["X_pca"], struct_numpy, cfg["boundary"]["slope"], cfg["boundary"]["offset"])
    prior_numpy = apply_boundary_control(prior_numpy, cfg["boundary"]["mode"], cfg["boundary"]["shuffle_seed"])
    struct = torch.as_tensor(struct_numpy, dtype=torch.long, device=device)
    prior = torch.as_tensor(prior_numpy, device=device)
    np.savez_compressed(directory / "boundary_prior.npz", edge_index=struct_numpy, probability=prior_numpy)
    weights = loss_weights(cfg["variant"], cfg["boundary"])
    history = []
    checkpoint_path = directory / "checkpoint.pt"
    first_epoch = 1
    elapsed_before = 0.0
    if resume and checkpoint_path.is_file():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if state["config_id"] != config_id or state["seed"] != seed:
            raise ValueError("Checkpoint scientific configuration does not match")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        first_epoch = state["epoch"] + 1
        history = state["history"]
        elapsed_before = state.get("training_seconds", 0.0)
        random.setstate(state["random_state"])
        np.random.set_state(state["numpy_state"])
        torch.set_rng_state(state["torch_state"].cpu())
        if device.type == "cuda" and state.get("cuda_state") is not None:
            torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda_state"]])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    try:
        for epoch in range(first_epoch, cfg["training"]["epochs"] + 1):
            model.train()
            if hasattr(model, "gates"):
                for p in model.gates.parameters():
                    p.requires_grad_(epoch > cfg["training"]["gate_warmup"])
            optimizer.zero_grad(set_to_none=True)
            output = model(x, graphs)
            reconstruction = F.mse_loss(output.reconstruction, x)
            intra = reconstruction.new_zeros(())
            inter = reconstruction.new_zeros(())
            scale = ramp_factor(epoch, cfg["training"]["warmup"], cfg["training"]["ramp"])
            if any(weights):
                intra, inter = structure_losses(output.embedding, struct, prior, cfg["boundary"]["margin"])
            loss = reconstruction + scale * (weights[0] * intra + weights[1] * inter)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip"])
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"Non-finite gradient at epoch {epoch}")
            optimizer.step()
            row = {"epoch": epoch, "total": float(loss.detach()), "reconstruction": float(reconstruction.detach()),
                   "intra": float(intra.detach()), "inter": float(inter.detach()), "structure_scale": scale,
                   "gradient_norm": float(gradient_norm), "lr": optimizer.param_groups[0]["lr"],
                   "gate_lr": optimizer.param_groups[-1]["lr"], "seconds": elapsed_before + time.perf_counter() - start}
            history.append(row)
            if epoch == 1 or epoch % cfg["training"]["checkpoint_every"] == 0 or epoch == cfg["training"]["epochs"]:
                print(f"{cfg['variant']} seed={seed} epoch={epoch}/{cfg['training']['epochs']} loss={row['total']:.6f} lr={row['lr']:.6g}", flush=True)
                pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
                checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                              "history": history, "config_id": config_id, "seed": seed,
                              "training_seconds": row["seconds"], "random_state": random.getstate(),
                              "numpy_state": np.random.get_state(), "torch_state": torch.get_rng_state(),
                              "cuda_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}
                temporary = directory / "checkpoint.partial.pt"
                torch.save(checkpoint, temporary)
                temporary.replace(checkpoint_path)
            if epoch in cfg["training"]["snapshot_epochs"]:
                model.eval()
                with torch.no_grad():
                    snapshot = model(x, graphs)
                    snapshot_dir = directory / "snapshots"
                    snapshot_dir.mkdir(exist_ok=True)
                    np.savez_compressed(snapshot_dir / f"epoch_{epoch}.npz", embedding=snapshot.embedding.cpu().numpy(),
                                        relation_weights=snapshot.relation_weights.cpu().numpy())
                    del snapshot
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        training_seconds = elapsed_before + time.perf_counter() - start
        model.eval()
        with torch.no_grad():
            output = model(x, graphs)
            z = output.embedding.cpu().numpy()
            reconstructed = output.reconstruction.cpu().numpy()
            relation_weights = output.relation_weights.cpu().numpy()
            attention = {key: value.cpu().numpy() for key, value in output.attention.items()}
        if not np.isfinite(z).all() or not np.isfinite(reconstructed).all():
            raise FloatingPointError("Non-finite final model output")
        np.save(directory / "embedding.npy", z)
        np.savez_compressed(directory / "attention.npz", **attention)
        adata.obsm['X_model']=z
        adata.obsm['relation_weights']=relation_weights
        adata.layers['reconstruction']=reconstructed
        common=dict(variant=cfg['variant'],seed=seed,split=split,tag=tag,status='complete',config_id=config_id,
            protocol_id=protocol_id,cache_id=integrity['cache_id'],epochs=cfg['training']['epochs'],
            lr=cfg['training']['lr'],intra_weight=weights[0],inter_weight=weights[1],n_spots=adata.n_obs,n_genes=adata.n_vars,
            training_seconds=training_seconds,device=str(device),
            gpu_peak_memory_mb=float(torch.cuda.max_memory_allocated(device)/1024**2) if device.type=='cuda' else 0.,
            experiment_id=cfg['experiment']['id'],task=cfg['experiment']['task'],graph_mode=cfg['graph'].get('mode','single'))
        if cfg['experiment']['task']=='denoising':
            clipped=np.maximum(reconstructed,0)
            measured=adata.X.toarray()
            adata.layers['R2A_ReX']=clipped
            metrics=dict(common,ARI=None,NMI=None,HS=None,n_evaluated=0,clustering_performed=False,
                reconstruction_MSE_signed=float(np.mean((reconstructed-measured)**2,dtype=np.float64)),
                reconstruction_MSE_clipped=float(np.mean((clipped-measured)**2,dtype=np.float64)),
                negative_reconstruction_fraction=float(np.mean(reconstructed<0)))
        else:
            predictions,cluster_metadata=cluster(adata,cfg['evaluation'],directory)
            metrics=clustering_metrics(adata.obs['ground_truth'],predictions,adata.obs['evaluation_mask'])
            metrics.update(common)
            metrics['n_clusters']=len(set(predictions))
            metrics['clustering']=cluster_metadata
            metrics['annotation_status']='provided' if metrics['n_evaluated'] else 'not_provided'
            if cfg['evaluation']['method']=='mclust': metrics['mclust']=cluster_metadata
            adata.obs['prediction']=pd.Categorical(predictions,categories=sorted(set(predictions),key=int))
        score_sum,degree=np.zeros(adata.n_obs),np.zeros(adata.n_obs)
        for endpoint in struct_numpy:
            np.add.at(score_sum,endpoint,prior_numpy);np.add.at(degree,endpoint,1)
        adata.obs['boundary_score']=score_sum/np.maximum(degree,1)
        adata.uns['experiment_json']=__import__('json').dumps(metrics)
        if cfg['experiment']['task']!='denoising':
            compute_visualizations(adata,cfg['evaluation'],directory)
            adata.obs[['ground_truth','evaluation_mask','prediction','boundary_score']].to_csv(directory/'predictions.csv',index_label='barcode')
        pd.DataFrame(z,index=adata.obs_names).to_csv(directory/'embedding.csv',index_label='barcode')
        adata.write_h5ad(directory/'result.h5ad',compression='gzip')
        metrics["total_seconds_including_analysis"] = elapsed_before + time.perf_counter() - start
        write_json(metrics_path, metrics)
        write_json(directory / "status.json", {"status": "complete", "seed": seed, "variant": cfg["variant"], "split": split})
        print(f"COMPLETE {cfg['experiment']['id']} seed={seed}: {directory}",flush=True)
    except BaseException as exc:
        write_json(directory / "status.json", {"status": "failed", "type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        del model, optimizer, x, graphs
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return directory


def verify_checkpoint(directory):
    import scanpy as sc
    directory=Path(directory)
    cfg=complete_config(read_json(directory/'config.json'))
    a=sc.read_h5ad(directory/'result.h5ad')
    state=torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=False)
    with np.load(directory/'graphs.npz') as stored:
        graphs={k:torch.as_tensor(stored[k].copy(),dtype=torch.long) for k in stored.files}
    model=build_model(cfg['variant'],a.n_vars,**cfg['model'])
    model.load_state_dict(state['model']); model.eval()
    with torch.no_grad(): out=model(torch.as_tensor(a.X.toarray(),dtype=torch.float32),graphs)
    np.testing.assert_allclose(out.embedding.numpy(),a.obsm['X_model'],rtol=2e-4,atol=2e-4)
    np.testing.assert_allclose(out.reconstruction.numpy(),a.layers['reconstruction'],rtol=2e-4,atol=2e-4)
    if 'R2A_ReX' in a.layers: np.testing.assert_array_equal(a.layers['R2A_ReX'],np.maximum(a.layers['reconstruction'],0))
    return {'checkpoint_outputs_match':True,'n_spots':a.n_obs,'n_genes':a.n_vars}
