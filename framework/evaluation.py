from __future__ import annotations

import copy
import os
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score

from .io import read_json


def r_process_env() -> dict:
    env = os.environ.copy()
    if os.name == "nt" and env.get("LC_ALL", "C.UTF-8") == "C.UTF-8":
        # C.UTF-8 is a Linux locale name; R for Windows needs its supported UTF-8 locale.
        env["LC_ALL"] = "en_US.UTF-8"
        env["LANG"] = "en_US.UTF-8"
    return env


def r_environment(rscript: str) -> dict:
    expression = ('if (!requireNamespace("mclust", quietly=TRUE)) stop("mclust missing"); '
                  'cat(as.character(getRversion()), "\\n", as.character(packageVersion("mclust")), sep="")')
    result = subprocess.run([rscript, "--vanilla", "-e", expression], capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=60, env=r_process_env())
    if result.returncode:
        raise RuntimeError("R/mclust preflight failed: " + result.stderr[-1000:])
    lines = result.stdout.strip().splitlines()
    if len(lines) != 2:
        raise RuntimeError("Unexpected R/mclust version output: " + result.stdout)
    return {"R_version": lines[0], "mclust_version": lines[1]}


def cluster_mclust(z: np.ndarray, barcodes, cfg: dict, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    input_path, output_path = directory / "embedding_for_mclust.csv", directory / "mclust_predictions.csv"
    metadata_path = directory / "mclust_metadata.json"
    pd.DataFrame(z, index=barcodes).to_csv(input_path, index_label="barcode", float_format="%.17g")
    command = [cfg["rscript"], "--vanilla", str(Path(__file__).with_name("mclust.R")),
               str(input_path.resolve()), str(output_path.resolve()), str(metadata_path.resolve()),
               str(cfg["n_clusters"]), cfg["mclust_model"], str(cfg["mclust_seed"])]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600, env=r_process_env())
    except FileNotFoundError as exc:
        raise RuntimeError("Rscript was not found; pass --rscript /path/to/Rscript. See README.md") from exc
    (directory / "mclust.log").write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"mclust failed. See {directory / 'mclust.log'}: {completed.stderr[-1200:]}")
    prediction = pd.read_csv(output_path, dtype={"barcode": str, "prediction": str}).set_index("barcode")
    if not prediction.index.is_unique or set(prediction.index) != set(barcodes):
        raise ValueError("mclust barcode alignment failed")
    return prediction.loc[barcodes, "prediction"].to_numpy(), read_json(metadata_path)


def clustering_metrics(truth, prediction, mask) -> dict:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return {"ARI": None, "NMI": None, "HS": None, "n_evaluated": 0,
                "annotation_status": "not_provided", "n_clusters": len(set(prediction))}
    truth = np.asarray(truth, dtype=object)[mask].astype(str)
    prediction = np.asarray(prediction, dtype=str)[mask]
    return {"ARI": float(adjusted_rand_score(truth, prediction)),
            "NMI": float(normalized_mutual_info_score(truth, prediction)),
            "HS": float(homogeneity_score(truth, prediction)), "n_evaluated": int(mask.sum())}


def clustering_environment():
    import importlib.metadata
    # Fail before training if the exact tutorial clustering backend is unavailable.
    import louvain
    return {name: importlib.metadata.version(name) for name in ("scanpy", "igraph", "louvain")}


def cluster_louvain(adata, cfg, directory):
    import scanpy as sc
    sc.pp.neighbors(adata, n_neighbors=cfg["neighbors"], use_rep=cfg["use_rep"],
                    metric="euclidean", random_state=cfg["plot_seed"])
    sc.tl.louvain(adata, resolution=cfg["resolution"], random_state=cfg["cluster_seed"],
                  key_added="louvain", flavor="vtraag", directed=True, use_weights=False)
    prediction = adata.obs["louvain"].astype(str).to_numpy()
    metadata = dict(method="louvain", flavor="vtraag", resolution=cfg["resolution"],
                    random_state=cfg["cluster_seed"], directed=True, use_weights=False,
                    use_rep=cfg["use_rep"], n_clusters=len(set(prediction)))
    pd.DataFrame({"barcode": adata.obs_names, "prediction": prediction}).to_csv(
        directory / "louvain_predictions.csv", index=False)
    return prediction, metadata


def _single_visualizations(adata, cfg: dict, directory: Path):
    import scanpy as sc
    sc.tl.umap(adata, min_dist=cfg["umap_min_dist"], random_state=cfg["plot_seed"], init_pos="spectral")
    adata.obs["prediction"] = adata.obs["prediction"].cat.remove_unused_categories()
    categories = adata.obs["prediction"].cat.categories.astype(str).to_list()
    if len(categories) == 1:
        connectivity = sparse.csr_matrix((1, 1))
    else:
        sc.tl.paga(adata, groups="prediction")
        connectivity = adata.uns["paga"]["connectivities"].copy()
    pd.DataFrame(connectivity.toarray(), index=categories, columns=categories).to_csv(
        directory / "paga_pred_connectivities.csv")
    sparse.save_npz(directory / "paga_neighbors.npz", adata.obsp["connectivities"])
    pd.Series(adata.obs_names, name="barcode").to_csv(directory / "paga_barcodes.csv", index=False)
    adata.uns["paga_pred"] = dict(connectivities=connectivity, groups="prediction",
                                 group_order=np.asarray(categories, dtype=str), threshold=cfg["paga_threshold"])
    return {"pred": dict(connectivities=connectivity.toarray(), categories=categories,
                         labels=adata.obs["prediction"].astype(str).to_numpy(), umap=adata.obsm["X_umap"].copy())}


def _labeled_visualizations(adata, cfg: dict, directory: Path):
    import scanpy as sc
    n_neighbors = min(cfg["neighbors"], adata.n_obs - 1)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X_model", metric="euclidean", random_state=cfg["plot_seed"])
    sc.tl.umap(adata, min_dist=cfg["umap_min_dist"], random_state=cfg["plot_seed"], init_pos="spectral")
    # Both PAGA groupings use exactly the same labeled spots and the same latent graph.
    used = adata[np.asarray(adata.obs["evaluation_mask"], dtype=bool)].copy()
    sc.pp.neighbors(used, n_neighbors=min(cfg["neighbors"], used.n_obs - 1), use_rep="X_model",
                    metric="euclidean", random_state=cfg["plot_seed"])
    sparse.save_npz(directory / "paga_neighbors.npz", used.obsp["connectivities"])
    pd.Series(used.obs_names, name="barcode").to_csv(directory / "paga_barcodes.csv", index=False)
    paga = {}
    for group, key in (("prediction", "pred"), ("ground_truth", "truth")):
        used.obs[group] = used.obs[group].cat.remove_unused_categories()
        sc.tl.paga(used, groups=group)
        connectivity = used.uns["paga"]["connectivities"].copy()
        categories = used.obs[group].cat.categories.astype(str).to_list()
        pd.DataFrame(connectivity.toarray(), index=categories, columns=categories).to_csv(directory / f"paga_{key}_connectivities.csv")
        adata.uns[f"paga_{key}"] = {"connectivities": connectivity, "groups": group,
                                    "group_order": np.asarray(categories, dtype=str), "n_spots": used.n_obs,
                                    "threshold": cfg["paga_threshold"]}
        paga[key] = {"connectivities": connectivity.toarray(), "categories": categories,
                     "labels": used.obs[group].astype(str).to_numpy(), "umap": used.obsm["X_umap"].copy()}
    adata.uns["paga_evaluation_barcodes"] = np.asarray(used.obs_names, dtype=str)
    return paga

def backend_environment(cfg):
    method=cfg['evaluation']['method']
    if method=='mclust': return r_environment(cfg['evaluation']['rscript'])
    if method=='louvain': return clustering_environment()
    return {'task':'denoising','clustering':'not_used'}

def cluster(adata,cfg,directory):
    if cfg['method']=='louvain': return cluster_louvain(adata,cfg,directory)
    return cluster_mclust(adata.obsm['X_model'],adata.obs_names,cfg,directory)

def compute_visualizations(adata,cfg,directory):
    if cfg.get('paga_policy')=='labeled_recomputed':
        return _labeled_visualizations(adata,cfg,directory)
    if cfg['method']=='mclust':
        import scanpy as sc
        sc.pp.neighbors(adata,n_neighbors=cfg['neighbors'],use_rep='X_model',metric='euclidean',random_state=cfg['plot_seed'])
    return _single_visualizations(adata,cfg,directory)
