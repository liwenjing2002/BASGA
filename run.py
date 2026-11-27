"""Run BASGA on DLPFC 151676 with preprocessing, clustering and visualization."""
from __future__ import annotations
from pathlib import Path
import sys
from types import SimpleNamespace
EXPERIMENT_DIR = Path(__file__).resolve().parent
from framework.runtime import Experiment,main as experiment_main
from framework.io import read_json as unified_read_json,write_json as unified_write_json

DEFAULT_CONFIG = {'data': {'directory': 'data',
          'sample': '151676',
          'cache_dir': 'cache',
          'n_hvg': 3000,
          'pca_dim': 30,
          'target_sum': 10000.0,
          'expected_spots': 3460,
          'expected_genes': 33538,
          'expected_labeled': 3431},
 'graph': {'radius': 150.0,
           'expression_k': 10,
           'expected_spatial_edges': 20052,
           'expression_scope': 'global'},
 'model': {'hidden_dim': 512, 'latent_dim': 30, 'expression_weight': 0.1, 'expression_cap': 0.2},
 'training': {'epochs': 1000,
              'lr': 0.001,
              'weight_decay': 0.0001,
              'gradient_clip': 5.0,
              'warmup': 100,
              'ramp': 100,
              'checkpoint_every': 100,
              'threads': 4,
              'deterministic': True,
              'strict_determinism': True,
              'adam_foreach': False,
              'adam_fused': False,
              'gate_lr_factor': 1.0,
              'gate_warmup': 0,
              'snapshot_epochs': [100, 500, 1000]},
 'boundary': {'intra_weight': 0.0,
              'inter_weight': 0.01,
              'margin': 1.0,
              'slope': 2.0,
              'offset': 1.0,
              'mode': 'expression',
              'shuffle_seed': 17},
 'evaluation': {'n_clusters': 7,
                'mclust_model': 'EEE',
                'mclust_seed': 2020,
                'rscript': 'Rscript',
                'neighbors': 15,
                'umap_min_dist': 0.5,
                'plot_seed': 0,
                'paga_threshold': 0.05,
                'method': 'mclust',
                'paga_policy': 'labeled_recomputed'},
 'output': {'directory': 'outputs'}}
VISUAL_DEFAULTS = {'dpi': 600, 'umap_size': 6, 'label_size': 8}



# src\spatial_framework\data.py (inlined; no dependency on the old project)
from pathlib import Path as data_Path
import json as data_json
import numpy as data_np
import pandas as data_pd
from scipy import sparse as data_sparse
from framework.config import digest as data_digest
from framework.graphs import spatial_edges as data_spatial_edges, graph_summary as data_graph_summary
from framework.io import sha256_file as data_sha256_file, write_json as data_write_json
data_PREPROCESS_VERSION = 1
data_LABEL_ORDER = [f'Layer_{i}' for i in range(1, 7)] + ['WM']

def data_input_files(cfg: dict) -> dict[str, data_Path]:
    root = data_Path(cfg['data']['directory'])
    sample = cfg['data']['sample']
    files = {'counts': root / f'{sample}_filtered_feature_bc_matrix.h5', 'positions': root / 'spatial' / 'tissue_positions_list.csv', 'truth': root / f'{sample}_truth.txt'}
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(f'Required data file is missing: {path}')
    return files

def data_load_raw(cfg: dict):
    import scanpy as sc
    files = data_input_files(cfg)
    adata = sc.read_10x_h5(files['counts'])
    adata.var_names_make_unique()
    if not adata.obs_names.is_unique:
        raise ValueError('Expression barcodes must be unique')
    position_columns = ['barcode', 'in_tissue', 'array_row', 'array_col', 'pixel_row', 'pixel_col']
    positions = data_pd.read_csv(files['positions'], header=None, names=position_columns, dtype={'barcode': str})
    if positions['barcode'].duplicated().any():
        raise ValueError('Duplicate spatial barcodes')
    positions = positions.set_index('barcode')
    missing = adata.obs_names.difference(positions.index)
    if len(missing):
        raise ValueError(f'Spatial coordinates are missing for {len(missing)} expression barcodes')
    aligned = positions.loc[adata.obs_names]
    if not (data_pd.to_numeric(aligned['in_tissue']) == 1).all():
        raise ValueError('Filtered expression matrix unexpectedly contains out-of-tissue spots')
    coords = aligned[['pixel_row', 'pixel_col']].apply(data_pd.to_numeric, errors='raise').to_numpy(dtype=data_np.float64)
    if not data_np.isfinite(coords).all():
        raise ValueError('Non-finite spatial coordinates')
    adata.obsm['spatial'] = coords
    truth = data_pd.read_csv(files['truth'], sep='\t', header=None, names=['barcode', 'ground_truth'], keep_default_na=False, dtype=str).set_index('barcode')
    if not truth.index.is_unique:
        raise ValueError('Duplicate truth barcodes')
    if len(adata.obs_names.difference(truth.index)):
        raise ValueError('Truth table must contain one row per expression barcode (empty labels are allowed)')
    labels = truth.reindex(adata.obs_names)['ground_truth'].str.strip()
    labels = labels.mask(labels.str.lower().isin(('', 'nan', 'na', 'none', 'null')))
    unexpected = set(labels.dropna()) - set(data_LABEL_ORDER)
    if unexpected:
        raise ValueError(f'Unexpected layer labels: {sorted(unexpected)}')
    adata.obs['ground_truth'] = data_pd.Categorical(labels, categories=data_LABEL_ORDER, ordered=True)
    adata.obs['evaluation_mask'] = labels.notna().to_numpy()
    if data_sparse.issparse(adata.X):
        values = adata.X.data
    else:
        values = data_np.asarray(adata.X).ravel()
    if not data_np.isfinite(values).all() or (values < 0).any() or (not data_np.allclose(values, data_np.round(values))):
        raise ValueError('The input must contain nonnegative raw integer counts')
    report = {'sample': cfg['data']['sample'], 'n_spots': adata.n_obs, 'n_genes_raw': adata.n_vars, 'n_labeled': int(labels.notna().sum()), 'n_unlabeled': int(labels.isna().sum()), 'labels': {str(k): int(v) for k, v in labels.value_counts().items()}, 'images_required': False}
    for config_key, report_key in (('expected_spots', 'n_spots'), ('expected_genes', 'n_genes_raw'), ('expected_labeled', 'n_labeled')):
        expected = cfg['data'].get(config_key)
        if expected is not None and report[report_key] != expected:
            raise ValueError(f'Data integrity failed: {report_key}={report[report_key]}, expected {expected}')
    return (adata, report)

def data_prepare(cfg: dict):
    import scanpy as sc
    files = data_input_files(cfg)
    identity = {'version': data_PREPROCESS_VERSION, 'sources': {key: data_sha256_file(path) for key, path in files.items()}, 'n_hvg': cfg['data']['n_hvg'], 'pca_dim': cfg['data']['pca_dim'], 'target_sum': cfg['data']['target_sum'], 'sample': cfg['data']['sample']}
    cache_id = data_digest(identity)
    folder = data_Path(cfg['data']['cache_dir']) / f"{cfg['data']['sample']}_{cache_id}"
    cache = folder / 'prepared.h5ad'
    if cache.is_file():
        adata = sc.read_h5ad(cache)
        if adata.uns.get('cache_id') != cache_id:
            raise ValueError('Cached preprocessing identity mismatch')
        report = data_json.loads(adata.uns['integrity_json'])
    else:
        adata, report = data_load_raw(cfg)
        n_hvg = cfg['data']['n_hvg']
        print(f'Preparing {adata.n_obs} spots × {adata.n_vars} genes; selecting {n_hvg} HVGs', flush=True)
        sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=n_hvg)
        chosen = adata.var['highly_variable'].to_numpy()
        raw_selected = adata.X[:, chosen].copy()
        sc.pp.normalize_total(adata, target_sum=cfg['data']['target_sum'])
        sc.pp.log1p(adata)
        adata = adata[:, chosen].copy()
        adata.X = data_sparse.csr_matrix(adata.X, dtype=data_np.float32)
        adata.layers['counts'] = data_sparse.csr_matrix(raw_selected)
        sc.pp.pca(adata, n_comps=cfg['data']['pca_dim'], svd_solver='arpack', random_state=0)
        report['n_hvg'] = adata.n_vars
        adata.uns.update(cache_id=cache_id, input_identity_json=data_json.dumps(identity), integrity_json=data_json.dumps(report))
        folder.mkdir(parents=True, exist_ok=True)
        partial = folder / 'prepared.partial.h5ad'
        adata.write_h5ad(partial, compression='gzip')
        partial.replace(cache)
    graph = data_spatial_edges(adata.obsm['spatial'], cfg['graph']['radius'])
    report = dict(report, spatial_graph=data_graph_summary(graph, adata.n_obs), cache_id=cache_id, cache_file=str(cache))
    expected = cfg['graph'].get('expected_spatial_edges')
    if expected is not None and report['spatial_graph']['directed_nonself_edges'] != expected:
        raise ValueError('Unexpected spatial edge count; check coordinate units and graph radius')
    folder.mkdir(parents=True, exist_ok=True)
    data_write_json(folder / 'integrity.json', report)
    return (adata, report)
data_module = SimpleNamespace(__file__=__file__, LABEL_ORDER=data_LABEL_ORDER, PREPROCESS_VERSION=data_PREPROCESS_VERSION, Path=data_Path, digest=data_digest, graph_summary=data_graph_summary, input_files=data_input_files, json=data_json, load_raw=data_load_raw, np=data_np, pd=data_pd, prepare=data_prepare, sha256_file=data_sha256_file, sparse=data_sparse, spatial_edges=data_spatial_edges, write_json=data_write_json)



# scripts\r2a_protocol.py (inlined; no dependency on the old project)
import sys as protocol_sys
from pathlib import Path as protocol_Path
protocol_ROOT = EXPERIMENT_DIR
from framework.io import read_json as protocol_read_json, sha256_file as protocol_sha256_file


def protocol_verify_config(cfg):
    expected = {'variant': 'baseline_boundary', 'data': {'sample': '151676', 'n_hvg': 3000, 'pca_dim': 30, 'target_sum': 10000.0}, 'graph': {'radius': 150.0}, 'model': {'hidden_dim': 512, 'latent_dim': 30}, 'boundary': {'intra_weight': 0.0, 'inter_weight': 0.01, 'margin': 1.0, 'slope': 2.0, 'offset': 1.0, 'mode': 'expression'}, 'training': {'lr': 0.001, 'weight_decay': 0.0001, 'gradient_clip': 5.0, 'warmup': 100, 'ramp': 100, 'adam_foreach': False, 'adam_fused': False, 'strict_determinism': True, 'deterministic': True, 'threads': 4}, 'evaluation': {'n_clusters': 7, 'mclust_model': 'EEE', 'mclust_seed': 2020, 'neighbors': 15, 'umap_min_dist': 0.5, 'plot_seed': 0, 'paga_threshold': 0.05}}
    for section, values in expected.items():
        if isinstance(values, dict):
            for key, value in values.items():
                if cfg[section].get(key) != value:
                    raise ValueError(f'Expected frozen R2-A {section}.{key}={value!r}')
        elif cfg.get(section) != values:
            raise ValueError(f'Expected {section}={values!r}')
    if cfg['training']['epochs'] != (3 if cfg.get('smoke') else 1000):
        raise ValueError('R2-A uses 1000 epochs; the separately labeled smoke run uses 3.')
protocol_module = SimpleNamespace(__file__=__file__, Path=protocol_Path, ROOT=protocol_ROOT, read_json=protocol_read_json, sha256_file=protocol_sha256_file, sys=protocol_sys, verify_config=protocol_verify_config)



# scripts\paper_style\paper_plots.py (inlined; no dependency on the old project)
from dataclasses import dataclass as style_dataclass, asdict as style_asdict
import hashlib as style_hashlib
import json as style_json
from pathlib import Path as style_Path
import numpy as style_np
import pandas as style_pd
from scipy.optimize import linear_sum_assignment as style_linear_sum_assignment
from scipy.spatial import cKDTree as style_cKDTree
style_LAYERS = [f'Layer_{i}' for i in range(1, 7)] + ['WM']
style_COLORS = dict(zip(style_LAYERS, ['#1f77b4', '#ff7f0e', '#279e68', '#d62728', '#aa40fc', '#8c564b', '#e377c2']))
style_UNLABELED = '#bdbdbd'

@style_dataclass
class style_Style:
    dpi: int = 600
    formats: tuple = ('png', 'pdf', 'svg')
    panel_inches: float = 2.8
    umap_point_size: float = 6.0
    spot_radius_fraction: float = 0.49
    label_fontsize: float = 8.0
    spatial_background: str = '#aaafae'
    flip_embedding_y: bool = True
    paga_node_size: float = 140.0

def style_pyplot():
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10, 'axes.linewidth': 0.8, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
    return plt

def style_file_hash(path):
    h = style_hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def style_array_hash(array):
    a = style_np.ascontiguousarray(array)
    return style_hashlib.sha256(str((a.shape, str(a.dtype))).encode() + a.tobytes()).hexdigest()

def style_match_clusters(pred, truth, mask):
    """One global one-to-one color assignment; NEVER relabel individual spots."""
    groups = sorted(set(pred), key=lambda s: (0, int(s)) if s.isdigit() else (1, s))
    counts = style_np.array([[style_np.sum((pred[mask] == p) & (truth[mask] == layer)) for layer in style_LAYERS] for p in groups])
    rows, cols = style_linear_sum_assignment(-counts)
    mapping = {groups[r]: style_LAYERS[c] for r, c in zip(rows, cols)}
    colors = {group: style_COLORS[mapping[group]] if group in mapping else '#777777' for group in groups}
    names = {group: mapping.get(group, f'Cluster {group}').replace('_', ' ') for group in groups}
    records = []
    for r, group in enumerate(groups):
        layer = mapping.get(group)
        n = int(style_np.sum((pred == group) & mask))
        overlap = int(counts[r, style_LAYERS.index(layer)]) if layer else 0
        records.append({'cluster': group, 'display_matched_layer': layer or 'unmatched', 'hex_color': colors[group], 'labeled_cluster_spots': n, 'matched_overlap': overlap, 'matched_fraction': overlap / n if n else 0, 'meaning': 'Display alignment using annotations; predicted membership is unchanged'})
    return (colors, names, records)

def style_frame(ax, xy, title, spatial=False, radius=0, background='white', flip_y=False):
    low, high = (style_np.min(xy, axis=0), style_np.max(xy, axis=0))
    center = (low + high) / 2
    width = max(float(style_np.max(high - low)) + 2 * radius, 1e-06) * 1.09
    ax.set_xlim(center[0] - width / 2, center[0] + width / 2)
    ax.set_ylim(center[1] - width / 2, center[1] + width / 2)
    if spatial or flip_y:
        ax.invert_yaxis()
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(background)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(0.8)
    ax.set_title(title, pad=6)

def style_draw_labels(ax, centers, names, fontsize):
    """Center labels on their actual group/node centers, without offsets or guides."""
    from matplotlib import patheffects
    for key, center in centers.items():
        text = ax.text(*center, names[key], ha='center', va='center', fontsize=fontsize, weight='bold', color='black', zorder=6)
        text.set_path_effects([patheffects.withStroke(linewidth=2.0, foreground='white')])

def style_spatial(ax, xy, labels, palette, title, style):
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Circle
    spacing = float(style_np.median(style_cKDTree(xy).query(xy, k=2)[0][:, 1]))
    if not style_np.isfinite(spacing) or spacing <= 0:
        raise ValueError('Cannot determine positive spot spacing')
    radius = spacing * style.spot_radius_fraction
    patches = [Circle(point, radius) for point in xy]
    color = [palette.get(label, style_UNLABELED) for label in labels]
    ax.add_collection(PatchCollection(patches, facecolors=color, edgecolors='none', linewidths=0, alpha=1))
    style_frame(ax, xy, title, spatial=True, radius=radius, background=style.spatial_background)
    return radius

def style_umap(ax, xy, labels, palette, names, title, style):
    ax.scatter(xy[:, 0], xy[:, 1], s=style.umap_point_size, c=[palette.get(label, style_UNLABELED) for label in labels], edgecolors='none', linewidths=0, alpha=1, rasterized=False)
    style_frame(ax, xy, title, flip_y=style.flip_embedding_y)
    centers = {group: style_np.median(xy[labels == group], axis=0) for group in names if style_np.any(labels == group)}
    style_draw_labels(ax, centers, names, style.label_fontsize)

def style_paga_plot(ax, xy, labels, categories, connectivity, palette, names, threshold, title, style):
    centers = {g: style_np.median(xy[labels == g], axis=0) for g in categories if style_np.any(labels == g)}
    if len(centers) != len(categories):
        raise ValueError('Saved PAGA group order does not match plotted observations')
    edges = []
    for i, source in enumerate(categories):
        for j in range(i + 1, len(categories)):
            target = categories[j]
            weight = float(connectivity[i, j])
            shown = weight >= threshold and weight > 0
            edges.append({'source': source, 'target': target, 'connectivity': weight, 'shown': shown})
            if shown:
                points = style_np.array([centers[source], centers[target]])
                ax.plot(points[:, 0], points[:, 1], color='black', linewidth=0.15 + 4.5 * weight, zorder=1)
    sizes = style_np.array([style_np.sum(labels == g) for g in categories])
    for group in categories:
        ax.scatter(*centers[group], s=style.paga_node_size, color=palette[group], edgecolors='white', linewidths=0.4, zorder=3)
    style_frame(ax, xy, title, flip_y=style.flip_embedding_y)
    style_draw_labels(ax, centers, names, style.label_fontsize)
    nodes = [{'group': g, 'display_label': names[g], 'x': float(centers[g][0]), 'y': float(centers[g][1]), 'n_spots': int(n)} for g, n in zip(categories, sizes)]
    return (edges, nodes)

def style_figure(style, rows=1, cols=1):
    plt = style_pyplot()
    fig, axes = plt.subplots(rows, cols, figsize=(style.panel_inches * cols, style.panel_inches * rows), squeeze=False)
    fig.subplots_adjust(left=0.045, right=0.955, bottom=0.055, top=0.9 if rows == 1 else 0.96, wspace=0.12, hspace=0.22)
    return (fig, axes)

def style_save(fig, stem, style, preview=False):
    if preview:
        fig.text(0.5, 0.008, 'STYLE PREVIEW - SMOKE RUN; NOT A FORMAL RESULT', color='#a00000', fontsize=5.5, ha='center')
    for fmt in style.formats:
        fig.savefig(stem.with_suffix('.' + fmt), dpi=style.dpi, facecolor='white')
    style_pyplot().close(fig)

def style_render_run(run, output=None, style=None, model_name=None):
    import anndata as ad
    from sklearn.metrics import adjusted_rand_score
    run = style_Path(run).resolve()
    renderer_sha256 = style_file_hash(style_Path(__file__))
    style = style or style_Style()
    if not 0 < style.spot_radius_fraction <= 0.5:
        raise ValueError('spot_radius_fraction must be in (0, .5]; do not overlap neighboring spots')
    required = ('result.h5ad', 'metrics.json', 'config.json', 'history.csv')
    missing = [name for name in required if not (run / name).is_file()]
    if missing:
        raise FileNotFoundError(f'Need original server run files, not PNG/PDF alone: {run}; missing {missing}')
    protected = [p for p in run.iterdir() if p.is_file() and p.suffix in ('.json', '.csv', '.h5ad', '.npz', '.pt')]
    before = {p.name: style_file_hash(p) for p in protected}
    metrics = style_json.loads((run / 'metrics.json').read_text(encoding='utf-8'))
    cfg = style_json.loads((run / 'config.json').read_text(encoding='utf-8'))
    if metrics.get('status') != 'complete':
        raise ValueError('Only completed saved runs can be rendered')
    output = style_Path(output).resolve() if output else run / 'paper_style'
    if output == run or not output.name:
        raise ValueError('Use a separate output directory; preserve original audit figures')
    output.mkdir(parents=True, exist_ok=True)
    a = ad.read_h5ad(run / 'result.h5ad', backed='r')
    try:
        pred = a.obs['prediction'].astype(str).to_numpy()
        truth = a.obs['ground_truth'].astype(object).to_numpy()
        mask = a.obs['evaluation_mask'].to_numpy(dtype=bool)
        if set(truth[mask]) - set(style_LAYERS):
            raise ValueError('Unexpected ground-truth layers; this palette is for DLPFC Layer_1..6/WM')
        actual_ari = float(adjusted_rand_score(truth[mask].astype(str), pred[mask]))
        if abs(actual_ari - metrics['ARI']) > 1e-10 or int(mask.sum()) != metrics['n_evaluated']:
            raise ValueError('Saved annotations/predictions do not reproduce metrics.json')
        xy = style_np.asarray(a.obsm['spatial'])[:, ::-1].copy()
        embedding = style_np.asarray(a.obsm['X_umap']).copy()
        if xy.shape != (a.n_obs, 2) or embedding.shape != (a.n_obs, 2):
            raise ValueError('Expected saved 2D spatial/UMAP coordinates')
        if not style_np.isfinite(xy).all() or not style_np.isfinite(embedding).all():
            raise ValueError('Non-finite saved coordinates')
        palette, names, mapping = style_match_clusters(pred, truth, mask)
        style_pd.DataFrame(mapping).to_csv(output / 'display_color_mapping.csv', index=False)
        boundary_cfg = cfg['boundary']
        active_weights = {key: float(metrics.get(f'{key}_weight', boundary_cfg[f'{key}_weight'])) for key in ('intra', 'inter')}
        model_name = model_name or 'Ours'
        preview = metrics.get('split') == 'smoke' or cfg.get('smoke', False)
        title = f'{model_name} (ARI={actual_ari:.2f})'
        truth_names = {g: g.replace('_', ' ') for g in style_LAYERS}
        graphs = {}
        for key, labels in (('pred', pred[mask]), ('truth', truth[mask].astype(str))):
            saved = a.uns[f'paga_{key}']
            matrix = saved['connectivities']
            matrix = matrix.toarray() if hasattr(matrix, 'toarray') else style_np.asarray(matrix)
            categories = list(map(str, saved['group_order']))
            if matrix.shape != (len(categories), len(categories)):
                raise ValueError('Invalid saved PAGA matrix')
            graphs[key] = {'labels': labels, 'categories': categories, 'connectivity': matrix.copy(), 'threshold': float(saved.get('threshold', cfg['evaluation']['paga_threshold']))}
        if 'paga_evaluation_barcodes' in a.uns and (not style_np.array_equal(style_np.asarray(a.uns['paga_evaluation_barcodes']).astype(str), style_np.asarray(a.obs_names[mask]).astype(str))):
            raise ValueError('Saved PAGA barcode order differs from evaluation mask')
        for key, labels, colors, label_names in (('pred', pred, palette, names), ('truth', truth, style_COLORS, truth_names)):
            panel_title = title if key == 'pred' else 'Ground truth'
            fig, axes = style_figure(style)
            style_spatial(axes[0, 0], xy, labels, colors, panel_title, style)
            style_save(fig, output / f'spatial_{key}', style, preview)
            fig, axes = style_figure(style)
            style_umap(axes[0, 0], embedding[mask], labels[mask], colors, label_names, panel_title, style)
            style_save(fig, output / f'umap_{key}', style, preview)
            graph = graphs[key]
            fig, axes = style_figure(style)
            edges, nodes = style_paga_plot(axes[0, 0], embedding[mask], **graph, palette=colors, names=label_names, title='PAGA / ' + (model_name if key == 'pred' else 'annotated layers'), style=style)
            style_save(fig, output / f'paga_{key}', style, preview)
            style_pd.DataFrame(edges).to_csv(output / f'paga_{key}_edges.csv', index=False)
            style_pd.DataFrame(nodes).to_csv(output / f'paga_{key}_nodes.csv', index=False)
            fig, axes = style_figure(style, rows=3)
            style_spatial(axes[0, 0], xy, pred, palette, title, style)
            style_umap(axes[1, 0], embedding[mask], labels[mask], colors, label_names, 'UMAP / ' + ('predicted groups' if key == 'pred' else 'annotated layers'), style)
            style_paga_plot(axes[2, 0], embedding[mask], **graph, palette=colors, names=label_names, title='PAGA / ' + ('predicted groups' if key == 'pred' else 'annotated layers'), style=style)
            style_save(fig, output / f'paper_column_{key}', style, preview)
        for kind in ('spatial', 'umap', 'paga'):
            fig, axes = style_figure(style, cols=2)
            for col, key, labels, colors, label_names in ((0, 'pred', pred, palette, names), (1, 'truth', truth, style_COLORS, truth_names)):
                panel_title = title if key == 'pred' else 'Ground truth'
                if kind == 'spatial':
                    style_spatial(axes[0, col], xy, labels, colors, panel_title, style)
                elif kind == 'umap':
                    style_umap(axes[0, col], embedding[mask], labels[mask], colors, label_names, panel_title, style)
                else:
                    style_paga_plot(axes[0, col], embedding[mask], **graphs[key], palette=colors, names=label_names, title='PAGA / ' + ('predicted groups' if key == 'pred' else 'annotated layers'), style=style)
            style_save(fig, output / f'{kind}_compare', style, preview)
        history = style_pd.read_csv(run / 'history.csv')
        fig, axes = style_figure(style, cols=2)
        for column, color in (('total', '#1f77b4'), ('reconstruction', '#ff7f0e')):
            axes[0, 0].plot(history['epoch'], history[column], color=color, lw=1.4, label=column)
        for column, coefficient, color in (('intra', active_weights['intra'], '#279e68'), ('inter', active_weights['inter'], '#aa40fc')):
            weighted = history['structure_scale'] * coefficient * history[column]
            axes[0, 1].plot(history['epoch'], weighted, color=color, lw=1.4, label=f'{column} (weight={coefficient:g})')
        for ax, label in zip(axes.flat, ('Reconstruction and total', 'Weighted structure terms')):
            from matplotlib.ticker import MaxNLocator
            ax.set_title(label)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.tick_params(labelsize=7)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3), useMathText=True)
            ax.legend(frameon=False, fontsize=6.5)
            ax.spines[['top', 'right']].set_visible(False)
        fig.subplots_adjust(left=0.11, right=0.97, bottom=0.19, top=0.85, wspace=0.45)
        style_save(fig, output / 'loss_curve', style, preview)
        if 'boundary_score' in a.obs:
            from matplotlib.collections import PatchCollection
            from matplotlib.patches import Circle
            fig = style_pyplot().figure(figsize=(style.panel_inches, style.panel_inches))
            ax = fig.add_axes([0.05, 0.12, 0.74, 0.74])
            colorbar_ax = fig.add_axes([0.84, 0.12, 0.035, 0.74])
            radius = float(style_np.median(style_cKDTree(xy).query(xy, k=2)[0][:, 1])) * style.spot_radius_fraction
            collection = PatchCollection([Circle(point, radius) for point in xy], cmap='magma', edgecolors='none')
            collection.set_array(a.obs['boundary_score'].to_numpy())
            collection.set_clim(0, 1)
            ax.add_collection(collection)
            style_frame(ax, xy, 'Boundary prior', spatial=True, radius=radius)
            fig.colorbar(collection, cax=colorbar_ax).ax.tick_params(labelsize=6)
            style_save(fig, output / 'boundary_map', style, preview)
        from matplotlib.lines import Line2D
        fig = style_pyplot().figure(figsize=(2.8, 0.85))
        handles = [Line2D([], [], marker='o', ls='', color=style_COLORS[g], label=truth_names[g], markersize=5) for g in style_LAYERS]
        fig.legend(handles=handles, loc='center', ncol=4, frameon=False, fontsize=7, columnspacing=0.6, handletextpad=0.2)
        style_save(fig, output / 'layer_legend', style, preview=False)
        proof = {'source_run': str(run), 'config_id': metrics.get('config_id'), 'seed': metrics['seed'], 'source_model_variant': metrics['variant'], 'display_model_name': model_name, 'renderer_sha256': renderer_sha256, 'ARI_before': metrics['ARI'], 'ARI_recomputed': actual_ari, 'n_spatial_spots': a.n_obs, 'n_umap_paga_spots': int(mask.sum()), 'style_preview_smoke_only': bool(preview), 'style': style_asdict(style), 'palette': style_COLORS, 'sources_sha256_before': before, 'active_structure_weights': active_weights, 'saved_umap_sha256': style_array_hash(embedding), 'paga_connectivities_sha256': {k: style_array_hash(v['connectivity']) for k, v in graphs.items()}, 'grouping_note': 'truth panels use annotations; pred panels use unchanged cluster memberships with annotation-matched display names', 'layout_note': 'PAGA nodes use original group UMAP medians; UMAP and PAGA y axes are inverted for display only when flip_embedding_y is true; text stays upright; no graph recomputation', 'label_note': 'UMAP labels and PAGA labels are centered at the group UMAP medians; no text offsets, repulsion or leader lines; PAGA nodes all use the same marker area', 'references': ['https://stagate.readthedocs.io/en/latest/T1_DLPFC.html', 'https://github.com/scverse/scanpy/blob/1.10.4/src/scanpy/plotting/palettes.py'], 'spatial_note': 'opaque circles at observed spots, radius .49 x median nearest-spot distance; solid reference-matched gray axes background; no smoothing or histology image'}
    finally:
        a.file.close()
    after = {p.name: style_file_hash(p) for p in protected}
    if before != after:
        raise RuntimeError('A source artifact changed during rendering')
    proof['source_files_unchanged'] = True
    (output / 'plot_provenance.json').write_text(style_json.dumps(proof, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'PAPER STYLE: {run.name} -> {output}; ARI={actual_ari:.6f}; source artifacts unchanged', flush=True)
    return output
style_module = SimpleNamespace(__file__=__file__, COLORS=style_COLORS, LAYERS=style_LAYERS, Path=style_Path, Style=style_Style, UNLABELED=style_UNLABELED, array_hash=style_array_hash, asdict=style_asdict, cKDTree=style_cKDTree, dataclass=style_dataclass, draw_labels=style_draw_labels, figure=style_figure, file_hash=style_file_hash, frame=style_frame, hashlib=style_hashlib, json=style_json, linear_sum_assignment=style_linear_sum_assignment, match_clusters=style_match_clusters, np=style_np, paga_plot=style_paga_plot, pd=style_pd, pyplot=style_pyplot, render_run=style_render_run, save=style_save, spatial=style_spatial, umap=style_umap)



# scripts\export_dlpfc.py (inlined; no dependency on the old project)
import argparse as plot_argparse
import json as plot_json
import sys as plot_sys
from pathlib import Path as plot_Path
import numpy as plot_np
import pandas as plot_pd
from scipy import sparse as plot_sparse
from sklearn.metrics import adjusted_rand_score as plot_adjusted_rand_score, normalized_mutual_info_score as plot_normalized_mutual_info_score, homogeneity_score as plot_homogeneity_score
plot_ROOT = protocol_module.ROOT
plot_verify_config = protocol_module.verify_config
from framework.io import read_json as plot_read_json, sha256_file as plot_sha256_file, write_json as plot_write_json
plot_Style = style_module.Style
plot_COLORS = style_module.COLORS
plot_LAYERS = style_module.LAYERS
plot_array_hash = style_module.array_hash
plot_figure = style_module.figure
plot_match_clusters = style_module.match_clusters
plot_spatial = style_module.spatial
plot_umap = style_module.umap
plot_paga_plot = style_module.paga_plot
plot_save = style_module.save
plot_REFERENCE = 'https://stagate.readthedocs.io/en/latest/T1_DLPFC.html'

def plot_official_paga(adata, cfg):
    """Preserve the full-data latent graph when subsetting to labeled spots."""
    import scanpy as sc
    if 'connectivities' not in adata.obsp or 'distances' not in adata.obsp:
        sc.pp.neighbors(adata, n_neighbors=cfg['neighbors'], use_rep='X_model', metric='euclidean', random_state=cfg['plot_seed'])
    if 'X_umap' not in adata.obsm:
        sc.tl.umap(adata, min_dist=cfg['umap_min_dist'], random_state=cfg['plot_seed'], init_pos='spectral')
    mask = adata.obs['evaluation_mask'].to_numpy(dtype=bool)
    used = adata[mask].copy()
    graphs = {}
    for key, group in (('truth', 'ground_truth'), ('pred', 'prediction')):
        used.obs[group] = used.obs[group].cat.remove_unused_categories()
        sc.tl.paga(used, groups=group)
        graphs[key] = {'connectivity': used.uns['paga']['connectivities'].toarray(), 'categories': used.obs[group].cat.categories.astype(str).to_list(), 'labels': used.obs[group].astype(str).to_numpy()}
    return (used, graphs)

def plot_export_run(run, output=None, dpi=600):
    import anndata as ad
    from filelock import FileLock
    from matplotlib.lines import Line2D
    pyplot = style_module.pyplot
    run = plot_Path(run).resolve()
    output = plot_Path(output).resolve() if output else run / 'replacement'
    if output == run or output in run.parents:
        raise ValueError('Export to a dedicated child or separate directory.')
    required = ('result.h5ad', 'config.json', 'metrics.json', 'history.csv')
    if any((not (run / name).is_file() for name in required)):
        raise FileNotFoundError('Supply a complete server run with result.h5ad, metrics.json, config.json and history.csv.')
    cfg, m = (plot_read_json(run / 'config.json'), plot_read_json(run / 'metrics.json'))
    plot_verify_config(cfg)
    if m.get('status') != 'complete':
        raise ValueError('A completed run is required.')
    protected = [p for p in run.iterdir() if p.is_file() and p.suffix in ('.h5ad', '.json', '.npz', '.csv', '.pt')]
    before = {p.name: plot_sha256_file(p) for p in protected}
    a = ad.read_h5ad(run / 'result.h5ad')
    mask = a.obs['evaluation_mask'].to_numpy(dtype=bool)
    truth = a.obs['ground_truth'].astype(object).to_numpy()
    pred = a.obs['prediction'].astype(str).to_numpy()
    if (a.n_obs, a.n_vars, int(mask.sum())) != (3460, 3000, 3431):
        raise ValueError("This exporter targets DLPFC 151676's 3460 spots, 3000 HVGs and 3431 labeled spots.")
    if set(truth[mask]) - set(plot_LAYERS):
        raise ValueError('Unexpected DLPFC annotation names.')
    values = dict(ARI=plot_adjusted_rand_score(truth[mask].astype(str), pred[mask]), NMI=plot_normalized_mutual_info_score(truth[mask].astype(str), pred[mask]), HS=plot_homogeneity_score(truth[mask].astype(str), pred[mask]))
    if m['n_evaluated'] != int(mask.sum()) or any((abs(values[k] - m[k]) > 1e-10 for k in values)):
        raise ValueError('Saved predictions and truth do not reproduce the recorded metrics.')
    xy = plot_np.asarray(a.obsm['spatial'])[:, ::-1].copy()
    z = plot_np.asarray(a.obsm['X_model'])
    if xy.shape != (3460, 2) or z.shape != (3460, 30) or (not plot_np.isfinite(z).all()) or (not plot_np.isfinite(xy).all()):
        raise ValueError('Unexpected latent or spatial coordinates.')
    style = plot_Style(dpi=dpi)
    preview = bool(cfg.get('smoke'))
    used, graphs = plot_official_paga(a, cfg['evaluation'])
    embedding = plot_np.asarray(a.obsm['X_umap'])
    if embedding.shape != (3460, 2) or not plot_np.isfinite(embedding).all():
        raise ValueError('Expected finite saved UMAP coordinates.')
    palette, names, mappings = plot_match_clusters(pred, truth, mask)
    truth_names = {g: g.replace('_', ' ') for g in plot_LAYERS}
    title = f"Ours (ARI={values['ARI']:.2f})"
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / 'export.lock'), timeout=0):
        plot_write_json(output / 'export_status.json', {'status': 'running', 'source': str(run)})
        plot_pd.DataFrame(mappings).to_csv(output / 'display_color_mapping.csv', index=False)
        plot_pd.DataFrame([dict(dataset='DLPFC_151676', seed=m['seed'], **values, n_evaluated=int(mask.sum()), n_spatial=3460)]).to_csv(output / 'metrics.csv', index=False)
        plot_pd.DataFrame(embedding, index=a.obs_names, columns=['UMAP1', 'UMAP2']).to_csv(output / 'umap_coordinates.csv', index_label='barcode')
        plot_pd.DataFrame(z, index=a.obs_names).to_csv(output / 'embedding.csv', index_label='barcode')
        a.obs[['ground_truth', 'prediction', 'evaluation_mask']].to_csv(output / 'predictions.csv', index_label='barcode')
        plot_pd.Series(used.obs_names, name='barcode').to_csv(output / 'paga_barcodes.csv', index=False)
        plot_sparse.save_npz(output / 'paga_induced_connectivities.npz', used.obsp['connectivities'])
        plot_sparse.save_npz(output / 'paga_induced_distances.npz', used.obsp['distances'])
        for key, labels, colors, display in (('pred', pred, palette, names), ('truth', truth, plot_COLORS, truth_names)):
            fig, axs = plot_figure(style)
            plot_spatial(axs[0, 0], xy, labels, colors, title if key == 'pred' else 'Ground truth', style)
            plot_save(fig, output / f'spatial_{key}', style, preview)
            fig, axs = plot_figure(style)
            selected = plot_np.ones(a.n_obs, dtype=bool) if key == 'pred' else mask
            plot_umap(axs[0, 0], embedding[selected], labels[selected], colors, display, 'UMAP / predicted clusters' if key == 'pred' else 'UMAP / annotated layers', style)
            plot_save(fig, output / f'umap_{key}', style, preview)
            graph = graphs[key]
            fig, axs = plot_figure(style)
            edges, nodes = plot_paga_plot(axs[0, 0], embedding[mask], graph['labels'], graph['categories'], graph['connectivity'], colors, display, cfg['evaluation']['paga_threshold'], 'PAGA / predicted clusters' if key == 'pred' else 'PAGA / annotated layers', style)
            plot_save(fig, output / f'paga_{key}', style, preview)
            plot_pd.DataFrame(graph['connectivity'], index=graph['categories'], columns=graph['categories']).to_csv(output / f'paga_{key}_connectivities.csv')
            plot_pd.DataFrame(edges).to_csv(output / f'paga_{key}_edges.csv', index=False)
            plot_pd.DataFrame(nodes).to_csv(output / f'paga_{key}_nodes.csv', index=False)
            fig, axs = plot_figure(style, rows=3)
            plot_spatial(axs[0, 0], xy, pred, palette, title, style)
            plot_umap(axs[1, 0], embedding[mask], labels[mask], colors, display, 'UMAP / predicted clusters' if key == 'pred' else 'UMAP / annotated layers', style)
            plot_paga_plot(axs[2, 0], embedding[mask], graph['labels'], graph['categories'], graph['connectivity'], colors, display, cfg['evaluation']['paga_threshold'], 'PAGA / predicted clusters' if key == 'pred' else 'PAGA / annotated layers', style)
            plot_save(fig, output / f'paper_column_{key}', style, preview)
        fig, axs = plot_figure(style, cols=2)
        plot_spatial(axs[0, 0], xy, pred, palette, title, style)
        plot_spatial(axs[0, 1], xy, truth, plot_COLORS, 'Ground truth', style)
        plot_save(fig, output / 'spatial_compare', style, preview)
        fig, axs = plot_figure(style, cols=2)
        plot_umap(axs[0, 0], embedding, pred, palette, names, title, style)
        plot_umap(axs[0, 1], embedding[mask], truth[mask], plot_COLORS, truth_names, 'Ground truth', style)
        plot_save(fig, output / 'umap_compare', style, preview)
        fig, axs = plot_figure(style, cols=2)
        plot_umap(axs[0, 0], embedding[mask], truth[mask], plot_COLORS, truth_names, 'Annotated layers', style)
        graph = graphs['truth']
        plot_paga_plot(axs[0, 1], embedding[mask], graph['labels'], graph['categories'], graph['connectivity'], plot_COLORS, truth_names, cfg['evaluation']['paga_threshold'], 'PAGA', style)
        plot_save(fig, output / 'paga_compare_truth', style, preview)
        fig = pyplot().figure(figsize=(2.8, 0.85))
        handles = [Line2D([], [], marker='o', ls='', color=plot_COLORS[g], label=truth_names[g], markersize=5) for g in plot_LAYERS]
        fig.legend(handles=handles, loc='center', ncol=4, frameon=False, fontsize=7, columnspacing=0.6, handletextpad=0.2)
        plot_save(fig, output / 'layer_legend', style, preview)
        manifest = [('Tutorial1_ARI', '#running-stagate', 'metrics.csv', 'ARI on 3431 manually labeled spots'), ('Tutorial1_spatial_Ours', '#running-stagate', 'spatial_pred', 'Unchanged clustering; colors aligned to annotations'), ('Tutorial1_spatial_comparison', '#running-stagate', 'spatial_compare', 'Ours and manual layers'), ('Tutorial1_UMAP_comparison', '#running-stagate', 'umap_compare', 'Full-data UMAP, prediction and truth colors'), ('Tutorial1_PAGA', '#spatial-trajectory-inference-paga', 'paga_compare_truth', 'Manual groups, full latent graph subset'), ('Paper_DLPFC_domain_panel', '#running-stagate', 'spatial_pred', 'Gray background, Ours title, original memberships'), ('Paper_DLPFC_UMAP_panel', '#running-stagate', 'umap_truth', 'Manual layers on the R2-A embedding'), ('Paper_DLPFC_PAGA_panel', '#spatial-trajectory-inference-paga', 'paga_truth', 'Manual layer graph on R2-A latent neighbors'), ('Paper_DLPFC_column', '', 'paper_column_truth', 'Predicted spatial domains; annotated UMAP and PAGA')]
        plot_pd.DataFrame([dict(reference=item, source_url=plot_REFERENCE + anchor, output=stem, formats='csv' if stem.endswith('.csv') else 'png,pdf,svg', meaning=meaning) for item, anchor, stem, meaning in manifest]).to_csv(output / 'replacement_manifest.csv', index=False)
        after = {p.name: plot_sha256_file(p) for p in protected}
        if before != after:
            raise RuntimeError('Source result artifacts changed during export.')
        plot_write_json(output / 'provenance.json', {'source_run': str(run), 'source_config_id': m['config_id'], 'seed': m['seed'], 'metrics_recomputed': values, 'n_spatial': a.n_obs, 'n_paga': used.n_obs, 'exporter_sha256': plot_sha256_file(plot_Path(__file__)), 'plot_library_sha256': plot_sha256_file(Path(__file__)), 'umap_sha256': plot_array_hash(embedding), 'paga_connectivities_sha256': {k: plot_array_hash(v['connectivity']) for k, v in graphs.items()}, 'source_artifacts_sha256': before, 'source_artifacts_unchanged': True, 'reference_url': plot_REFERENCE, 'smoke_only': preview, 'paga_protocol': 'Compute neighbors on all 3460 spots, subset induced graph to 3431 annotated spots, then PAGA.', 'grouping': 'truth uses manual layers; pred uses unchanged mclust memberships. Both graphs are exported.', 'colors': 'One global annotation-matched cluster palette for presentation only.', 'display': 'Gray spatial axes, opaque spots, centered labels, identical PAGA node areas; flip UMAP/PAGA y for display.', 'backend_note': 'R2-A uses the frozen Round2 PyTorch runtime and 1000 epochs. T1 reports TensorFlow and 500 epochs.', 'comparison_scope': 'Dataset, annotations, spatial radius and downstream grouping follow T1. R2-A replaces the learned representation.'})
        plot_write_json(output / 'export_status.json', {'status': 'complete', 'source': str(run), 'smoke_only': preview})
    print(f"EXPORTED: {output}; ARI={values['ARI']:.6f}")
    return output
plot_module = SimpleNamespace(__file__=__file__, COLORS=plot_COLORS, LAYERS=plot_LAYERS, Path=plot_Path, REFERENCE=plot_REFERENCE, ROOT=plot_ROOT, Style=plot_Style, adjusted_rand_score=plot_adjusted_rand_score, argparse=plot_argparse, array_hash=plot_array_hash, export_run=plot_export_run, figure=plot_figure, homogeneity_score=plot_homogeneity_score, json=plot_json, match_clusters=plot_match_clusters, normalized_mutual_info_score=plot_normalized_mutual_info_score, np=plot_np, official_paga=plot_official_paga, paga_plot=plot_paga_plot, pd=plot_pd, read_json=plot_read_json, save=plot_save, sha256_file=plot_sha256_file, sparse=plot_sparse, spatial=plot_spatial, sys=plot_sys, umap=plot_umap, verify_config=plot_verify_config, write_json=plot_write_json)


def verify(cfg):
    protocol_verify_config(cfg)
def render(run,args):
    global plot_Style
    plot_Style=lambda **kw: style_Style(**(dict(umap_point_size=args.umap_size,label_fontsize=args.label_size)|kw))
    plot_export_run(run,output=Path(run)/'figures',dpi=args.dpi)


def plot_training_diagnostics(run,args):
    """Local diagnostics from saved outputs; no model, clustering or coordinate mutation."""
    import numpy as np
    import pandas as pd
    import anndata as ad
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    run=Path(run); output=run/'figures'
    cfg=unified_read_json(run/'config.json')
    history=pd.read_csv(run/'history.csv');smoke=bool(cfg.get('smoke'))
    def save_diagnostic(fig,stem):
        if smoke:fig.text(.5,.005,'STYLE PREVIEW - SMOKE RUN; NOT A FORMAL RESULT',ha='center',color='#a00000',fontsize=6)
        for ext in ('png','pdf','svg'):fig.savefig(output/(stem+'.'+ext),dpi=args.dpi,facecolor='white')
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(8,3))
    for key in ('total','reconstruction'):axes[0].plot(history.epoch,history[key],label=key)
    for key in ('intra','inter'):axes[1].plot(history.epoch,history[key],label=key)
    axes[0].set_title('Training objective');axes[1].set_title('Unweighted structural losses')
    for ax in axes:ax.set_xlabel('Epoch');ax.legend(frameon=False)
    fig.tight_layout(rect=[0,.035,1,1]);save_diagnostic(fig,'loss_curve')
    a=ad.read_h5ad(run/'result.h5ad',backed='r')
    try:
        xy=np.asarray(a.obsm['spatial']).copy();scores=a.obs.boundary_score.to_numpy(dtype=float)
        barcodes=np.asarray(a.obs_names).astype(str)
    finally:a.file.close()
    pd.DataFrame(dict(barcode=barcodes,boundary_prior=scores)).to_csv(output/'boundary_scores.csv',index=False)
    xy=xy[:,[1,0]]
    fig,ax=plt.subplots(figsize=(4,4))
    dots=ax.scatter(xy[:,0],xy[:,1],c=scores,s=6,marker='.',cmap='magma',vmin=0,vmax=1,linewidths=0)
    ax.set_aspect('equal');ax.axis('off');ax.set_title('Expression-derived boundary prior')
    ax.invert_yaxis()
    fig.colorbar(dots,ax=ax,shrink=.7,label='Mean incident-edge boundary probability')
    fig.tight_layout(rect=[0,.035,1,1]);save_diagnostic(fig,'boundary_map')

def render_complete(run,args):
    render(run,args)
    plot_training_diagnostics(run,args)

SPEC = Experiment(experiment_id='R2A_dataset_reproduction',kind='dlpfc',root=EXPERIMENT_DIR,
    config=DEFAULT_CONFIG,visual_defaults=VISUAL_DEFAULTS,prepare=data_prepare,verify=verify,
    render=render_complete,tag='dlpfc',control=None,comparison=None)

if __name__=='__main__':
    experiment_main(SPEC)

