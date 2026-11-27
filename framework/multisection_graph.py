"""Official Tutorial5 topology: within-section XY radii and adjacent-section XY radii."""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from .graphs import canonical_edges,graph_summary

SECTION_ORDER = [f'Puck_180531_{i}' for i in [13,16,17,18,19,22,23]]
SECTION_COUNTS = [1692,1476,1138,1363,1788,1835,1616]
WITHIN_COUNTS = [22974,20546,11794,18278,22344,24544,18922]
CROSS_COUNTS = [34482,23078,25284,24944,26128,21110]


def build_hippo_graphs(adata,radius=50.,cross_radius=50.):
    xy = np.asarray(adata.obsm['spatial'],dtype=float)
    section = adata.obs.Section_id.astype(str).to_numpy()
    if xy.shape!=(adata.n_obs,2) or not np.isfinite(xy).all() or set(section)!=set(SECTION_ORDER):
        raise ValueError('Expected all seven sections with valid aligned XY coordinates')
    within,cross,within_rows,cross_rows = [],[],[],[]
    for name in SECTION_ORDER:
        indices = np.flatnonzero(section==name)
        pairs = cKDTree(xy[indices]).query_pairs(radius,output_type='ndarray')
        if len(pairs):
            pairs = pairs[np.linalg.norm(xy[indices[pairs[:,0]]]-xy[indices[pairs[:,1]]],axis=1)>0]
        within.append(indices[pairs])
        within_rows.append(dict(section=name,n_spots=len(indices),directed_nonself_edges=2*len(pairs)))
    for first,second in zip(SECTION_ORDER[:-1],SECTION_ORDER[1:]):
        left,right = np.flatnonzero(section==first),np.flatnonzero(section==second)
        neighbors = cKDTree(xy[left]).query_ball_tree(cKDTree(xy[right]),cross_radius)
        pairs = np.asarray([(int(left[i]),int(right[j])) for i,row in enumerate(neighbors) for j in row],dtype=np.int64).reshape(-1,2)
        if len(pairs):
            pairs = pairs[np.linalg.norm(xy[pairs[:,0]]-xy[pairs[:,1]],axis=1)>0]
        cross.append(pairs)
        cross_rows.append(dict(section1=first,section2=second,directed_nonself_edges=2*len(pairs)))
    within_pairs,cross_pairs = np.concatenate(within),np.concatenate(cross)
    graph_2d = canonical_edges(within_pairs,adata.n_obs)
    graph_3d = canonical_edges(np.concatenate([within_pairs,cross_pairs]),adata.n_obs)
    report = dict(within_sections=within_rows,adjacent_pairs=cross_rows,
        graph_2d=graph_summary(graph_2d,adata.n_obs),graph_3d=graph_summary(graph_3d,adata.n_obs),
        cross_directed_nonself_edges=2*len(cross_pairs),section_order=SECTION_ORDER,
        coordinate_rule='Aligned XY Euclidean distance, not XYZ distance; only declared adjacent sections',
        radius=radius,cross_radius=cross_radius)
    if adata.n_obs==10908 and radius==50 and cross_radius==50:
        if [r['n_spots'] for r in within_rows]!=SECTION_COUNTS or [r['directed_nonself_edges'] for r in within_rows]!=WITHIN_COUNTS:
            raise ValueError('Within-section counts differ from official Tutorial5')
        if [r['directed_nonself_edges'] for r in cross_rows]!=CROSS_COUNTS:
            raise ValueError('Adjacent-section edge counts differ from official Tutorial5')
        if report['graph_2d']['directed_nonself_edges']!=139402 or report['graph_3d']['directed_nonself_edges']!=294428:
            raise ValueError('Expected official Tutorial5 graph sizes')
    return {'2d':graph_2d,'3d':graph_3d},report


def graph_for_cfg(adata,cfg):
    graphs,_ = build_hippo_graphs(adata,cfg['graph']['radius'],cfg['graph']['cross_radius'])
    return graphs[cfg['graph']['mode']]
