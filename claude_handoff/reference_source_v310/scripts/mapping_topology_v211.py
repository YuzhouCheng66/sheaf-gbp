"""Exact connectivity counts over all pairs of truly observed-free cells.

Unknown is not traversable ground truth. Predicted free paths may use any
decoded known-free cell, so unsupported shortcuts are NOT silently masked.
Counts are combinatorial, not sampled pairs or costly quadratic enumeration.
"""
import numpy as np
from scipy.ndimage import label


def choose2(counts):
    counts=np.asarray(counts,dtype=np.int64)
    return int(np.sum(counts*(counts-1)//2))


def topology_metrics(prediction,target):
    prediction=np.asarray(prediction);target=np.asarray(target)
    if prediction.ndim!=2 or prediction.shape!=target.shape or not np.isin(prediction,[-1,0,1]).all() or not np.isin(target,[-1,0,1]).all():raise ValueError('Equal ternary grids required')
    structure=np.array([[0,1,0],[1,1,1],[0,1,0]],dtype=np.uint8)
    ground,g_count=label(target==0,structure);predicted,p_count=label(prediction==0,structure)
    valid=target==0;g=ground[valid].astype(np.int64);p=predicted[valid].astype(np.int64)
    total=choose2([len(g)]);connected=choose2(np.unique(g,return_counts=True)[1])
    positive=p>0
    both=choose2(np.unique(g[positive]*(p_count+1)+p[positive],return_counts=True)[1])
    claimed=choose2(np.unique(p[positive],return_counts=True)[1])
    false_connected=claimed-both;disconnected=total-connected
    return dict(observed_free_cells=len(g),ground_known_free_components=g_count,predicted_known_free_components=p_count,
        connected_pair_count=connected,disconnected_pair_count=disconnected,
        free_pair_connectivity_recall=both/connected if connected else None,
        spurious_pair_connection_rate=false_connected/disconnected if disconnected else None,
        pair_topology_accuracy=(both+disconnected-false_connected)/total if total else None,
        observed_free_retained_as_free=float(np.mean(positive)) if len(g) else None,
        connectivity_is_four_neighbor=True,unknown_gt_not_traversable=True)
