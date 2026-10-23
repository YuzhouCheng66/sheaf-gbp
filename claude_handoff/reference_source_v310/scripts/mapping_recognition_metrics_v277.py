"""CPU raw metrics against each recipient's CAUSALLY reachable observations.

Unknown predictions are never repaired. Unknown target cells are not free.
Door labels are intersected with the CURRENT target, not sample['union'].
Topology evaluates pairs of target-known free cells; predicted paths may use
unsupported decoded free cells, so false shortcuts are not silently masked.
Empty scopes/classes return None, with explicit denominator counts. Erasing
all predicted walls when true walls exist scores precision=F1=0, not None.
"""
import numpy as np

from scripts.mapping_topology_v211 import topology_metrics
from scripts.train_mapping_implicit64_v192 import door_measures


INTERIOR_BORDER = 6
METRIC_SCOPE = {
    'union_accuracy': 'Raw ternary equality over this recipient target>=0; unknown prediction is wrong.',
    'known_precision': 'Correctly predicted known / all predicted known, including unsupported predictions.',
    'known_recall': 'Correctly predicted known / target-known cells.',
    'unknown_known_fpr': 'Predicted known / target-unknown cells; None when no target-unknown cell.',
    'own': 'Original own>=0, required to be a label-consistent subset of current target.',
    'peer': 'Current target>=0 and original own<0; not all globally known cells.',
    'interior': 'Current target-known cells excluding the inherited6-pixel raster border.',
    'door': 'Public evaluation door_mask AND current target>=0; target door cells must be free.',
    'topology': 'v211 exact4-neighbor pair connectivity on target-known free cells; raw predicted paths unmasked.',
    'empty': 'None for empty scopes or absent wall class on both truth/prediction; counts identify eligibility.',
}


def _rate(mask, correct):
    count = int(np.count_nonzero(mask))
    return float(np.count_nonzero(mask & correct) / count) if count else None


def _scope(pred, target, mask):
    cells = int(mask.sum())
    true_wall = (target == 1) & mask
    pred_wall = (pred == 1) & mask
    actual, guessed = int(true_wall.sum()), int(pred_wall.sum())
    correct = int((true_wall & pred_wall).sum())
    return dict(cells=cells, wall_cells=actual, predicted_wall_cells=guessed,
        wall_true_positive_cells=correct, typed_accuracy=_rate(mask, pred == target),
        wall_recall=correct / actual if actual else None,
        wall_precision=correct / guessed if guessed else (0. if actual else None),
        wall_f1=2 * correct / (actual + guessed) if actual + guessed else None)


def node_metrics(prediction, own, targets, sample):
    """Return one numeric/None dictionary per node; no averaging or selection.

    Inputs are integer ternary [N,H,W] arrays; sample is used ONLY for its
    evaluation door_mask. Raw maps and inputs remain untouched. Counts use
    cells rather than rescaled percentages. No model or accelerator is called.
    """
    prediction, own, targets = (np.asarray(value) for value in (prediction, own, targets))
    if (prediction.ndim != 3 or min(prediction.shape) < 1 or own.shape != prediction.shape
            or targets.shape != prediction.shape or any(value.dtype.kind not in 'iu'
            or not np.isin(value, [-1, 0, 1]).all() for value in (prediction, own, targets))):
        raise ValueError('Equal nonempty integer ternary [N,H,W] arrays required')
    if np.any((own >= 0) & (own != targets)):
        raise ValueError('Original own observations must be a consistent subset of each current target')
    if not isinstance(sample, dict) or 'door_mask' not in sample:
        raise ValueError('Explicit evaluation door_mask required; no hidden full-map target fallback')
    door = np.asarray(sample['door_mask'])
    if door.shape != prediction.shape[-2:] or door.dtype != np.bool_:
        raise ValueError('Boolean door mask matching the raster required')
    interior = np.zeros(prediction.shape[-2:], dtype=bool)
    interior[INTERIOR_BORDER:-INTERIOR_BORDER, INTERIOR_BORDER:-INTERIOR_BORDER] = True
    result = []
    for node, pred in enumerate(prediction):
        target = targets[node]
        known, local = target >= 0, own[node] >= 0
        predicted_known = pred >= 0
        known_count, predicted_count = int(known.sum()), int(predicted_known.sum())
        known_true = int((known & predicted_known).sum())
        base = _scope(pred, target, known)
        row = {k:v for k,v in base.items() if k not in ('cells', 'typed_accuracy')}
        row.update(union_accuracy=base['typed_accuracy'], raw_target_union_acc=base['typed_accuracy'],
            raw_target_typed_acc=float(np.mean(pred == target)), target_known_cells=known_count,
            predicted_known_cells=predicted_count, correct_known_cells=known_true,
            target_unknown_cells=int((~known).sum()), unsupported_known_cells=int((~known & predicted_known).sum()),
            known_precision=known_true / predicted_count if predicted_count else (0. if known_count else None),
            known_recall=known_true / known_count if known_count else None,
            unknown_known_fpr=_rate(~known, predicted_known),
            unsupported_false_known=_rate(~known, predicted_known))
        for name, mask in (('own', local), ('peer', known & ~local), ('interior', known & interior),
                           ('own_interior', local & interior), ('peer_interior', known & ~local & interior)):
            row.update({name + '_' + key:value for key,value in _scope(pred, target, mask).items()})
        current_door = door & known
        if np.any(current_door & (target != 0)):
            raise ValueError('An observed evaluation door must be free in the actual target')
        row['door_known_cells'] = int(current_door.sum())
        row.update(door_measures(pred, dict(door_mask=door, union=target)))
        row.update(topology_metrics(pred, target))
        result.append(row)
    return result
