"""Reachable normal-response penalty for an explicitly paired training pilot."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts import v283_supervised_dinomaly_longtrain as trainer

ARMS = ('legacy_probability', 'raw_distance')


def paired_seed(fold_id: int, penalty_space: str) -> int:
    if penalty_space not in ARMS or fold_id not in range(4):
        raise ValueError('Invalid pilot arm or fold')
    return 33201 + int(fold_id)


def supervised_losses(encoded: Sequence[Any], decoded: Sequence[Any], masks: Any, *,
                       out_size: int, penalty_space: str) -> dict[str, Any]:
    if penalty_space not in ARMS:
        raise ValueError('Unknown normal-penalty space')
    losses = trainer.compute_supervised_losses(encoded, decoded, masks, out_size=out_size,
                                               normal_loss_weight=0.0, normal_mean_budget=.006,
                                               normal_quantile_budget=.060)
    penalty = losses['normal_fp']
    if penalty_space == 'raw_distance':
        penalty = trainer.normal_false_positive_loss(losses['map'].clamp_min(0), masks,
                                                      mean_budget=.006, quantile_budget=.060)
    return {**losses, 'normal_fp': penalty, 'total': losses['total'] + .5 * penalty}


def validate_training_rows(rows: Sequence[Mapping[str, Any]], held_out: set[str],
                            approved_root: Path | None) -> None:
    if not rows:
        raise ValueError('Empty training partition')
    for row in rows:
        if row['category'] in held_out:
            raise ValueError('A held-out category entered training')
        for key in ('image_path', 'mask_path'):
            if not row.get(key):
                continue
            resolved = Path(row[key]).resolve()
            if 'test_b' in str(resolved).replace('\\', '/').lower():
                raise ValueError('Test_B is forbidden in training')
            if approved_root is not None and not resolved.is_relative_to(approved_root.resolve()):
                raise ValueError('Training source provenance violation')
