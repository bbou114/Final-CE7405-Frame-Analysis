"""CE 7405 | Step 6: optional one-at-a-time beam-end sensitivity mapping.

For each eligible *beam end*, change only that end to the specified trial
connection, re-solve the full frame, and compare the maximum interstory drift
ratio to the *selected/manual* baseline. Nothing is optimized or auto-applied.

Positive influence = decreased maximum drift, negative = increased drift.
The normal score is relative drift reduction (%). If baseline drift is zero,
relative changes are undefined; use the absolute drift change (percentage
points) instead.

Uses the Step 4 solver exactly as it stands. Units are SI internally.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Iterable
import numpy as np

from semirigid_connections import (
    ConnectionKey, ConnectionSetting, ConnectedFrameResult,
    all_beam_ends, analyze_connected_building, connection,
)
from multistory_frame import BuildingModel


@dataclass(frozen=True)
class ConnectionTrial:
    key: ConnectionKey
    starting_kind: str
    starting_alpha: float
    trial_kind: str
    trial_alpha: float
    status: str  # 'ok', 'unchanged', or 'invalid'
    baseline_drift: float
    trial_drift: float | None
    drift_reduction_pp: float | None
    improvement_percent: float | None
    roof_displacement_m: float | None
    worst_story: int | None
    error: str = ""


@dataclass
class SensitivityStudy:
    baseline: ConnectedFrameResult
    records: list[ConnectionTrial]
    trial_setting: ConnectionSetting
    baseline_settings: dict[ConnectionKey, ConnectionSetting]

    @property
    def relative_score_available(self) -> bool:
        return self.baseline.max_story_drift_ratio > 1e-12

    @property
    def successful(self) -> list[ConnectionTrial]:
        """Sorted from greatest to smallest reduction in maximum drift."""
        return sorted((r for r in self.records if r.status == 'ok'),
                      key=lambda r: r.drift_reduction_pp, reverse=True)


def map_connection_sensitivity(
    model: BuildingModel,
    baseline_settings: Mapping[ConnectionKey, ConnectionSetting] | None = None,
    trial_setting: ConnectionSetting | None = None,
    eligible_keys: Iterable[ConnectionKey] | None = None,
    baseline_result: ConnectedFrameResult | None = None,
) -> SensitivityStudy:
    """Evaluate one trial setting at each eligible beam end, independently.

    - Baseline is the user's currently selected configuration, NOT necessarily
      the all-rigid frame.
    - Each candidate starts afresh from the baseline settings.
    - Unchanged candidates are skipped rather than given a misleading rank.
    - Unstable/singular trial configurations are reported as invalid.
    - Results never modify the user-supplied settings.
    """
    baseline_settings = dict(baseline_settings or {})
    if trial_setting is None:
        trial_setting = connection('semi', alpha=5.0)
    if not isinstance(trial_setting, ConnectionSetting):
        raise TypeError('trial_setting must be a ConnectionSetting.')

    available = set(all_beam_ends(model))
    if eligible_keys is None:
        candidates = all_beam_ends(model)
    else:
        candidates = list(dict.fromkeys(eligible_keys))
        unknown = [key for key in candidates if key not in available]
        if unknown:
            raise ValueError(f'Beam end does not exist in frame: {unknown!r}')

    unknown_baseline = [key for key in baseline_settings if key not in available]
    if unknown_baseline:
        raise ValueError(f'Baseline includes invalid beam ends: {unknown_baseline!r}')

    if baseline_result is None:
        baseline_result = analyze_connected_building(model, baseline_settings)
    D0 = float(baseline_result.max_story_drift_ratio)
    if not np.isfinite(D0) or D0 < 0:
        raise ValueError('Baseline drift must be nonnegative and finite.')

    records = []
    for key in candidates:
        original = baseline_settings.get(key, connection('rigid'))
        if original == trial_setting:
            records.append(ConnectionTrial(
                key, original.kind, original.alpha, trial_setting.kind,
                trial_setting.alpha, 'unchanged', D0, None, None,
                None, None, None, 'Already in trial state',
            ))
            continue

        # Independently re-solve from original settings, changing ONLY one end.
        modified = baseline_settings.copy()
        if trial_setting.kind == 'rigid':
            modified.pop(key, None)
        else:
            modified[key] = trial_setting
        try:
            response = analyze_connected_building(model, modified)
            Di = float(response.max_story_drift_ratio)
            if not np.isfinite(Di):
                raise ValueError('Nonfinite candidate drift.')
            change = D0 - Di
            records.append(ConnectionTrial(
                key=key, starting_kind=original.kind,
                starting_alpha=original.alpha, trial_kind=trial_setting.kind,
                trial_alpha=trial_setting.alpha, status='ok',
                baseline_drift=D0, trial_drift=Di,
                drift_reduction_pp=100.0 * change,
                improvement_percent=100.0 * change / D0 if D0 > 1e-12 else None,
                roof_displacement_m=float(response.roof_max_displacement),
                worst_story=int(response.worst_story),
            ))
        except (ValueError, np.linalg.LinAlgError) as exc:
            records.append(ConnectionTrial(
                key, original.kind, original.alpha, trial_setting.kind,
                trial_setting.alpha, 'invalid', D0, None, None,
                None, None, None, str(exc),
            ))

    return SensitivityStudy(
        baseline=baseline_result, records=records,
        trial_setting=trial_setting, baseline_settings=baseline_settings,
    )


def sensitivity_table_rows(study: SensitivityStudy) -> list[dict]:
    """CSV / Streamlit-ready output. Score sorting is per actual drift change."""
    ordered = study.successful + [
        r for r in study.records if r.status != 'ok'
    ]
    rows = []
    for r in ordered:
        rows.append({
            'Rank': len(rows) + 1 if r.status == 'ok' else '',
            'Floor': r.key[0], 'Bay': r.key[1], 'End': r.key[2],
            'Before': r.starting_kind +
                (f' (α={r.starting_alpha:g})' if r.starting_kind == 'semi' else ''),
            'Trial': r.trial_kind +
                (f' (α={r.trial_alpha:g})' if r.trial_kind == 'semi' else ''),
            'Status': r.status,
            'Baseline maximum drift (%)': round(100*r.baseline_drift, 7),
            'Trial maximum drift (%)': round(100*r.trial_drift, 7)
                if r.trial_drift is not None else '',
            'Drift reduction (percentage points)': round(r.drift_reduction_pp, 7)
                if r.drift_reduction_pp is not None else '',
            'Relative drift reduction (%)': round(r.improvement_percent, 6)
                if r.improvement_percent is not None else '',
            'Trial roof |Ux| (mm)': round(1000*r.roof_displacement_m, 5)
                if r.roof_displacement_m is not None else '',
            'Error / reason': r.error,
        })
    return rows
