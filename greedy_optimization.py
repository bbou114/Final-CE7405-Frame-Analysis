"""CE 7405 | Step 7: optional, discrete greedy beam-end connection upgrades.

The objective is maximum interstory drift with an optional target-first
preference for lower connection restraint; this is not actual monetary cost,
strength, global optimality, or building code compliance. Every trial calls
exactly the same verified Step 4 first-order frame solver.

Each eligible beam end may be upgraded at most once per optimization run.
A semi-rigid upgrade uses a user-selected *fixed* normalized stiffness alpha;
rigid upgrades are enforced as exact shared rotations, never a huge spring.

No user model or settings object is mutated and recommendations are not applied
without explicit action in the Streamlit GUI.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping
import numpy as np

from multistory_frame import BuildingModel
from semirigid_connections import (
    ConnectionKey, ConnectionSetting, ConnectedFrameResult,
    all_beam_ends, analyze_connected_building, connection,
)


def _canonical(settings: Mapping[ConnectionKey, ConnectionSetting] | None):
    """Represent rigid ends by absence from the settings mapping."""
    supplied = dict(settings or {})
    if any(not isinstance(value, ConnectionSetting) for value in supplied.values()):
        raise TypeError('Connection settings must contain ConnectionSetting objects.')
    return {k: v for k, v in supplied.items() if v.kind != 'rigid'}


def _upgrade_options(current: ConnectionSetting, allow_semi: bool,
                     upgrade_alpha: float, *, allow_rigid: bool = True,
                     candidate_alphas: tuple[float, ...] | None = None) -> list[ConnectionSetting]:
    """Generate physically admissible stiffening alternatives for one beam end."""
    if current.kind == 'rigid':
        return []
    options: list[ConnectionSetting] = []
    current_alpha = current.alpha if current.kind == 'semi' else 0.0
    if allow_semi:
        levels = candidate_alphas if candidate_alphas is not None else (upgrade_alpha,)
        for alpha in levels:
            if alpha > current_alpha + 1e-10:
                options.append(connection('semi', alpha=alpha))
    if allow_rigid:
        options.append(connection('rigid'))
    return options


def _extra_restraint(before: ConnectionSetting, after: ConnectionSetting) -> float:
    """Change in dimensionless stiffness α; rigid is an unbounded idealization.

    Used only to prefer less restraint *when different trials already meet the
    drift target*. It is not a construction-cost estimate.
    """
    if after.kind == 'rigid':
        return float('inf')
    old_alpha = before.alpha if before.kind == 'semi' else 0.0
    return after.alpha - old_alpha


@dataclass(frozen=True)
class GreedyStep:
    number: int
    key: ConnectionKey
    before: ConnectionSetting
    after: ConnectionSetting
    drift_before_ratio: float
    drift_after_ratio: float
    drift_improvement_pp: float
    relative_improvement_percent: float | None
    roof_after_m: float
    candidate_solves_this_iteration: int


@dataclass
class GreedyResult:
    baseline_result: ConnectedFrameResult
    final_result: ConnectedFrameResult
    baseline_settings: dict[ConnectionKey, ConnectionSetting]
    recommended_settings: dict[ConnectionKey, ConnectionSetting]
    steps: list[GreedyStep]
    target_ratio: float
    modification_budget: int
    status: str
    eligible_keys: list[ConnectionKey]
    attempted_solves: int
    invalid_solves: int
    allow_semi: bool
    upgrade_alpha: float
    allow_rigid: bool
    candidate_alphas: tuple[float, ...] | None

    @property
    def target_met(self) -> bool:
        return self.final_result.max_story_drift_ratio <= self.target_ratio + 1e-12

    @property
    def baseline_drift(self) -> float:
        return self.baseline_result.max_story_drift_ratio

    @property
    def final_drift(self) -> float:
        return self.final_result.max_story_drift_ratio


def optimize_connections(
    model: BuildingModel,
    baseline_settings: Mapping[ConnectionKey, ConnectionSetting] | None = None,
    target_drift_ratio: float = 0.0015,
    modification_budget: int = 4,
    *,
    allow_semi: bool = False,
    upgrade_alpha: float = 15.0,
    allow_rigid: bool = True,
    candidate_alphas: Iterable[float] | None = None,
    eligible_keys: Iterable[ConnectionKey] | None = None,
    baseline_result: ConnectedFrameResult | None = None,
) -> GreedyResult:
    """Recommend beneficial upgrades, greedily choosing one connection at a time.

    Starting at the last analyzed/manual baseline, the algorithm tests allowable
    stiffness-increasing alternatives at every eligible *unused* beam end. The
    largest positive decrease in maximum story drift is accepted, after which
    all alternatives are re-solved for the new structural configuration.

    Stops when the target is achieved, the number of changed ends reaches the
    budget, or there is no strictly beneficial candidate. A beam end can be
    selected once, so one budget unit always means one distinct beam end.

    `target_drift_ratio` is dimensionless: 0.0015 means 0.15 percent.
    """
    if not np.isfinite(target_drift_ratio) or target_drift_ratio < 0:
        raise ValueError('Target drift ratio must be finite and nonnegative.')
    if isinstance(modification_budget, bool) or not isinstance(modification_budget, (int, np.integer)) or modification_budget < 0:
        raise ValueError('Modification budget must be a nonnegative integer.')
    if not np.isfinite(upgrade_alpha) or upgrade_alpha <= 0:
        raise ValueError('Upgrade alpha must be positive and finite.')
    if candidate_alphas is not None:
        levels = tuple(sorted(set(float(a) for a in candidate_alphas)))
        if any(not np.isfinite(a) or a <= 0 for a in levels):
            raise ValueError('Every candidate alpha must be positive and finite.')
        candidate_alphas = levels

    available = all_beam_ends(model)
    allowed_set = set(available)
    if eligible_keys is None:
        eligible = available
    else:
        eligible = list(dict.fromkeys(eligible_keys))
        if any(k not in allowed_set for k in eligible):
            raise ValueError('Requested eligible beam end does not exist in this frame.')

    start = _canonical(baseline_settings)
    if any(k not in allowed_set for k in start):
        raise ValueError('Baseline includes a beam end that does not exist.')
    if any(not isinstance(v, ConnectionSetting) for v in start.values()):
        raise TypeError('Baseline connection values must be ConnectionSetting instances.')

    if baseline_result is None:
        baseline_result = analyze_connected_building(model, start)
    current_result = baseline_result
    current = start.copy()
    steps: list[GreedyStep] = []
    used_ends: set[ConnectionKey] = set()
    attempted = 0
    invalid = 0
    status = 'no_eligible_upgrades'

    if current_result.max_story_drift_ratio <= target_drift_ratio + 1e-12:
        status = 'already_satisfied'
    elif modification_budget == 0:
        status = 'budget_exhausted'
    else:
        while len(steps) < modification_budget:
            D0 = float(current_result.max_story_drift_ratio)
            tolerance = max(1e-12, 1e-9 * D0)
            best = None  # (gain, key, option, response)
            best_meeting = None  # (restraint_increment, gain, key, option, response)
            tries_this_iteration = 0
            options_available = 0

            for key in eligible:
                if key in used_ends:
                    continue
                original = current.get(key, connection('rigid'))
                for option in _upgrade_options(
                    original, allow_semi, upgrade_alpha,
                    allow_rigid=allow_rigid, candidate_alphas=candidate_alphas
                ):
                    options_available += 1
                    candidate = current.copy()
                    if option.kind == 'rigid':
                        candidate.pop(key, None)
                    else:
                        candidate[key] = option
                    attempted += 1
                    tries_this_iteration += 1
                    try:
                        response = analyze_connected_building(model, candidate)
                        Di = float(response.max_story_drift_ratio)
                        if not np.isfinite(Di) or Di < 0:
                            raise ValueError('Candidate returned invalid drift.')
                    except (ValueError, np.linalg.LinAlgError):
                        invalid += 1
                        continue
                    gain = D0 - Di
                    # Strict positive improvement, including numerical noise guard.
                    # Ties preserve deterministic floor/bay/end ordering and favor
                    # semi over rigid because semi alternatives are evaluated first.
                    if gain > tolerance and (best is None or gain > best[0] + tolerance):
                        best = (gain, key, option, response)
                    # If a trial already reaches the user target, prefer the
                    # smallest additional normalized rotational restraint.
                    # Rigid is considered only if no finite spring meets it.
                    if gain > tolerance and Di <= target_drift_ratio + 1e-12:
                        restraint = _extra_restraint(original, option)
                        if (best_meeting is None or restraint < best_meeting[0] - 1e-10
                            or ((restraint == best_meeting[0] or abs(restraint - best_meeting[0]) <= 1e-10)
                                and gain > best_meeting[1] + tolerance)):
                            best_meeting = (restraint, gain, key, option, response)

            if best is None:
                status = ('no_eligible_upgrades' if options_available == 0
                          else 'no_positive_improvement')
                break

            if best_meeting is not None:
                _effort, gain, key, option, response = best_meeting
            else:
                gain, key, option, response = best
            before = current.get(key, connection('rigid'))
            if option.kind == 'rigid':
                current.pop(key, None)
            else:
                current[key] = option
            used_ends.add(key)
            steps.append(GreedyStep(
                number=len(steps) + 1,
                key=key,
                before=before,
                after=option,
                drift_before_ratio=D0,
                drift_after_ratio=float(response.max_story_drift_ratio),
                drift_improvement_pp=100.0 * gain,
                relative_improvement_percent=(100.0 * gain / D0) if D0 > 1e-12 else None,
                roof_after_m=float(response.roof_max_displacement),
                candidate_solves_this_iteration=tries_this_iteration,
            ))
            current_result = response
            if current_result.max_story_drift_ratio <= target_drift_ratio + 1e-12:
                status = 'target_satisfied'
                break
        else:
            status = 'budget_exhausted'

    return GreedyResult(
        baseline_result=baseline_result,
        final_result=current_result,
        baseline_settings=start,
        recommended_settings=current,
        steps=steps,
        target_ratio=float(target_drift_ratio),
        modification_budget=int(modification_budget),
        status=status,
        eligible_keys=eligible,
        attempted_solves=attempted,
        invalid_solves=invalid,
        allow_semi=allow_semi,
        upgrade_alpha=float(upgrade_alpha),
        allow_rigid=allow_rigid,
        candidate_alphas=candidate_alphas,
    )


def optimization_table_rows(result: GreedyResult) -> list[dict]:
    """One step per modified beam end; suitable for CSV or Streamlit table."""
    result_rows = []
    for s in result.steps:
        def label(setting):
            if setting.kind == 'semi':
                return f'semi (α={setting.alpha:g})'
            return setting.kind
        result_rows.append({
            'Step': s.number,
            'Floor': s.key[0],
            'Bay': s.key[1],
            'Beam end': s.key[2],
            'Before': label(s.before),
            'Upgrade to': label(s.after),
            'Before maximum drift (%)': round(100*s.drift_before_ratio, 7),
            'After maximum drift (%)': round(100*s.drift_after_ratio, 7),
            'Drift improvement (pp)': round(s.drift_improvement_pp, 7),
            'Relative improvement (%)': round(s.relative_improvement_percent, 6)
                if s.relative_improvement_percent is not None else '',
            'Roof |Ux| after (mm)': round(1000*s.roof_after_m, 5),
            'Candidates solved this step': s.candidate_solves_this_iteration,
        })
    return result_rows
