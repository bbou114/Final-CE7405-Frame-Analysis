"""Color-coded sensitivity map, with a marker for EACH beam end."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
import numpy as np

from sensitivity_mapping import SensitivityStudy


SENSITIVITY_CMAP = LinearSegmentedColormap.from_list(
    'connection_change_priority', ['#249553', '#FFE08A', '#C92B3A'], N=256
)


def sensitivity_figure(model, study: SensitivityStudy):
    """Plot original structural grid + signed one-change effect at beam ends.

    GREEN: trial increases drift (not helpful).
    YELLOW: trial has little effect.
    RED: trial reduces drift (priority candidate for this trial setting).
    GREY: unchanged or invalid trial.

    Ranking is based on the actual one-at-a-time drift reduction, not a
    judgment about the current connection in isolation. Each beam-end marker
    includes a signed numeric score so colors are not the only cue.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5), layout='constrained')
    xy = np.array(model.nodes, dtype=float)
    span_x = max(float(np.ptp(xy[:, 0])), 1.0)
    span_y = max(float(np.ptp(xy[:, 1])), 1.0)

    for i, j, *_ in model.elements:
        ax.plot(xy[[i, j], 0], xy[[i, j], 1], color='#8A94A6',
                linewidth=1.6, alpha=.85, zorder=1)
    for i in range(model.n_bays+1):
        ax.plot(xy[model.node_id(0, i), 0], 0, marker='^',
                color='#545B68', markersize=7)

    scored = [r for r in study.records if r.status == 'ok']
    relative = study.relative_score_available
    values = [(r.improvement_percent if relative else r.drift_reduction_pp)
              for r in scored]
    extent = max([abs(float(v)) for v in values] + [1e-6])
    norm = Normalize(vmin=-extent, vmax=extent)
    # Map signed improvement: negative = GREEN, zero = YELLOW, positive = RED.
    # Keep zero at the midpoint even when all evaluated changes have one sign.
    cmap = SENSITIVITY_CMAP

    for record in study.records:
        floor, bay, end = record.key
        left = xy[model.node_id(floor, bay-1)]
        right = xy[model.node_id(floor, bay)]
        location = left + (0.15 if end == 'left' else 0.85) * (right - left)
        if record.status == 'ok':
            v = record.improvement_percent if relative else record.drift_reduction_pp
            face = cmap(norm(v))
        else:
            face = '#C4C9D0'
        ax.scatter(*location, s=135, color=face, edgecolors='#33383F',
                   linewidth=.55, zorder=4)
        ax.annotate(f'{float(v):+.1f}' if record.status == 'ok' else '—',
                    (location[0], location[1]), xytext=(0, 10),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=6.7, color='#30343B', zorder=5)

    mapper = ScalarMappable(cmap=cmap, norm=norm)
    mapper.set_array([])
    cb = fig.colorbar(mapper, ax=ax, shrink=.75, pad=.02)
    cb.set_label('Relative drift reduction (%)' if relative
                 else 'Drift reduction (percentage points)')
    ax.set_xlim(-.05*span_x, 1.05*span_x)
    ax.set_ylim(-.07*span_y, 1.10*span_y)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Horizontal coordinate (m)')
    ax.set_ylabel('Elevation (m)')
    ax.set_title('Connection sensitivity: recommended change priority')
    ax.grid(alpha=.12)
    ax.text(.01, .01, 'GREEN: more drift   •   YELLOW: minimal effect   •   RED: less drift   •   GREY: skipped/invalid',
            transform=ax.transAxes, fontsize=8, ha='left', va='bottom')
    return fig
