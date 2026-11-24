"""Plotting and export helpers for the CE7405 Streamlit interface.

Independent of Streamlit so these helpers can be regression-tested.
"""
from __future__ import annotations

import csv
import io

import matplotlib.pyplot as plt
import numpy as np

from semirigid_connections import ConnectionSetting

PURPLE = "#461D70"
GOLD = "#D8A11E"
BLUE = "#2870B3"
RED = "#B93B48"
GRAY = "#B0B6C0"


def frame_figure(model, result, settings: dict, show_undeformed=True):
    """Original and displaced member centerlines, with nonrigid beam-end markers."""
    coords = np.asarray(model.nodes, float)
    U = np.asarray(result.displacements, float).reshape(-1, 3)
    span = max(float(np.ptp(coords[:, 0])), float(np.ptp(coords[:, 1])), 1.0)
    movement = np.max(np.abs(U[:, :2]))
    deformation_scale = (0.10 * span / movement) if movement > 1e-12 else 1.0
    new_coords = coords + deformation_scale * U[:, :2]

    fig, ax = plt.subplots(figsize=(7.2, 5.2), layout="constrained")
    for i, j, *_ in model.elements:
        if show_undeformed:
            ax.plot(coords[[i,j],0], coords[[i,j],1], color=GRAY,
                    lw=1.4, alpha=.70, linestyle="--")
        ax.plot(new_coords[[i,j],0], new_coords[[i,j],1],
                color=PURPLE, lw=2.6, solid_capstyle="round")

    for (floor, bay, end), setting in settings.items():
        if setting.kind == "rigid":
            continue
        col = (bay - 1) if end == "left" else bay
        node = model.node_id(floor, col)
        x, y = new_coords[node]
        ax.scatter(x, y, s=68, facecolor=GOLD if setting.kind == "semi" else RED,
                   edgecolor="white", linewidth=1.1, zorder=5)

    for col in range(model.n_bays + 1):
        x, y = coords[model.node_id(0, col)]
        ax.plot(x, y - .05*span, marker="^", markersize=7, color="#4A5360")

    ax.plot([], [], color=PURPLE, lw=2.6, label="Deformed (exaggerated)")
    if show_undeformed:
        ax.plot([], [], color=GRAY, ls="--", lw=1.4, label="Original")
    if any(v.kind == "semi" for v in settings.values()):
        ax.scatter([], [], color=GOLD, label="Semi-rigid", s=60)
    if any(v.kind == "pinned" for v in settings.values()):
        ax.scatter([], [], color=RED, label="Pinned", s=60)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Horizontal coordinate, m")
    ax.set_ylabel("Elevation, m")
    ax.set_title(f"2D frame response   |   display scale ×{deformation_scale:,.1f}")
    ax.grid(alpha=.13)
    ax.legend(loc="best", fontsize=8)
    return fig


def drift_figure(result, base_result, target_percent=None):
    """Max interstory drift measured along the individual column lines."""
    pct = 100.0 * np.asarray(result.story_drift_ratios)
    baseline = 100.0 * np.asarray(base_result.story_drift_ratios)
    locations = np.arange(len(pct))
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    ax.bar(locations-.19, baseline, width=.38, color=GRAY, label="All rigid")
    ax.bar(locations+.19, pct, width=.38, color=PURPLE, label="Selected model")
    if target_percent is not None and target_percent > 0:
        ax.axhline(target_percent, ls="--", color=RED, lw=1.2,
                   label="User screening limit")
    ax.set_xticks(locations, [f"Story {i+1}" for i in locations])
    ax.set_ylabel("Maximum interstory drift ratio, %")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=.15)
    ax.legend(fontsize=8)
    return fig


def moment_figure(model, result, member_index):
    """Local internal bending moment: M(x)=-M1+V1*x+q*x²/2.

    Local member end forces must include the fixed-end force corrections.
    """
    i, j, *_ = model.elements[member_index]
    xy1 = np.asarray(model.nodes[i], float)
    xy2 = np.asarray(model.nodes[j], float)
    L = float(np.linalg.norm(xy2-xy1))
    f = np.asarray(result.member_end_forces[member_index], float)
    q = float(model.local_y_uniform_loads.get(member_index, 0.0))
    x = np.linspace(0.0, L, 120)
    M = (-f[2] + f[1]*x + .5*q*x*x) / 1000.0
    fig, ax = plt.subplots(figsize=(7, 3), layout="constrained")
    ax.plot(x, M, color=PURPLE, lw=2.2)
    ax.fill_between(x, 0, M, color=GOLD, alpha=.22)
    ax.axhline(0, color="#4B5563", lw=1)
    ax.set_xlabel("Distance from element start, m")
    ax.set_ylabel("Internal bending moment, kN·m")
    ax.set_title(f"Member {member_index+1}: {model.element_types[member_index].title()} bending moment")
    ax.grid(alpha=.16)
    return fig


def member_force_rows(model, result):
    rows = []
    for idx, ((n1, n2, E, A, I), forces) in enumerate(zip(model.elements, result.member_end_forces), start=1):
        n1_force, v1, m1, n2_force, v2, m2 = np.asarray(forces, float)
        rows.append({
            "Member": idx, "Type": model.element_types[idx-1],
            "Start node": n1, "End node": n2,
            "N1 (kN)": n1_force / 1000,
            "V1 (kN)": v1 / 1000,
            "M1 (kN·m)": m1 / 1000,
            "N2 (kN)": n2_force / 1000,
            "V2 (kN)": v2 / 1000,
            "M2 (kN·m)": m2 / 1000,
        })
    return rows


def reaction_rows(model, result):
    rows = []
    for col in range(model.n_bays + 1):
        node = model.node_id(0, col)
        fx, fy, mz = np.asarray(result.reactions[3*node:3*node+3], float)
        rows.append({"Base node": node, "Rx (kN)": fx/1000,
                     "Ry (kN)": fy/1000, "Rz (kN·m)": mz/1000})
    return rows


def csv_text(rows):
    if not rows:
        return ""
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()
