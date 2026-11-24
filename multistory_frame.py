"""CE 7405 | Step 3: generate and analyze regular multi-story 2D rigid frames.

Reuses the verified direct stiffness solver from main.py.  All joints are
fully rigid in this step; connection springs come in the next project stage.

Units: N, m, Pa and radians.  Floor forces and beam UDL input use kN and
kN/m for convenient building-scale data entry; they are converted to SI.

Sign convention: global X to right, global Y upward, positive Mz CCW.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from main import analyze_frame, element_matrices


@dataclass
class BuildingModel:
    nodes: list[tuple[float, float]]
    elements: list[tuple[int, int, float, float, float]]
    element_types: list[str]
    nodal_loads: dict[int, tuple[float, float, float]]
    fixed_dofs: list[int]
    # local_y_uniform_loads[e] in N/m, positive in the local +y direction.
    local_y_uniform_loads: dict[int, float]
    n_stories: int
    n_bays: int
    story_heights: tuple[float, ...]
    bay_widths: tuple[float, ...]
    story_lateral_loads_kN: tuple[float, ...]
    beam_udl_kN_per_m: float
    base_support: str

    def node_id(self, level: int, column: int) -> int:
        """Nodes are numbered row-by-row: ground floor first, then upward."""
        return level * (self.n_bays + 1) + column


@dataclass
class BuildingResult:
    displacements: np.ndarray  # global [ux, uy, theta] for every node
    reactions: np.ndarray      # same global DOF order; nonzero at supports
    member_end_forces: list[np.ndarray]  # LOCAL [N1,V1,M1,N2,V2,M2]
    stiffness: np.ndarray
    effective_nodal_forces: np.ndarray  # nodal loads + distributed equivalents
    story_drift_ratios: np.ndarray      # max |ux(top)-ux(bottom)| / height over columns
    mean_floor_displacements: np.ndarray # mean ux per floor, ground included
    roof_max_displacement: float
    max_story_drift_ratio: float
    worst_story: int            # 1-based story index


def generate_building(
    n_stories: int = 4,
    n_bays: int = 3,
    bay_widths: Sequence[float] | None = None,
    story_heights: Sequence[float] | None = None,
    E: float = 200e9,
    column_A: float = 0.020,
    column_I: float = 2.0e-4,
    beam_A: float = 0.015,
    beam_I: float = 1.2e-4,
    story_lateral_loads_kN: Sequence[float] | None = None,
    beam_udl_kN_per_m: float = 8.0,
    base_support: str = "fixed",
) -> BuildingModel:
    """Generate an orthogonal 2D building frame from a few editable inputs.

    n_stories x n_bays defaults to 4 x 3 (20 nodes and 28 members).
    story_lateral_loads_kN contains TOTAL horizontal force for each story;
    it is divided equally among the (n_bays+1) joints on that floor.
    beam_udl_kN_per_m is the uniform *downward* load on each floor beam.
    Only fixed and pinned bases are supported here; no beam-end springs yet.
    """
    if not isinstance(n_stories, int) or not isinstance(n_bays, int):
        raise ValueError("n_stories and n_bays must be integers.")
    if not 1 <= n_stories <= 10 or not 1 <= n_bays <= 8:
        raise ValueError("Supported range: 1-10 stories and 1-8 bays.")
    if base_support not in ("fixed", "pinned"):
        raise ValueError("base_support must be 'fixed' or 'pinned'.")
    if min(E, column_A, column_I, beam_A, beam_I) <= 0:
        raise ValueError("Elastic modulus, areas, and inertias must be positive.")
    if beam_udl_kN_per_m < 0:
        raise ValueError("beam_udl_kN_per_m is a nonnegative downward load.")

    if bay_widths is None:
        bay_widths = [6.0] * n_bays
    if story_heights is None:
        story_heights = [3.5] * n_stories
    if story_lateral_loads_kN is None:
        # Gradually increasing demonstration story loads, bottom to top.
        story_lateral_loads_kN = [10.0 + 5.0 * j for j in range(n_stories)]
    if len(bay_widths) != n_bays:
        raise ValueError("Supply one bay width for every bay.")
    if len(story_heights) != n_stories:
        raise ValueError("Supply one story height for every story.")
    if len(story_lateral_loads_kN) != n_stories:
        raise ValueError("Supply one TOTAL lateral force for every story.")
    if not all(np.isfinite(x) and x > 0 for x in bay_widths):
        raise ValueError("All bay widths must be finite and positive.")
    if not all(np.isfinite(x) and x > 0 for x in story_heights):
        raise ValueError("All story heights must be finite and positive.")
    if not all(np.isfinite(x) for x in story_lateral_loads_kN):
        raise ValueError("Story lateral forces must be finite.")

    x_positions = np.r_[0.0, np.cumsum(bay_widths)]
    y_positions = np.r_[0.0, np.cumsum(story_heights)]
    nodes = [
        (float(x), float(y))
        for y in y_positions
        for x in x_positions
    ]

    def nid(level: int, col: int) -> int:
        return level * (n_bays + 1) + col

    elements = []
    element_types = []
    beam_uniform_loads = {}

    # All columns are oriented bottom to top.
    for story in range(n_stories):
        for col in range(n_bays + 1):
            elements.append((nid(story, col), nid(story + 1, col),
                             E, column_A, column_I))
            element_types.append("column")

    # All beams are oriented left to right: local +y is global upward.
    for floor in range(1, n_stories + 1):
        for bay in range(n_bays):
            e = len(elements)
            elements.append((nid(floor, bay), nid(floor, bay + 1),
                             E, beam_A, beam_I))
            element_types.append("beam")
            if beam_udl_kN_per_m:
                beam_uniform_loads[e] = -beam_udl_kN_per_m * 1000.0

    nodal_loads = {}
    for floor, fx_kN in enumerate(story_lateral_loads_kN, start=1):
        force_at_each_joint = 1000.0 * fx_kN / (n_bays + 1)
        for col in range(n_bays + 1):
            nodal_loads[nid(floor, col)] = (force_at_each_joint, 0.0, 0.0)

    fixed_dofs = []
    for col in range(n_bays + 1):
        base = nid(0, col)
        fixed_dofs.extend([3 * base, 3 * base + 1])
        if base_support == "fixed":
            fixed_dofs.append(3 * base + 2)

    return BuildingModel(
        nodes=nodes, elements=elements, element_types=element_types,
        nodal_loads=nodal_loads, fixed_dofs=fixed_dofs,
        local_y_uniform_loads=beam_uniform_loads,
        n_stories=n_stories, n_bays=n_bays,
        story_heights=tuple(float(v) for v in story_heights),
        bay_widths=tuple(float(v) for v in bay_widths),
        story_lateral_loads_kN=tuple(float(v) for v in story_lateral_loads_kN),
        beam_udl_kN_per_m=float(beam_udl_kN_per_m),
        base_support=base_support,
    )


def equivalent_uniform_load_local(q_local_y: float, L: float) -> np.ndarray:
    """Euler-Bernoulli CONSISTENT local nodal forces from uniform local-y load.

    DOF order: [u1, v1, theta1, u2, v2, theta2].
    A negative local-y load on a left-to-right horizontal beam is downward.
    Fixed-end member forces are the negative of this vector at zero displacement.
    Based on textbook Ch. 5, Sec. 5.2 (equivalent nodal loads).
    """
    return np.array([
        0.0, q_local_y * L / 2.0, q_local_y * L**2 / 12.0,
        0.0, q_local_y * L / 2.0, -q_local_y * L**2 / 12.0,
    ])


def analyze_building(model: BuildingModel) -> BuildingResult:
    """Call original solver with consistent beam loads; recover TOTAL end forces."""
    effective_forces = np.zeros(3 * len(model.nodes))
    for node, (Fx, Fy, Mz) in model.nodal_loads.items():
        effective_forces[3*node:3*node+3] += [Fx, Fy, Mz]

    beam_f_local = {}
    for e, q in model.local_y_uniform_loads.items():
        i, j, E, A, I = model.elements[e]
        k_local, T, k_global, L = element_matrices(
            *model.nodes[i], *model.nodes[j], E, A, I
        )
        f_local = equivalent_uniform_load_local(q, L)
        beam_f_local[e] = f_local
        indices = [3*i, 3*i+1, 3*i+2, 3*j, 3*j+1, 3*j+2]
        effective_forces[indices] += T.T @ f_local

    # The existing analyze_frame() accepts nodal loads, so pass it the sum of
    # direct nodal loads and all transformed beam equivalent nodal loads.
    effective_dict = {
        i: tuple(effective_forces[3*i:3*i+3])
        for i in range(len(model.nodes))
    }
    U, reactions, member_end_forces, K = analyze_frame(
        model.nodes, model.elements, effective_dict, model.fixed_dofs
    )

    # Original solver returns k*d, which excludes member fixed-end forces.
    # Correct *each loaded member* with -f_equivalent_local.
    for e, f_local in beam_f_local.items():
        member_end_forces[e] = member_end_forces[e] - f_local

    mean_ux_by_floor = np.array([
        np.mean([U[3*model.node_id(floor, col)]
                 for col in range(model.n_bays + 1)])
        for floor in range(model.n_stories + 1)
    ])
    ratios = []
    for story in range(1, model.n_stories + 1):
        h = model.story_heights[story - 1]
        drifts = [
            abs(U[3*model.node_id(story, col)] -
                U[3*model.node_id(story - 1, col)]) / h
            for col in range(model.n_bays + 1)
        ]
        ratios.append(max(drifts))
    ratios = np.array(ratios)
    roof_max = max(abs(U[3*model.node_id(model.n_stories, col)])
                   for col in range(model.n_bays + 1))

    return BuildingResult(
        displacements=U,
        reactions=reactions,
        member_end_forces=member_end_forces,
        stiffness=K,
        effective_nodal_forces=effective_forces,
        story_drift_ratios=ratios,
        mean_floor_displacements=mean_ux_by_floor,
        roof_max_displacement=roof_max,
        max_story_drift_ratio=float(max(ratios)),
        worst_story=int(np.argmax(ratios) + 1),
    )


def equilibrium_residuals(model: BuildingModel, result: BuildingResult):
    """Return global external+reaction residuals [Fx,Fy,Mz] about origin.

    Consistent nodal equivalents reproduce the load resultant and first
    moment of uniform member loads exactly.
    """
    residual = np.zeros(3)
    net = result.reactions + result.effective_nodal_forces
    for node, (x, y) in enumerate(model.nodes):
        Fx, Fy, Mz = net[3*node:3*node+3]
        residual += [Fx, Fy, x*Fy - y*Fx + Mz]
    return residual
