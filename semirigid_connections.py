"""CE 7405 | Step 4: linearly elastic rotational beam-column connections.

Model: columns connect rigidly to the physical joint rotations; an optional
independent beam-end rotation is linked to that joint by a zero-length
rotational spring. Translational displacements remain shared at the joint.

Connection definitions are indexed by (floor, bay, end), where floor/bay are
1-based integers and end is 'left' or 'right'. For example, (2, 2, 'left')
identifies the left end of the beam at floor 2, bay 2.

This implementation follows McGuire, Gallagher & Ziemian, Matrix Structural
Analysis, 2nd edition, section 13.5.1, Eq. 13.30 and Example 13.7.

All internal calculations use N, m, Pa, radians.
"""

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from main import element_matrices
from multistory_frame import (
    BuildingModel,
    equivalent_uniform_load_local,
)

ConnectionKey = tuple[int, int, str]


@dataclass(frozen=True)
class ConnectionSetting:
    """kind: rigid, pinned, or semi; semi alpha = k_theta*L/(E*I)."""
    kind: str
    alpha: float = 0.0

    def __post_init__(self):
        if self.kind not in ('rigid', 'pinned', 'semi'):
            raise ValueError("Connection kind must be rigid, pinned, or semi.")
        if self.kind == 'semi' and (not np.isfinite(self.alpha) or self.alpha <= 0):
            raise ValueError("Semi-rigid connections require a positive, finite alpha.")


def connection(kind: str, alpha: float = 0.0) -> ConnectionSetting:
    """Create a connection: connection('semi', alpha=10), etc."""
    return ConnectionSetting(kind=kind, alpha=alpha)


@dataclass
class SpringResponse:
    key: ConnectionKey
    kind: str
    alpha: float
    k_theta_Nm_per_rad: float
    beam_end_rotation_rad: float
    column_joint_rotation_rad: float
    relative_rotation_rad: float
    beam_end_moment_Nm: float
    transmitted_moment_Nm: float
    beam_end_balance_residual_Nm: float


@dataclass
class ConnectedFrameResult:
    # First fields intentionally mirror Step 3, so earlier reporting and
    # plotting logic can use these arrays without knowing about extra DOFs.
    displacements: np.ndarray  # physical joint DOFs only, length = 3 * n_nodes
    reactions: np.ndarray      # reactions at physical joint DOFs only
    member_end_forces: list[np.ndarray]   # local N,V,M,N,V,M
    stiffness: np.ndarray      # FULL assembled K, including extra beam rotations
    effective_nodal_forces: np.ndarray  # external load resultant on physical nodes
    story_drift_ratios: np.ndarray
    mean_floor_displacements: np.ndarray
    roof_max_displacement: float
    max_story_drift_ratio: float
    worst_story: int
    full_displacements: np.ndarray
    full_residual: np.ndarray
    spring_responses: dict[ConnectionKey, SpringResponse]
    number_of_extra_rotations: int


def _settings_for_model(
    model: BuildingModel,
    settings: Mapping[ConnectionKey, ConnectionSetting] | None,
) -> dict[ConnectionKey, ConnectionSetting]:
    settings = {} if settings is None else dict(settings)
    for key, setting in settings.items():
        if not (isinstance(key, tuple) and len(key) == 3):
            raise ValueError(f"Bad connection key: {key!r}; use (floor, bay, end).")
        floor, bay, end = key
        if not (isinstance(floor, int) and 1 <= floor <= model.n_stories
                and isinstance(bay, int) and 1 <= bay <= model.n_bays
                and end in ('left', 'right')):
            raise ValueError(f"Connection key {key!r} does not exist in this frame.")
        if not isinstance(setting, ConnectionSetting):
            raise TypeError(f"Use connection('pinned'), connection('rigid'), "
                            f"or connection('semi', alpha=...). Bad value: {setting!r}")
    return settings


def all_beam_ends(model: BuildingModel) -> list[ConnectionKey]:
    return [(floor, bay, end)
            for floor in range(1, model.n_stories + 1)
            for bay in range(1, model.n_bays + 1)
            for end in ('left', 'right')]


def analyze_connected_building(
    model: BuildingModel,
    settings: Mapping[ConnectionKey, ConnectionSetting] | None = None,
) -> ConnectedFrameResult:
    """Analyze building with optional independent rotations at beam ends.

    - rigid: beam and column use same rotation DOF (exact equality).
    - pinned: separate beam rotation DOF, zero spring stiffness.
    - semi: separate beam rotation DOF + finite rotational spring.

    Member uniform loads are converted to consistent local nodal loads and
    end forces are recovered as k_local*d_local - f_equivalent_local.
    """
    selected = _settings_for_model(model, settings)
    n_physical = 3 * len(model.nodes)
    n_columns = model.n_stories * (model.n_bays + 1)
    if len(model.elements) != n_columns + model.n_stories * model.n_bays:
        raise ValueError("Expected the element numbering from generate_building().")

    # Element-global DOF mapping. Each nonrigid beam end gets an extra DOF.
    element_maps: list[list[int]] = []
    springs = []
    extra_index = n_physical
    for e, (i, j, E, A, I) in enumerate(model.elements):
        dofs = [3*i, 3*i+1, 3*i+2, 3*j, 3*j+1, 3*j+2]
        if e >= n_columns:
            beam_n = e - n_columns
            floor = beam_n // model.n_bays + 1
            bay = beam_n % model.n_bays + 1
            x_i, y_i = model.nodes[i]
            x_j, y_j = model.nodes[j]
            L = float(np.hypot(x_j-x_i, y_j-y_i))
            for end_name, p, node in (('left', 2, i), ('right', 5, j)):
                key = (floor, bay, end_name)
                setting = selected.get(key, ConnectionSetting('rigid'))
                if setting.kind == 'rigid':
                    continue
                joint_rotation_dof = 3*node + 2
                end_rotation_dof = extra_index
                extra_index += 1
                dofs[p] = end_rotation_dof
                k = (setting.alpha*E*I/L) if setting.kind == 'semi' else 0.0
                springs.append((key, setting, k, end_rotation_dof,
                                joint_rotation_dof, e, p))
        element_maps.append(dofs)

    total_dofs = extra_index
    K = np.zeros((total_dofs, total_dofs))
    F = np.zeros(total_dofs)
    # The physical-node load resultant is stored separately. The full RHS puts
    # distributed-load end moments onto their actual (possibly extra) DOFs.
    physical_equivalent = np.zeros(n_physical)
    member_data = []

    for node, (fx, fy, mz) in model.nodal_loads.items():
        F[3*node:3*node+3] += (fx, fy, mz)
        physical_equivalent[3*node:3*node+3] += (fx, fy, mz)

    for e, (i, j, E, A, I) in enumerate(model.elements):
        kl, T, kg, L = element_matrices(*model.nodes[i], *model.nodes[j], E, A, I)
        indices = element_maps[e]
        K[np.ix_(indices, indices)] += kg
        f_local = np.zeros(6)
        if e in model.local_y_uniform_loads:
            f_local = equivalent_uniform_load_local(model.local_y_uniform_loads[e], L)
            f_global = T.T @ f_local
            F[indices] += f_global
            geom_indices = [3*i, 3*i+1, 3*i+2, 3*j, 3*j+1, 3*j+2]
            physical_equivalent[geom_indices] += f_global
        member_data.append((indices, kl, T, f_local))

    for key, setting, k, beam_dof, joint_dof, e, p in springs:
        if k > 0:
            K[beam_dof, beam_dof] += k
            K[joint_dof, joint_dof] += k
            K[beam_dof, joint_dof] -= k
            K[joint_dof, beam_dof] -= k

    restrained = sorted(set(model.fixed_dofs))
    free = [d for d in range(total_dofs) if d not in restrained]
    Kff = K[np.ix_(free, free)]
    if not free:
        raise ValueError("No free degrees of freedom.")
    # The stiffness may be badly scaled due to translation/rotation units;
    # solve and check numerically instead of using an arbitrary rank tolerance.
    try:
        Ufree = np.linalg.solve(Kff, F[free])
    except np.linalg.LinAlgError as exc:
        raise ValueError("Unstable frame or incompatible releases: singular stiffness matrix.") from exc
    U = np.zeros(total_dofs)
    U[free] = Ufree
    if not np.all(np.isfinite(U)):
        raise ValueError("Unstable frame: nonfinite displacement solution.")
    residual = K @ U - F
    # Explicit equilibrium check for free DOFs. A singular but nearly solvable
    # system is additionally caught with the residual and displacement scale.
    tol = 2e-7 * max(1.0, float(np.max(np.abs(F))))
    if np.max(np.abs(residual[free])) > tol:
        raise ValueError("Structural solve failed free-DOF equilibrium check.")

    member_forces = []
    for indices, kl, T, f_local in member_data:
        dlocal = T @ U[indices]
        member_forces.append(kl @ dlocal - f_local)

    response = {}
    for key, setting, k, beam_dof, joint_dof, e, moment_index in springs:
        theta_beam = U[beam_dof]
        theta_joint = U[joint_dof]
        moment = member_forces[e][moment_index]
        # Moment on the beam from its connection is k*(theta_joint-theta_beam).
        transmitted = k*(theta_joint - theta_beam)
        response[key] = SpringResponse(
            key=key, kind=setting.kind, alpha=setting.alpha,
            k_theta_Nm_per_rad=k,
            beam_end_rotation_rad=theta_beam,
            column_joint_rotation_rad=theta_joint,
            relative_rotation_rad=theta_beam - theta_joint,
            beam_end_moment_Nm=moment,
            transmitted_moment_Nm=transmitted,
            beam_end_balance_residual_Nm=moment-transmitted,
        )

    physical_U = U[:n_physical]
    floor_ux = np.array([
        np.mean([physical_U[3*model.node_id(f, c)]
                 for c in range(model.n_bays+1)])
        for f in range(model.n_stories+1)
    ])
    drifts = np.array([
        max(abs(physical_U[3*model.node_id(f, c)] -
                physical_U[3*model.node_id(f-1, c)])
            for c in range(model.n_bays+1)) / model.story_heights[f-1]
        for f in range(1, model.n_stories+1)
    ])
    roof = max(abs(physical_U[3*model.node_id(model.n_stories, c)])
               for c in range(model.n_bays+1))

    return ConnectedFrameResult(
        displacements=physical_U,
        reactions=residual[:n_physical],
        member_end_forces=member_forces,
        stiffness=K,
        effective_nodal_forces=physical_equivalent,
        story_drift_ratios=drifts,
        mean_floor_displacements=floor_ux,
        roof_max_displacement=float(roof),
        max_story_drift_ratio=float(np.max(drifts)),
        worst_story=int(np.argmax(drifts)+1),
        full_displacements=U,
        full_residual=residual,
        spring_responses=response,
        number_of_extra_rotations=total_dofs - n_physical,
    )
