"""CE 7405 - Step 1: Direct stiffness method for a 2D cantilever frame element.

This single-file starter is suitable for running in Spyder or a terminal.
Units: N, m, Pa; rotational DOF in radians.
Textbook: McGuire, Gallagher & Ziemian, Matrix Structural Analysis,
2nd ed., Ch. 3 (assembly), Ch. 4 (frame stiffness), Ch. 5 (transformation).
"""

import numpy as np


def local_stiffness(E, A, I, L):
    """Local 6x6 Euler-Bernoulli frame element stiffness matrix."""
    a = E * A / L
    b = 12 * E * I / L**3
    c = 6 * E * I / L**2
    d = 4 * E * I / L
    e = 2 * E * I / L
    return np.array([
        [ a,  0,  0, -a,  0,  0],
        [ 0,  b,  c,  0, -b,  c],
        [ 0,  c,  d,  0, -c,  e],
        [-a,  0,  0,  a,  0,  0],
        [ 0, -b, -c,  0,  b, -c],
        [ 0,  c,  e,  0, -c,  d],
    ], dtype=float)


def element_matrices(x1, y1, x2, y2, E, A, I):
    """Return local stiffness, transformation, global stiffness, and length."""
    L = np.hypot(x2 - x1, y2 - y1)
    if L <= 0 or min(E, A, I) <= 0:
        raise ValueError("Element length, E, A, and I must be positive.")
    c = (x2 - x1) / L
    s = (y2 - y1) / L
    T = np.array([
        [ c, s, 0,  0, 0, 0],
        [-s, c, 0,  0, 0, 0],
        [ 0, 0, 1,  0, 0, 0],
        [ 0, 0, 0,  c, s, 0],
        [ 0, 0, 0, -s, c, 0],
        [ 0, 0, 0,  0, 0, 1],
    ])
    k_local = local_stiffness(E, A, I, L)
    k_global = T.T @ k_local @ T
    return k_local, T, k_global, L


def analyze_frame(nodes, elements, nodal_loads, fixed_dofs):
    """Solve a frame with three DOFs/node, zero-displacement restraints.

    nodes: [(x,y), ...]
    elements: [(node_i, node_j, E, A, I), ...], 0-based node indices
    nodal_loads: {(node_index): (Fx, Fy, Mz), ...}
    fixed_dofs: list of restrained global DOF indices, starting from zero
    """
    n_dofs = 3 * len(nodes)
    K = np.zeros((n_dofs, n_dofs))
    F = np.zeros(n_dofs)
    member_data = []

    for i, j, E, A, I in elements:
        k_local, T, k_global, L = element_matrices(
            *nodes[i], *nodes[j], E, A, I
        )
        index = [3*i, 3*i+1, 3*i+2, 3*j, 3*j+1, 3*j+2]
        K[np.ix_(index, index)] += k_global
        member_data.append((index, k_local, T))

    for node, (Fx, Fy, Mz) in nodal_loads.items():
        F[3*node:3*node+3] += [Fx, Fy, Mz]

    fixed_dofs = sorted(set(fixed_dofs))
    free_dofs = [d for d in range(n_dofs) if d not in fixed_dofs]
    Kff = K[np.ix_(free_dofs, free_dofs)]
    if np.linalg.matrix_rank(Kff) < len(free_dofs):
        raise ValueError("Unstable frame: reduced stiffness matrix is singular.")

    U = np.zeros(n_dofs)
    U[free_dofs] = np.linalg.solve(Kff, F[free_dofs])
    reactions = K @ U - F
    member_end_forces = []
    for index, k_local, T in member_data:
        d_local = T @ U[index]
        member_end_forces.append(k_local @ d_local)
    return U, reactions, member_end_forces, K


if __name__ == "__main__":
    # Cantilever column with horizontal tip load
    E = 200e9          # N/m2
    A = 0.01           # m2
    I = 8e-5           # m4
    L = 3.0            # m
    P = 10_000.0       # N

    nodes = [(0, 0), (0, L)]
    elements = [(0, 1, E, A, I)]
    nodal_loads = {1: (P, 0, 0)}
    fixed_dofs = [0, 1, 2]   # base ux, uy, theta fixed

    U, reactions, member_forces, K = analyze_frame(
        nodes, elements, nodal_loads, fixed_dofs
    )

    analytical_deflection = P * L**3 / (3 * E * I)
    analytical_moment = P * L
    print(f"Tip displacement:  {U[3]*1000:.6f} mm")
    print(f"Analytical value:  {analytical_deflection*1000:.6f} mm")
    print(f"Base Fx reaction:  {reactions[0]/1000:.6f} kN")
    print(f"Base Mz reaction:  {reactions[2]/1000:.6f} kN*m")
    print(f"Analytical moment: {analytical_moment/1000:.6f} kN*m")

    assert np.isclose(U[3], analytical_deflection, rtol=1e-10)
    assert np.isclose(reactions[0], -P)
    assert np.isclose(reactions[2], analytical_moment)
    assert np.allclose(K, K.T)
    print("Verification passed.")
