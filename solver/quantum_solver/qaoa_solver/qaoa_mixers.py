from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def add_ising_problem_ham(qc, ising_problem, n, layer=0):
    """
    Apply the cost Hamiltonian U_C(gamma) = e^{-i*gamma*H_C} to the circuit.

    H_C is the Ising cost function: H_C = sum_i h_i*Z_i + sum_{i<j} J_ij*Z_i*Z_j
    Each term is implemented as a parametric gate:
      - Linear term h_i: single Rz(2*h*gamma) on qubit i
      - Quadratic term J_ij: CX-Rz(2*J*gamma)-CX pattern, which implements e^{-i*gamma*J*Z_i*Z_j}

    Gates with a zero coefficient are skipped entirely. Without this check, Qiskit would
    add Rz(0*gamma) gates that appear in the circuit diagram as real gates but are
    just identities. This was causing spurious single-Rz rotations at the start of
    the max-cut circuit (where all linear coefficients are 0).
    """
    gammas = ParameterVector(f'gamma_{layer}', 1)

    for i, h in ising_problem['linear'].items():
        if h != 0:
            qc.rz(2 * h * gammas[0], i)

    for (i, j), J in ising_problem['quadratic'].items():
        if J != 0:
            qc.cx(i, j)
            qc.rz(2 * J * gammas[0], j)
            qc.cx(i, j)

    return qc, list(gammas)


def add_ising_mixer_ham(qc, ising_problem, n, layer=0):
    """
    Apply the standard mixer Hamiltonian U_B(beta) = e^{-i*beta*H_B} to the circuit.

    H_B = sum_i X_i (transverse field), implemented as Rx(2*beta) on every qubit.
    This is the default QAOA mixer: it explores the full {-1,+1}^n space freely
    with no constraint awareness.
    """
    betas = ParameterVector(f'beta_{layer}', 1)

    for i in range(n):
        qc.rx(2 * betas[0], i)

    return qc, list(betas)


def add_mis_mixer_ham(qc, graph_edges, n, layer=0, ancilla=None):
    """
    Constrained mixer for the MIS (Maximum Independent Set) problem.

    A node can only be flipped (mixed) if none of its neighbors are currently active.
    This keeps the circuit inside the feasible subspace throughout the whole evolution,
    which is the key advantage over a standard Rx mixer with penalty terms.

    For isolated nodes (no neighbors):
      They are always valid members of the MIS, so we fix them to |1> with an X gate
      at layer 0 and never mix them afterwards.

    For nodes with exactly 1 neighbor:
      Use qc.crx() directly. CRX is a native 2-qubit gate in Qiskit, no synthesis needed.

    For nodes with k > 1 neighbors:
      RXGate(...).control(k) requires Qiskit to synthesize a parametric multi-controlled
      gate symbolically, which fails (even after installing qiskit-synthesis).
      The correct fix uses an ancilla qubit:
        1. X on each neighbor           (flip so gate fires when neighbors were |0>)
        2. MCX(neighbors, ancilla)      (compute AND of all neighbor conditions into ancilla)
        3. CRX(2*beta, ancilla, i)      (rotate i only when ancilla = |1>)
        4. MCX(neighbors, ancilla)      (uncompute ancilla back to |0>)
        5. X on each neighbor           (restore original state)
      MCX and CRX are both standard gates with no synthesis issues.
      The ancilla qubit must be added to the circuit by the caller (qubit index n).

    Parameters
    ----------
    ancilla : qubit index of the ancilla (required when any node has more than 1 neighbor).
              Typically n (the circuit has n+1 qubits total).
    """
    betas = ParameterVector(f'beta_mis_{layer}', 1)

    for i in range(n):
        neighbors = [j for (u, v) in graph_edges if (u == i and (j := v)) or (v == i and (j := u))]

        if not neighbors:
            # Isolated node: always in the MIS, fix it to |1> once and leave it alone.
            if layer == 0:
                qc.x(i)

        elif len(neighbors) == 1:
            # Single neighbor: CRX is natively supported, no ancilla needed.
            qc.x(neighbors[0])
            qc.crx(2 * betas[0], neighbors[0], i)
            qc.x(neighbors[0])

        else:
            # Multiple neighbors: compute AND into ancilla, then apply CRX.
            if ancilla is None:
                raise ValueError(
                    f"Node {i} has {len(neighbors)} neighbors but no ancilla qubit was provided. "
                    "Create the circuit with n+1 qubits and pass ancilla=n."
                )
            for neighbor in neighbors:
                qc.x(neighbor)
            qc.mcx(neighbors, ancilla)          # ancilla = 1 iff all neighbors were 0
            qc.crx(2 * betas[0], ancilla, i)    # rotate i only when constraint is satisfied
            qc.mcx(neighbors, ancilla)          # uncompute ancilla
            for neighbor in neighbors:
                qc.x(neighbor)

    return qc, list(betas)
