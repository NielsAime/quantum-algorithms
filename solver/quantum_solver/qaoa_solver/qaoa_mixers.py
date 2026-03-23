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


def add_mis_mixer_ham(qc, graph_edges, n, layer=0):
    """
    Constrained mixer for the MIS (Maximum Independent Set) problem.

    A node can only be flipped (mixed) if none of its neighbors are currently active.
    This keeps the circuit inside the feasible subspace throughout the whole evolution,
    which is the key advantage over a standard Rx mixer with penalty terms.

    For isolated nodes (no neighbors):
      They are always valid members of the MIS, so we fix them to |1> with an X gate
      at layer 0 and never mix them afterwards. Applying a free Rx rotation on them
      (old behavior) was incorrect because it would put them in superposition and let
      them collapse to |0> at measurement.

    For nodes with neighbors:
      We apply a multi-controlled Rx gate. The construction is:
        1. X on each neighbor  (flip |0>->|1> so the gate fires when neighbors were |0>)
        2. MCRx(2*beta, neighbors -> i)  (rotate i only when all controls are |1>)
        3. X on each neighbor  (unflip to restore original state)

    Why MCRx and not the old MCX-Rx-MCX pattern?
      MCX-Rx-MCX is NOT a controlled Rx. When controls are |0>, MCX does nothing but
      Rx still fires unconditionally on qubit i. When controls are |1>, the net effect
      is X*Rx(theta)*X = Rx(-theta), which has the wrong sign. Both problems at once.
      RXGate(angle).control(k) from Qiskit is the correct gate: it applies the rotation
      only when all k controls are |1>, and leaves the target untouched otherwise.

    Libraries:
      RXGate from qiskit.circuit.library: parametric Rx gate. Calling .control(k) on it
      produces a k-controlled Rx that works correctly with symbolic ParameterVector angles.
    """
    from qiskit.circuit.library import RXGate

    betas = ParameterVector(f'beta_mis_{layer}', 1)

    for i in range(n):
        neighbors = [j for (u, v) in graph_edges if (u == i and (j := v)) or (v == i and (j := u))]

        if not neighbors:
            # Isolated node: always in the MIS, fix it to |1> once and leave it alone.
            if layer == 0:
                qc.x(i)
            # No mixer rotation: this qubit is frozen at |1> for the entire circuit.
        else:
            # Flip neighbors: |0> becomes |1>, so the multi-controlled gate fires
            # exactly when all original neighbors were |0> (constraint satisfied).
            for neighbor in neighbors:
                qc.x(neighbor)

            # Apply Rx(2*beta) on qubit i only when all neighbor controls are |1>.
            # RXGate(angle).control(k) creates a proper k-controlled Rx gate.
            mcrx = RXGate(2 * betas[0]).control(len(neighbors))
            qc.append(mcrx, neighbors + [i])

            # Unflip neighbors to restore their original state.
            for neighbor in neighbors:
                qc.x(neighbor)

    return qc, list(betas)
