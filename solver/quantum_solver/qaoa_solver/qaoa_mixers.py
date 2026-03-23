from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def add_ising_problem_ham(qc, ising_problem, n, layer=0):
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
    betas = ParameterVector(f'beta_{layer}', 1)

    for i in range(n):
        qc.rx(2 * betas[0], i)

    return qc, list(betas)

def add_mis_mixer_ham(qc, graph_edges, n, layer=0):
    """
    Constrained mixer for the MIS problem.
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