from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def add_ising_problem_ham(qc, ising_problem, n, layer=0):
    gammas = ParameterVector(f'gamma_{layer}', 1)

    for i, h in ising_problem['linear'].items():
        qc.rz(2 * h * gammas[0], i)

    for (i, j), J in ising_problem['quadratic'].items():
        qc.cx(i, j)
        qc.rz(2 * J * gammas[0], j)
        qc.cx(i, j)

    return qc, list(gammas)


def add_ising_mixer_ham(qc, ising_problem, n, layer=0):
    betas = ParameterVector(f'beta_{layer}', 1)

    for i in range(n):
        qc.rx(2 * betas[0], i)

    return qc, list(betas)