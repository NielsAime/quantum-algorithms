from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

def add_ising_mixer_ham(qc, ising_problem, n):
    """
    Ajoute le mixing Hamiltonian au circuit (rotations Rx sur chaque qubit).
    Retourne le circuit et la liste des paramètres beta.
    """
    betas = ParameterVector('beta', 1)
    
    for i in range(n):
        qc.rx(2 * betas[0], i)
    
    return qc, list(betas)


def add_ising_problem_ham(qc, ising_problem, n):
    """
    Ajoute le problem Hamiltonian au circuit (rotations Rz pour les termes linéaires,
    et RZZ pour les termes quadratiques).
    Retourne le circuit et la liste des paramètres gamma.
    """
    gammas = ParameterVector('gamma', 1)
    
    # Termes linéaires : Rz sur chaque qubit
    for i, h in ising_problem['linear'].items():
        qc.rz(2 * h * gammas[0], i)
    
    # Termes quadratiques : RZZ entre les paires de qubits
    for (i, j), J in ising_problem['quadratic'].items():
        qc.cx(i, j)
        qc.rz(2 * J * gammas[0], j)
        qc.cx(i, j)
    
    return qc, list(gammas)