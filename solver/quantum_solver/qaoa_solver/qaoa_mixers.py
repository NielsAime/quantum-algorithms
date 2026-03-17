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

def add_mis_mixer_ham(qc, graph_edges, n, layer=0):
    """
    Mixer spécifique pour le problème MIS.
    Applique une rotation conditionnelle : on ne peut "activer" un noeud
    que si aucun de ses voisins n'est actif.
    """
    betas = ParameterVector(f'beta_mis_{layer}', 1)
    
    for i in range(n):
        # Récupère les voisins du noeud i
        neighbors = [j for (u, v) in graph_edges if (u == i and (j := v)) or (v == i and (j := u))]
        
        if not neighbors:
            qc.rx(2 * betas[0], i)
        else:
            # Applique Rx conditionnel : actif seulement si tous les voisins sont à 0
            for neighbor in neighbors:
                qc.x(neighbor)
            qc.mcx(neighbors, i)  # multi-controlled X
            qc.rx(2 * betas[0], i)
            qc.mcx(neighbors, i)
            for neighbor in neighbors:
                qc.x(neighbor)
    
    return qc, list(betas)