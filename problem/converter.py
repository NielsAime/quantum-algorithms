# Fonctions de conversion entre représentations de problèmes d'optimisation.
#
# QUBO  ↔  Ising  via la substitution x_i = (1 + s_i) / 2  (et son inverse)
# Knapsack → QUBO  via la méthode de pénalité avec variables de slack binaires
#
# Ces conversions sont utiles car certains solveurs (D-Wave) travaillent en Ising
# tandis que les formulations algorithmiques utilisent souvent le format QUBO.

import numpy as np
from sympy import symbols, sympify, expand

from .qubo import QuboProblem
from .ising import IsingProblem
from .knapsack import KnapsackProblem


def qubo_to_ising(qubo_problem):
    """
    Convertit un QuboProblem en IsingProblem.
    Substitution : x_i = (1 + s_i) / 2

    J_ij = Q_ij / 4                                    (i < j)
    h_i  = Q_ii / 2 + (1/4) * Σ_{j ≠ i} Q_sym[i, j]
    """
    Q     = qubo_problem.Q
    n     = qubo_problem.n
    Q_sym = Q + Q.T - np.diag(np.diag(Q))

    J = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            J[i, j] = Q[i, j] / 4

    h = np.zeros(n)
    for i in range(n):
        off_diag = np.sum(Q_sym[i, :]) - Q_sym[i, i]
        h[i] = Q[i, i] / 2 + off_diag / 4

    s = [symbols(f"s_{i}") for i in range(n)]
    expr = sympify(0)
    for i in range(n):
        for j in range(i + 1, n):
            if J[i, j] != 0:
                expr += J[i, j] * s[i] * s[j]
        if h[i] != 0:
            expr += h[i] * s[i]

    return IsingProblem(str(expr))


def ising_to_qubo(ising_problem):
    """
    Convertit un IsingProblem en QuboProblem.
    Substitution : s_i = 2 x_i - 1

    Q_ij = 4 * J_ij                            (i < j)
    Q_ii = 2 * h_i - 2 * Σ_{j ≠ i} J_ij
    """
    expr = expand(ising_problem.to_sympy_expr())

    syms = sorted(expr.free_symbols, key=lambda sym: int(str(sym).split('_')[1]))
    if not syms:
        return QuboProblem(np.zeros((1, 1)))

    n = int(str(syms[-1]).split('_')[1]) + 1
    s = [symbols(f"s_{i}") for i in range(n)]

    J = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            c = float(expr.coeff(s[i] * s[j]))
            J[i, j] = c
            J[j, i] = c

    h = np.zeros(n)
    for i in range(n):
        expr_i = expr.subs([(s[j], 0) for j in range(n) if j != i])
        h[i] = float(expr_i.coeff(s[i]))

    Q = np.zeros((n, n))
    for i in range(n):
        Q[i, i] = 2 * h[i] - 2 * np.sum(J[i, :])
        for j in range(i + 1, n):
            Q[i, j] = 4 * J[i, j]

    return QuboProblem(Q)


def knapsack_to_qubo(knapsack_problem, penalty=None):
    """
    Convertit un KnapsackProblem en QuboProblem via la méthode pénalité.

    Formulation QUBO (minimisation) :
        minimize  -Σ v_i·x_i  +  P·(Σ w_i·x_i + Σ a_k·y_k - W)²

    Les x_i ∈ {0,1} sont les variables de décision du Knapsack.
    Les y_k ∈ {0,1} sont des variables de slack binaires encodant s ∈ [0, W] :
        s = Σ_k a_k·y_k  avec  a_k = 2^k  (tant que 2^k ≤ reste)
        puis a_K = W - (2^K - 1)  pour couvrir exactement [0, W].

    Le paramètre de pénalité P doit satisfaire P > max(v_i) pour garantir
    que la contrainte de capacité est toujours respectée à l'optimum.
    Par défaut on choisit P = Σ v_i + 1 (borne sûre).

    Le problème Knapsack est une maximisation ; on le transforme en minimisation
    en négativant les valeurs dans l'objectif QUBO.

    Parameters
    ----------
    knapsack_problem : KnapsackProblem
    penalty          : float, optionnel — coefficient de pénalité P

    Returns
    -------
    QuboProblem de taille (n + num_slack) × (n + num_slack)
    """
    n       = knapsack_problem.n
    weights = knapsack_problem.weights
    values  = knapsack_problem.values
    W       = knapsack_problem.capacity

    if penalty is None:
        # P > max(v_i) est la condition minimale ; on prend sum + 1 pour plus de robustesse
        penalty = float(sum(values) + 1)

    # --- Encodage binaire des variables de slack pour s ∈ [0, W] ---
    # s = Σ a_k · y_k  avec les puissances de 2 jusqu'à couvrir W
    slack_coeffs = []
    remaining = W
    k = 0
    while remaining > 0:
        a = min(2 ** k, remaining)
        slack_coeffs.append(float(a))
        remaining -= a
        k += 1

    num_slack   = len(slack_coeffs)
    total_vars  = n + num_slack

    # Coefficients dans la contrainte pénalisée : c_i = w_i (variables x)
    #                                              c_{n+k} = a_k (variables slack y)
    c = [float(w) for w in weights] + slack_coeffs

    Q = np.zeros((total_vars, total_vars))

    # --- Terme objectif : minimize -Σ v_i · x_i ---
    for i in range(n):
        Q[i, i] -= values[i]

    # --- Terme de pénalité : P · (Σ c_i · z_i - W)² ---
    # Développement (en exploitant z_i² = z_i) :
    #   diagonale   : P · c_i · (c_i - 2W)
    #   hors-diag   : P · 2 · c_i · c_j   (i < j, triangulaire supérieure)
    for i in range(total_vars):
        Q[i, i] += penalty * c[i] * (c[i] - 2 * W)
        for j in range(i + 1, total_vars):
            Q[i, j] += penalty * 2.0 * c[i] * c[j]

    return QuboProblem(Q)
