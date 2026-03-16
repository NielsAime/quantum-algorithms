# Algorithme de Descente de Gradient par relaxation continue.
#
# Pour les problèmes à variables binaires ({0,1} ou {-1,+1}), les variables sont
# relâchées dans un domaine continu [lo, hi]. La descente s'effectue dans cet espace
# continu via un gradient calculé numériquement (différences finies), puis la solution
# est arrondie au plus proche binaire pour obtenir une solution admissible.
#
# Compatibilité avec les problèmes du projet :
#   - KnapsackProblem  : domain=(0,1),  maximize=True
#   - QuboProblem      : domain=(0,1),  maximize=False
#   - IsingProblem     : domain=(-1,1), maximize=False  (solution = dict {i: val})
#   - MaxCutProblem    : domain=(-1,1), maximize=False  (solution = dict {i: val})
#
# Note : la descente de gradient ne s'applique pas au TSP (espace de permutations).

import numpy as np


class GradientDescent:

    def __init__(self, problem, maximize=False,
                 learning_rate=0.05, n_iterations=500,
                 domain=(0, 1), epsilon=1e-5, round_solution=True):
        """
        Parameters
        ----------
        problem         : objet avec .eval(solution) et .n
        maximize        : True pour maximiser eval(), False (défaut) pour minimiser
        learning_rate   : pas de gradient (α)
        n_iterations    : nombre de pas de descente
        domain          : (lo, hi) — domaine de relaxation continu.
                          (0, 1)  pour variables binaires {0,1}  (QUBO, Knapsack)
                          (-1, 1) pour variables de spin {-1,+1} (Ising, MaxCut)
                          (lo,hi) quelconque pour ContinuousFunction
        epsilon         : pas de différences finies pour le gradient numérique (δ)
        round_solution  : True  → arrondi final vers {lo, hi} (problèmes binaires)
                          False → retourne la solution continue telle quelle
                                  (à utiliser avec ContinuousFunction)
        """
        self.problem        = problem
        self.maximize       = maximize
        self.learning_rate  = learning_rate
        self.n_iterations   = n_iterations
        self.domain         = domain
        self._epsilon       = epsilon
        self.round_solution = round_solution

    # ------------------------------------------------------------------
    # Énergie interne (on minimise toujours en interne).
    # eval() retournant -inf (infaisable) est remplacé par une grande valeur.
    # ------------------------------------------------------------------
    def _energy(self, x):
        val = self.problem.eval(list(x))
        if val == float('-inf') or val == float('inf') or (isinstance(val, float) and np.isnan(val)):
            return 1e9
        return -float(val) if self.maximize else float(val)

    # ------------------------------------------------------------------
    # Gradient numérique par différences finies centrées
    # ------------------------------------------------------------------
    def _gradient(self, x):
        grad = np.zeros(len(x))
        for i in range(len(x)):
            x_plus  = x.copy(); x_plus[i]  += self._epsilon
            x_minus = x.copy(); x_minus[i] -= self._epsilon
            grad[i] = (self._energy(x_plus) - self._energy(x_minus)) / (2 * self._epsilon)
        return grad

    # ------------------------------------------------------------------
    # Arrondi au plus proche valeur binaire dans le domaine
    # ------------------------------------------------------------------
    def _round_to_binary(self, x):
        lo, hi = self.domain
        mid = (lo + hi) / 2
        return [hi if xi > mid else lo for xi in x]

    # ------------------------------------------------------------------

    def solve(self, initial_solution=None, return_trajectory=False):
        """
        Parameters
        ----------
        initial_solution   : solution de départ (optionnelle).
                             Par défaut : vecteur aléatoire dans le domaine continu.
        return_trajectory  : si True, retourne aussi la liste des positions x à
                             chaque itération (utile pour visualiser la descente).

        Returns
        -------
        (value, solution)                     si return_trajectory=False
        (value, solution, trajectory)         si return_trajectory=True
            value      : valeur de eval() pour la solution finale
            solution   : solution finale (continue si round_solution=False,
                         arrondie binaire sinon)
            trajectory : liste de tableaux numpy, un par itération
        """
        lo, hi = self.domain
        n = self.problem.n

        if initial_solution is None:
            x = np.random.uniform(lo, hi, n)
        else:
            x = np.array(initial_solution, dtype=float)

        trajectory = [x.copy()] if return_trajectory else None

        for _ in range(self.n_iterations):
            grad = self._gradient(x)
            x    = x - self.learning_rate * grad
            x    = np.clip(x, lo, hi)
            if return_trajectory:
                trajectory.append(x.copy())

        if self.round_solution:
            solution = self._round_to_binary(x)
        else:
            solution = list(x)

        value = self.problem.eval(solution)

        if return_trajectory:
            return value, solution, trajectory
        return value, solution
