# Algorithme Simulated Annealing (recuit simulé) générique.
#
# Inspiré de la physique statistique : on simule le refroidissement d'un métal fondu.
# À haute température, on accepte souvent des solutions plus mauvaises (exploration).
# À basse température, on n'accepte presque plus les dégradations (exploitation).
#
# Critère de Metropolis : on accepte une nouvelle configuration de coût delta_E > 0
# avec une probabilité p = exp(-delta_E / T), où T est la température courante.
#
# Ce solveur est générique : il fonctionne avec tout problème qui expose une méthode
# eval(solution). Le sens d'optimisation (minimisation ou maximisation) et la
# génération de voisins sont configurables via des paramètres.
#
# Compatibilité avec les problèmes du projet :
#   - KnapsackProblem  : maximize=True,  neighbor_fn par défaut (flip de bit {0,1})
#   - QuboProblem      : maximize=False, neighbor_fn par défaut (flip de bit {0,1})
#   - IsingProblem     : maximize=False, neighbor_fn custom (flip de spin {-1,+1})
#   - MaxCutProblem    : maximize=False, neighbor_fn custom (flip de spin {-1,+1})
#   - TspProblem       : maximize=False, neighbor_fn custom (swap de villes)

import numpy as np


class SimulatedAnnealing:

    def __init__(self, problem, maximize=False,
                 initial_temperature=100.0, cooling_rate=0.95, min_temperature=0.01,
                 steps_per_temperature=1, neighbor_fn=None):
        """
        Parameters
        ----------
        problem                 : objet avec .eval(solution) et .n
        maximize                : True pour maximiser eval(), False (défaut) pour minimiser
        initial_temperature     : température de départ T₀
        cooling_rate            : facteur β tel que T ← β·T  (décroissance exponentielle)
        min_temperature         : seuil d'arrêt : on stoppe quand T < min_temperature
        steps_per_temperature   : nombre de tentatives de déplacement à chaque niveau de T
        neighbor_fn             : fonction (solution) → solution voisine.
                                  Par défaut : flip d'un bit aléatoire pour vecteurs binaires {0,1}.
                                  Fournir une fonction custom pour Ising {-1,+1} ou TSP.
        """
        self.problem = problem
        self.maximize = maximize
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.steps_per_temperature = steps_per_temperature
        self._neighbor_fn = neighbor_fn or self._flip_bit_neighbor

    # ------------------------------------------------------------------
    # Voisinage par défaut : flip d'un bit (problèmes binaires {0,1})
    # ------------------------------------------------------------------
    def _flip_bit_neighbor(self, solution):
        neighbor = list(solution)
        idx = np.random.randint(len(neighbor))
        neighbor[idx] ^= 1
        return neighbor

    # ------------------------------------------------------------------
    # Énergie interne : on minimise toujours en interne.
    # Pour maximiser, on travaille sur l'opposé de eval().
    # Si eval() retourne -inf (solution infaisable), énergie = +inf.
    # ------------------------------------------------------------------
    def _energy(self, solution):
        val = self.problem.eval(solution)
        if val == float('-inf'):
            return float('inf')
        return -val if self.maximize else float(val)

    # ------------------------------------------------------------------

    def solve(self, initial_solution=None):
        """
        Parameters
        ----------
        initial_solution : solution de départ (optionnelle).
                           Par défaut : vecteur binaire {0,1} aléatoire de taille problem.n.

        Returns
        -------
        (best_value, best_solution)
            best_value    : valeur de eval() pour la meilleure solution trouvée
            best_solution : la solution correspondante (même format que initial_solution)
        """
        if initial_solution is None:
            initial_solution = [np.random.randint(0, 2) for _ in range(self.problem.n)]

        solution = list(initial_solution)
        current_energy = self._energy(solution)

        best_solution = solution[:]
        best_energy = current_energy

        temperature = self.initial_temperature

        while temperature > self.min_temperature:
            for _ in range(self.steps_per_temperature):
                neighbor = self._neighbor_fn(solution)
                neighbor_energy = self._energy(neighbor)

                delta_E = neighbor_energy - current_energy

                # Acceptation : amélioration immédiate ou critère de Metropolis
                if delta_E < 0 or np.random.rand() < np.exp(-delta_E / temperature):
                    solution = neighbor
                    current_energy = neighbor_energy

                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_solution = solution[:]

            temperature *= self.cooling_rate

        best_value = self.problem.eval(best_solution)
        return best_value, best_solution
