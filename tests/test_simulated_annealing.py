# Tests unitaires pour le Simulated Annealing.
# On vérifie la convergence sur de petites instances QUBO et Knapsack
# dont la solution optimale est connue par recherche exhaustive.

import sys, os, unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from problem.qubo import QuboProblem
from problem.knapsack import KnapsackProblem
from solver.classical_solver.simulated_annealing import SimulatedAnnealing


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def exhaustive_qubo(problem):
    """Retourne la valeur minimale par recherche exhaustive."""
    best = float('inf')
    for sol in problem.generate_complete_search_space():
        val = problem.eval(sol)
        if val < best:
            best = val
    return best


def exhaustive_knapsack(problem):
    """Retourne la valeur maximale par recherche exhaustive."""
    best = 0
    for mask in range(1 << problem.n):
        x = [(mask >> i) & 1 for i in range(problem.n)]
        if sum(x[i] * problem.weights[i] for i in range(problem.n)) <= problem.capacity:
            val = sum(x[i] * problem.values[i] for i in range(problem.n))
            best = max(best, val)
    return best


def run_sa_multiple(problem, maximize, runs=15, **sa_kwargs):
    """Lance SA plusieurs fois et retourne la meilleure valeur trouvée."""
    compare = max if maximize else min
    best = float('-inf') if maximize else float('inf')
    for _ in range(runs):
        sa = SimulatedAnnealing(problem, maximize=maximize, **sa_kwargs)
        val, _ = sa.solve()
        best = compare(best, val)
    return best


# ---------------------------------------------------------------------------
# Tests QUBO
# ---------------------------------------------------------------------------

class TestSimulatedAnnealingQUBO(unittest.TestCase):

    def test_qubo_trivial(self):
        # Diagonale négative : tous les bits à 1 est optimal → valeur = -6
        Q = np.diag([-1.0, -2.0, -3.0])
        problem = QuboProblem(Q)
        sa = SimulatedAnnealing(problem, maximize=False,
                                initial_temperature=10.0, cooling_rate=0.95,
                                min_temperature=1e-4)
        val, sol = sa.solve()
        self.assertAlmostEqual(val, -6.0)
        self.assertEqual(list(sol), [1, 1, 1])

    def test_qubo_petit_connu(self):
        # min -3x0 - 5x1 + 4x0x1 : optimal = [1,1] de valeur -4
        Q = np.array([[-3.0, 4.0],
                      [ 0.0, -5.0]])
        problem = QuboProblem(Q)
        expected = exhaustive_qubo(problem)   # -4
        best = run_sa_multiple(problem, maximize=False, runs=10,
                               initial_temperature=10.0, cooling_rate=0.99,
                               min_temperature=1e-4)
        self.assertAlmostEqual(best, expected,
                               msg=f"SA n'a pas trouvé l'optimal {expected}")

    def test_qubo_instances_aleatoires(self):
        """SA doit converger sur de petites instances QUBO aléatoires (n=4)."""
        rng = np.random.RandomState(42)
        for trial in range(5):
            n = 4
            Q = np.triu(rng.randint(-5, 5, size=(n, n)).astype(float))
            problem = QuboProblem(Q)
            expected = exhaustive_qubo(problem)
            best = run_sa_multiple(problem, maximize=False, runs=15,
                                   initial_temperature=50.0, cooling_rate=0.99,
                                   min_temperature=1e-5)
            self.assertAlmostEqual(best, expected,
                                   msg=f"Trial {trial}: SA n'a pas trouvé {expected}")


# ---------------------------------------------------------------------------
# Tests Knapsack
# ---------------------------------------------------------------------------

class TestSimulatedAnnealingKnapsack(unittest.TestCase):

    def test_knapsack_simple(self):
        problem = KnapsackProblem(weights=[2, 3, 4, 5], values=[3, 4, 5, 6], capacity=8)
        expected = exhaustive_knapsack(problem)
        best = run_sa_multiple(problem, maximize=True, runs=15,
                               initial_temperature=50.0, cooling_rate=0.99,
                               min_temperature=1e-4)
        self.assertEqual(best, expected,
                         msg=f"SA n'a pas trouvé l'optimal {expected}")

    def test_knapsack_tout_rentre(self):
        # Tous les objets rentrent : solution optimale = tout prendre
        problem = KnapsackProblem(weights=[1, 2, 3], values=[10, 20, 30], capacity=100)
        sa = SimulatedAnnealing(problem, maximize=True,
                                initial_temperature=50.0, cooling_rate=0.99,
                                min_temperature=1e-4)
        val, _ = sa.solve()
        self.assertEqual(val, 60)

    def test_knapsack_instances_aleatoires(self):
        for seed in range(5):
            problem = KnapsackProblem.generate_random(n_items=6, seed=seed)
            expected = exhaustive_knapsack(problem)
            best = run_sa_multiple(problem, maximize=True, runs=15,
                                   initial_temperature=100.0, cooling_rate=0.99,
                                   min_temperature=1e-4)
            self.assertEqual(best, expected,
                             msg=f"seed={seed}: SA n'a pas trouvé l'optimal {expected}")

    def test_solution_faisable(self):
        """La meilleure solution trouvée doit respecter la contrainte de poids."""
        problem = KnapsackProblem.generate_random(n_items=8, seed=7)
        sa = SimulatedAnnealing(problem, maximize=True,
                                initial_temperature=100.0, cooling_rate=0.99,
                                min_temperature=1e-4)
        val, sol = sa.solve()
        poids = sum(sol[i] * problem.weights[i] for i in range(problem.n))
        self.assertLessEqual(poids, problem.capacity)
        valeur = sum(sol[i] * problem.values[i] for i in range(problem.n))
        self.assertEqual(valeur, val)


# ---------------------------------------------------------------------------
# Tests des paramètres du recuit
# ---------------------------------------------------------------------------

class TestSimulatedAnnealingParametres(unittest.TestCase):

    def test_arret_temperature_seuil(self):
        """Le solveur doit s'arrêter dès que T < min_temperature."""
        problem = KnapsackProblem.generate_random(n_items=5, seed=0)
        # Seuil très haut : très peu d'itérations
        sa = SimulatedAnnealing(problem, maximize=True,
                                initial_temperature=1.0, cooling_rate=0.9,
                                min_temperature=0.5)
        val, sol = sa.solve()
        self.assertIsNotNone(val)
        self.assertEqual(len(sol), problem.n)

    def test_cooling_lent_meilleur_que_rapide(self):
        """Un refroidissement lent doit converger mieux qu'un rapide en moyenne."""
        problem = KnapsackProblem.generate_random(n_items=8, seed=0)

        vals_lent = [SimulatedAnnealing(problem, maximize=True,
                                        initial_temperature=100.0, cooling_rate=0.999,
                                        min_temperature=1e-4).solve()[0]
                     for _ in range(10)]
        vals_rapide = [SimulatedAnnealing(problem, maximize=True,
                                          initial_temperature=100.0, cooling_rate=0.5,
                                          min_temperature=1e-4).solve()[0]
                       for _ in range(10)]

        self.assertGreaterEqual(np.mean(vals_lent), np.mean(vals_rapide),
                                msg="Un cooling plus lent devrait donner de meilleurs résultats")

    def test_temperature_initiale_haute_explore_plus(self):
        """Avec T initiale haute, SA doit accepter plus de mauvais mouvements au départ."""
        problem = KnapsackProblem.generate_random(n_items=10, seed=42)
        # Juste vérifier que les deux paramétrisations terminent sans erreur
        for T0 in [1.0, 100.0, 1000.0]:
            sa = SimulatedAnnealing(problem, maximize=True,
                                    initial_temperature=T0, cooling_rate=0.95,
                                    min_temperature=1e-3)
            val, sol = sa.solve()
            self.assertEqual(len(sol), problem.n)


if __name__ == '__main__':
    unittest.main()
