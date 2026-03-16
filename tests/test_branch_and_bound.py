# Tests unitaires pour vérifier que le Branch and Bound fonctionne correctement.
# On compare les résultats du B&B avec une recherche exhaustive (brute force)
# sur des petites instances où on est sûr de la solution optimale.

import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from problem.knapsack import KnapsackProblem
from solver.classical_solver.branch_and_bound import BranchAndBound


def recherche_exhaustive(problem):
    meilleure_valeur = 0
    for mask in range(1 << problem.n):
        poids_total = sum(problem.weights[i] for i in range(problem.n) if mask & (1 << i))
        valeur_totale = sum(problem.values[i] for i in range(problem.n) if mask & (1 << i))
        if poids_total <= problem.capacity:
            meilleure_valeur = max(meilleure_valeur, valeur_totale)
    return meilleure_valeur


class TestBranchAndBound(unittest.TestCase):

    def _verifier(self, problem):
        attendu = recherche_exhaustive(problem)
        bb = BranchAndBound(problem)
        valeur, items = bb.solve()
        self.assertEqual(valeur, attendu)
        poids_solution = sum(w for w, inc in zip(problem.weights, items) if inc)
        self.assertLessEqual(poids_solution, problem.capacity)
        valeur_solution = sum(v for v, inc in zip(problem.values, items) if inc)
        self.assertEqual(valeur_solution, valeur)

    def test_instance_simple(self):
        problem = KnapsackProblem(weights=[2, 3, 4, 5], values=[3, 4, 5, 6], capacity=8)
        self._verifier(problem)

    def test_tout_rentre(self):
        problem = KnapsackProblem(weights=[1, 2, 3], values=[10, 20, 30], capacity=100)
        bb = BranchAndBound(problem)
        valeur, items = bb.solve()
        self.assertEqual(valeur, 60)
        self.assertTrue(all(items))

    def test_capacite_zero(self):
        problem = KnapsackProblem(weights=[2, 3, 4], values=[3, 4, 5], capacity=0)
        bb = BranchAndBound(problem)
        valeur, items = bb.solve()
        self.assertEqual(valeur, 0)
        self.assertFalse(any(items))

    def test_instances_aleatoires(self):
        for seed in range(10):
            problem = KnapsackProblem.generate_random(n_items=8, seed=seed)
            self._verifier(problem)


if __name__ == '__main__':
    unittest.main()