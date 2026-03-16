# Tests unitaires pour GradientDescent.
# On vérifie la convergence sur des fonctions convexes à solution connue,
# le bon fonctionnement du clipping, et l'impact des paramètres clés.

import sys, os, unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from problem.qubo import QuboProblem
from problem.knapsack import KnapsackProblem
from problem.continuous import ContinuousFunction
from solver.classical_solver.gradient_descent import GradientDescent


# ---------------------------------------------------------------------------
# Tests sur fonctions continues (ContinuousFunction + round_solution=False)
# ---------------------------------------------------------------------------

class TestGradientDescentContinu(unittest.TestCase):

    def test_convexe_simple_x2_y2(self):
        """f(x,y) = x²+y² → minimum global en (0,0) de valeur 0."""
        fn = lambda v: v[0]**2 + v[1]**2
        prob = ContinuousFunction(fn, bounds=(-2, 2), n=2)
        gd = GradientDescent(prob, maximize=False,
                             learning_rate=0.1, n_iterations=300,
                             domain=(-2, 2), round_solution=False)
        val, sol = gd.solve(initial_solution=[1.5, -1.5])
        self.assertAlmostEqual(val,    0.0, places=3)
        self.assertAlmostEqual(sol[0], 0.0, places=2)
        self.assertAlmostEqual(sol[1], 0.0, places=2)

    def test_convexe_deplace(self):
        """f(x,y) = (x-1)²+(y-2)² → minimum en (1,2) de valeur 0."""
        fn = lambda v: (v[0] - 1)**2 + (v[1] - 2)**2
        prob = ContinuousFunction(fn, bounds=(-5, 5), n=2)
        gd = GradientDescent(prob, maximize=False,
                             learning_rate=0.1, n_iterations=500,
                             domain=(-5, 5), round_solution=False)
        val, sol = gd.solve(initial_solution=[4.0, -3.0])
        self.assertAlmostEqual(val,    0.0, places=3)
        self.assertAlmostEqual(sol[0], 1.0, places=2)
        self.assertAlmostEqual(sol[1], 2.0, places=2)

    def test_maximisation_continue(self):
        """f(x,y) = -(x²+y²) → maximum en (0,0) de valeur 0."""
        fn = lambda v: -(v[0]**2 + v[1]**2)
        prob = ContinuousFunction(fn, bounds=(-2, 2), n=2)
        gd = GradientDescent(prob, maximize=True,
                             learning_rate=0.1, n_iterations=300,
                             domain=(-2, 2), round_solution=False)
        val, sol = gd.solve(initial_solution=[-1.0, 1.0])
        self.assertAlmostEqual(val,    0.0, places=3)
        self.assertAlmostEqual(sol[0], 0.0, places=2)
        self.assertAlmostEqual(sol[1], 0.0, places=2)

    def test_clipping_borne(self):
        """La solution doit rester dans le domaine à chaque itération."""
        fn = lambda v: (v[0] - 10)**2 + (v[1] - 10)**2  # minimum hors domaine
        prob = ContinuousFunction(fn, bounds=(-2, 2), n=2)
        gd = GradientDescent(prob, maximize=False,
                             learning_rate=0.5, n_iterations=200,
                             domain=(-2, 2), round_solution=False)
        val, sol = gd.solve()
        # Le minimum projeté sur [-2,2]² est (2, 2)
        self.assertAlmostEqual(sol[0], 2.0, places=1)
        self.assertAlmostEqual(sol[1], 2.0, places=1)

    def test_solution_dans_domaine(self):
        """Toutes les composantes de la solution doivent être dans [lo, hi]."""
        fn = lambda v: np.sum(v**2)
        prob = ContinuousFunction(fn, bounds=(0, 10), n=3)
        gd = GradientDescent(prob, maximize=False,
                             learning_rate=0.1, n_iterations=200,
                             domain=(0, 10), round_solution=False)
        _, sol = gd.solve()
        for xi in sol:
            self.assertGreaterEqual(xi, 0.0)
            self.assertLessEqual(xi, 10.0)

    def test_return_trajectory(self):
        """return_trajectory=True doit retourner une liste de n_iterations+1 points."""
        fn = lambda v: v[0]**2 + v[1]**2
        prob = ContinuousFunction(fn, bounds=(-2, 2), n=2)
        gd = GradientDescent(prob, maximize=False,
                             learning_rate=0.1, n_iterations=50,
                             domain=(-2, 2), round_solution=False)
        val, sol, traj = gd.solve(return_trajectory=True)
        self.assertEqual(len(traj), 51)           # point initial + 50 itérations
        self.assertEqual(len(traj[0]), 2)         # chaque point est 2D
        # La trajectoire doit converger (énergie décroissante en moyenne)
        energies = [prob.eval(p) for p in traj]
        self.assertLess(energies[-1], energies[0])

    def test_learning_rate_impact(self):
        """Un learning rate plus grand doit converger plus vite (sur convexe)."""
        fn = lambda v: v[0]**2 + v[1]**2
        prob = ContinuousFunction(fn, bounds=(-2, 2), n=2)
        x0 = [1.5, 1.5]

        gd_lent   = GradientDescent(prob, maximize=False, learning_rate=0.01,
                                    n_iterations=100, domain=(-2, 2), round_solution=False)
        gd_rapide = GradientDescent(prob, maximize=False, learning_rate=0.3,
                                    n_iterations=100, domain=(-2, 2), round_solution=False)

        val_lent,   _ = gd_lent.solve(initial_solution=x0[:])
        val_rapide, _ = gd_rapide.solve(initial_solution=x0[:])
        self.assertLess(val_rapide, val_lent,
                        "Un α plus grand devrait converger vers un meilleur optimum en 100 iters")

    def test_epsilon_gradient(self):
        """Un epsilon plus petit donne un gradient plus précis sur f convexe."""
        fn = lambda v: v[0]**2 + v[1]**2
        prob = ContinuousFunction(fn, bounds=(-2, 2), n=2)
        x0 = [1.0, 1.0]

        gd_gros   = GradientDescent(prob, maximize=False, learning_rate=0.1,
                                    n_iterations=200, domain=(-2, 2),
                                    epsilon=0.5, round_solution=False)
        gd_fin    = GradientDescent(prob, maximize=False, learning_rate=0.1,
                                    n_iterations=200, domain=(-2, 2),
                                    epsilon=1e-6, round_solution=False)

        val_gros, _ = gd_gros.solve(initial_solution=x0[:])
        val_fin,  _ = gd_fin.solve(initial_solution=x0[:])
        self.assertLessEqual(val_fin, val_gros + 1e-3,
                             "Un ε plus petit devrait donner un gradient plus précis")


# ---------------------------------------------------------------------------
# Tests sur problèmes combinatoires (comportement existant préservé)
# ---------------------------------------------------------------------------

class TestGradientDescentQUBO(unittest.TestCase):

    def test_round_solution_true_par_defaut(self):
        """Sans round_solution=False, la solution doit être binaire."""
        Q = np.diag([-1.0, -2.0, -3.0])
        prob = QuboProblem(Q)
        gd = GradientDescent(prob, maximize=False,
                             learning_rate=0.05, n_iterations=300, domain=(0, 1))
        val, sol = gd.solve()
        for xi in sol:
            self.assertIn(xi, [0.0, 1.0], f"Valeur non binaire : {xi}")

    def test_qubo_trivial(self):
        """Q diagonale négative → optimal = [1,1,1], valeur = -6."""
        Q = np.diag([-1.0, -2.0, -3.0])
        prob = QuboProblem(Q)
        best_val = float('inf')
        for _ in range(10):
            gd = GradientDescent(prob, maximize=False,
                                 learning_rate=0.1, n_iterations=500, domain=(0, 1))
            val, _ = gd.solve()
            best_val = min(best_val, val)
        self.assertAlmostEqual(best_val, -6.0)

    def test_return_trajectory_binaire(self):
        """return_trajectory doit aussi fonctionner avec round_solution=True."""
        Q = np.diag([-1.0, -2.0])
        prob = QuboProblem(Q)
        gd = GradientDescent(prob, maximize=False,
                             learning_rate=0.1, n_iterations=20, domain=(0, 1))
        val, sol, traj = gd.solve(return_trajectory=True)
        self.assertEqual(len(traj), 21)
        for xi in sol:
            self.assertIn(xi, [0.0, 1.0])


if __name__ == '__main__':
    unittest.main()
