# Random search: randomly samples configurations from the solution space.
#
# At each iteration we draw a brand-new solution at random and compare it to
# the running best. If the new solution is better, it becomes the new best.
# No gradient or neighborhood structure is used, just pure random sampling.
#
# This approach scales to much larger problems than exhaustive search because
# the number of evaluations is fixed (n_iterations), not exponential. The
# trade-off is that there is no guarantee of finding the global optimum.
#
# Default solution generator:
#   A random binary vector of length problem.n, with values in {0, 1}.
#   This works directly for QuboProblem and KnapsackProblem.
#   For other problem types, pass a custom solution_gen_fn:
#     - Ising / MaxCut: sample random spins from {-1, +1} as a dict
#     - TSP: generate a random permutation of city names
#
# Libraries:
#   random (standard library): used to draw random integers for the binary default.

import random


class RandomSearch:

    def __init__(self, problem, maximize=False, n_iterations=1000, solution_gen_fn=None):
        """
        Parameters
        ----------
        problem          : any problem class with eval() and .n
        maximize         : True for maximization problems (e.g. Knapsack),
                           False (default) for minimization (e.g. QUBO).
        n_iterations     : number of random solutions to evaluate.
        solution_gen_fn  : optional callable with no arguments that returns a
                           random solution. Defaults to a binary list of length n.
                           Example for Ising: lambda: {i: random.choice([-1, 1]) for i in range(problem.n)}
        """
        self.problem = problem
        self.maximize = maximize
        self.n_iterations = n_iterations
        self._gen = solution_gen_fn or self._random_binary

    def _random_binary(self):
        """Default generator: a list of n random bits in {0, 1}."""
        return [random.randint(0, 1) for _ in range(self.problem.n)]

    def solve(self):
        """
        Sample n_iterations random solutions and return the best one found.
        The running best is updated every time a strictly better solution appears.
        Infeasible solutions (those where eval() returns -inf) are skipped.

        Returns
        -------
        (best_value, best_solution)
        """
        best_val = float('-inf') if self.maximize else float('inf')
        best_sol = None

        for _ in range(self.n_iterations):
            sol = self._gen()
            val = self.problem.eval(sol)

            if val == float('-inf'):
                continue  # infeasible (e.g. Knapsack capacity exceeded)

            if self.maximize and val > best_val:
                best_val, best_sol = val, sol
            elif not self.maximize and val < best_val:
                best_val, best_sol = val, sol

        return best_val, best_sol
