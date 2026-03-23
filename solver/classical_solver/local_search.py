# Local search for QUBO problems (greedy hill climbing).
#
# The algorithm starts from a random binary solution and repeatedly looks at
# a neighbor. A neighbor is produced by QuboProblem.gen_neighbor_sol(), which
# flips exactly one bit chosen uniformly at random (so a 1-bit Hamming distance move).
#
# We accept the neighbor only when it strictly improves the objective.
# This is "first improvement" hill climbing: we do not scan all neighbors,
# we just pick one at random and decide immediately.
#
# Because QUBO minimizes f(x) = x^T Q x, "better" always means "lower value".
#
# Pros and cons of greedy hill climbing:
#   + Always converges (finite space + strict improvement = no cycles)
#   + Fast per iteration: one eval() call per step
#   - Gets trapped in local optima, especially for non-convex QUBO matrices
#   - To get better coverage, run the solver several times from different
#     random starting points and keep the overall best result.
#
# Libraries:
#   random (standard library): used to draw the random starting solution.

import random


class LocalSearch:

    def __init__(self, problem, n_iterations=1000):
        """
        Parameters
        ----------
        problem      : QuboProblem instance. Must have eval() and gen_neighbor_sol().
        n_iterations : maximum number of neighbor evaluations before stopping.
                       The search may stop earlier if no improvement is found,
                       but here we keep trying (random restarts are left to the caller).
        """
        self.problem = problem
        self.n_iterations = n_iterations

    def _random_initial_solution(self):
        """Return a random starting point: dict {index: 0 or 1} for all n variables."""
        return {i: random.randint(0, 1) for i in range(self.problem.n)}

    def solve(self, initial_solution=None):
        """
        Run greedy hill climbing from an initial solution.
        At each step, a single random neighbor is evaluated.
        The move is accepted if and only if it lowers the QUBO cost.

        Parameters
        ----------
        initial_solution : optional dict {index: value in {0, 1}}.
                           If not provided, a random solution is generated.

        Returns
        -------
        (best_value, best_solution)
            best_value   : value of QUBO objective at the returned solution
            best_solution: dict {index: 0 or 1}
        """
        if initial_solution is None:
            solution = self._random_initial_solution()
        else:
            solution = dict(initial_solution)

        current_val = self.problem.eval(solution)

        for _ in range(self.n_iterations):
            neighbor = self.problem.gen_neighbor_sol(solution)
            neighbor_val = self.problem.eval(neighbor)

            # Accept only strict improvements (pure greedy, no probabilistic acceptance).
            if neighbor_val < current_val:
                solution = neighbor
                current_val = neighbor_val

        return current_val, solution
