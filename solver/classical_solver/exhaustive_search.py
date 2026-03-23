# Exhaustive search: tries every possible solution in the search space.
#
# For each problem, we call generate_complete_search_space(), which is a generator
# that yields all valid configurations. For binary problems (QUBO, Knapsack, Ising,
# MaxCut) that is 2^n solutions. For TSP it is (n-1)! permutations.
# We evaluate each one with eval() and keep the best.
#
# Because the search space grows exponentially, this only works for small instances
# (roughly n <= 20 for binary, n <= 10 for TSP).
#
# Parallelization strategy:
#   We also provide solve_parallel(), which splits the full list of solutions into
#   equal-sized chunks and hands each chunk to a worker process. Every worker
#   independently finds the best solution in its chunk, then the main process
#   picks the global winner from those partial results.
#   This uses Python's multiprocessing module (standard library).
#   Worker communication relies on pickle, which works for all problem types in
#   this project: numpy arrays (QuboProblem, KnapsackProblem) and SymPy expressions
#   (IsingProblem, MaxCutProblem) both support Python's pickle protocol.
#
# Libraries:
#   multiprocessing (standard library): process-based parallelism that bypasses
#   the Global Interpreter Lock (GIL), giving real CPU-level parallelism.

import multiprocessing


def _evaluate_chunk(args):
    """
    Worker function: given a list of solutions, find the best one.
    Must be defined at module level so multiprocessing can pickle it.

    Parameters
    ----------
    args : (problem, solutions, maximize)

    Returns
    -------
    (best_val, best_sol) for this chunk
    """
    problem, solutions, maximize = args
    best_val = float('-inf') if maximize else float('inf')
    best_sol = None

    for sol in solutions:
        val = problem.eval(sol)
        if val == float('-inf'):
            continue  # skip infeasible solutions
        if maximize and val > best_val:
            best_val, best_sol = val, sol
        elif not maximize and val < best_val:
            best_val, best_sol = val, sol

    return best_val, best_sol


class ExhaustiveSearch:

    def __init__(self, problem, maximize=False):
        """
        Parameters
        ----------
        problem  : any problem class that exposes generate_complete_search_space()
                   and eval(). Works with KnapsackProblem, TspProblem, IsingProblem,
                   QuboProblem, and MaxCutProblem.
        maximize : True for maximization problems (e.g. Knapsack),
                   False (default) for minimization (e.g. QUBO, TSP, Ising, MaxCut).
        """
        self.problem = problem
        self.maximize = maximize

    def solve(self):
        """
        Sequential exhaustive search.
        Uses the generator from generate_complete_search_space(), so solutions
        are produced one at a time without loading all 2^n configs into memory.

        Returns
        -------
        (best_value, best_solution)
        """
        best_val = float('-inf') if self.maximize else float('inf')
        best_sol = None

        for sol in self.problem.generate_complete_search_space():
            val = self.problem.eval(sol)
            if val == float('-inf'):
                continue  # skip infeasible solutions (Knapsack overload)
            if self.maximize and val > best_val:
                best_val, best_sol = val, sol
            elif not self.maximize and val < best_val:
                best_val, best_sol = val, sol

        return best_val, best_sol

    def solve_parallel(self, n_workers=None):
        """
        Parallel exhaustive search using multiprocessing.Pool.

        The full search space is split into equal chunks. Each worker process
        gets one chunk and returns its local best. The main process then
        combines all partial bests to find the global optimum.

        Warning: all solutions are loaded into memory at once before splitting.
        This can be large for n > 20.

        Parameters
        ----------
        n_workers : number of worker processes.
                    Defaults to the number of logical CPU cores (multiprocessing.cpu_count()).

        Returns
        -------
        (best_value, best_solution)
        """
        if n_workers is None:
            n_workers = multiprocessing.cpu_count()

        # Materialize the generator so we can slice it into chunks.
        all_solutions = list(self.problem.generate_complete_search_space())

        if not all_solutions:
            return None, None

        # Split into equal chunks, one per worker.
        chunk_size = max(1, len(all_solutions) // n_workers)
        chunks = [
            all_solutions[i: i + chunk_size]
            for i in range(0, len(all_solutions), chunk_size)
        ]

        # Pack the arguments for each worker: (problem, chunk, maximize).
        worker_args = [(self.problem, chunk, self.maximize) for chunk in chunks]

        with multiprocessing.Pool(processes=n_workers) as pool:
            results = pool.map(_evaluate_chunk, worker_args)

        # Each worker returned (best_val, best_sol). Keep the global best.
        best_val = float('-inf') if self.maximize else float('inf')
        best_sol = None

        for val, sol in results:
            if sol is None:
                continue
            if self.maximize and val > best_val:
                best_val, best_sol = val, sol
            elif not self.maximize and val < best_val:
                best_val, best_sol = val, sol

        return best_val, best_sol
