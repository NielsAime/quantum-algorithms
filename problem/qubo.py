# Problème QUBO (Quadratic Unconstrained Binary Optimization).
# Minimiser f(x) = x^T Q x  avec  x ∈ {0,1}^n.
# Q est une matrice triangulaire supérieure (ou symétrique, convertie en interne).

import itertools
import numpy as np
from sympy import symbols, sympify


class QuboProblem:

    def __init__(self, matrix):
        Q = np.array(matrix, dtype=float)

        is_upper_tri = np.allclose(Q, np.triu(Q))
        is_sym       = np.allclose(Q, Q.T)

        if not is_upper_tri and not is_sym:
            raise ValueError("Q doit être triangulaire supérieure ou symétrique.")

        if is_sym and not is_upper_tri:
            Q = np.triu(Q)

        self.Q = Q
        self.n = Q.shape[0]
        self._sympy_expr = None

    def __repr__(self):
        return f"QuboProblem(n={self.n})"

    def to_sympy_expr(self):
        if self._sympy_expr is not None:
            return self._sympy_expr
        x = [symbols(f"x_{i}") for i in range(self.n)]
        expr = sympify(0)
        for i in range(self.n):
            for j in range(i, self.n):
                if self.Q[i, j] != 0:
                    expr += self.Q[i, j] * x[i] * x[j]
        self._sympy_expr = expr
        return self._sympy_expr

    def eval(self, solution):
        if isinstance(solution, dict):
            x = np.array([solution[i] for i in range(self.n)], dtype=float)
        else:
            x = np.array(solution, dtype=float)
        return float(x @ self.Q @ x)

    def generate_complete_search_space(self):
        for combo in itertools.product([0, 1], repeat=self.n):
            yield {i: v for i, v in enumerate(combo)}
