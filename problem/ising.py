# Problème d'Ising.
# Minimiser H = Σ_{i<j} J_ij s_i s_j + Σ_i h_i s_i  avec  s_i ∈ {-1, +1}.
#
# Accepte deux formats d'entrée :
#   - Un graphe NetworkX : arêtes avec attribut 'weight' (= J_ij),
#     noeuds avec attribut optionnel 'h' (= champ local h_i).
#   - Une chaîne d'expression sympy : "2*s_0*s_1 - s_0 + s_1"

import itertools
import networkx as nx
from sympy import symbols, sympify, expand


class IsingProblem:

    def __init__(self, input_data):
        self._sympy_expr = None
        self._input = input_data

        if isinstance(input_data, str):
            self._sympy_expr = sympify(input_data)
        elif not isinstance(input_data, nx.Graph):
            raise ValueError("L'entrée doit être une chaîne sympy ou un graphe NetworkX.")

    def __repr__(self):
        return f"IsingProblem(expr={self.to_sympy_expr()})"

    def to_sympy_expr(self):
        if self._sympy_expr is not None:
            return self._sympy_expr

        graph = self._input
        node_syms = {node: symbols(f"s_{node}") for node in sorted(graph.nodes())}

        expr = sympify(0)
        for u, v, data in graph.edges(data=True):
            expr += data.get("weight", 1) * node_syms[u] * node_syms[v]
        for node, data in graph.nodes(data=True):
            h = data.get("h", 0)
            if h != 0:
                expr += h * node_syms[node]

        self._sympy_expr = expr
        return self._sympy_expr

    def eval(self, solution):
        expr = self.to_sympy_expr()
        subs = {}
        for sym in expr.free_symbols:
            parts = str(sym).split('_')
            try:
                idx = int(parts[-1])
                if idx in solution:
                    subs[sym] = solution[idx]
            except (ValueError, IndexError):
                pass
        return float(expr.subs(subs))

    def generate_complete_search_space(self):
        expr = self.to_sympy_expr()
        syms = sorted(expr.free_symbols, key=lambda s: int(str(s).split('_')[-1]))
        indices = [int(str(s).split('_')[-1]) for s in syms]
        for combo in itertools.product([-1, 1], repeat=len(indices)):
            yield {idx: val for idx, val in zip(indices, combo)}
