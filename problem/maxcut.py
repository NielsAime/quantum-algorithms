# Problème MaxCut.
# Maximiser le nombre (ou poids) d'arêtes coupées dans un graphe.
# Chaque nœud est assigné à un groupe : s_i ∈ {-1, +1}.
# Fonction de coût : -Σ_{(i,j) ∈ E} J_ij s_i s_j  (minimiser = maximiser la coupe).

import itertools
import matplotlib.pyplot as plt
import networkx as nx
from sympy import symbols, sympify


class MaxCutProblem:

    def __init__(self, graph):
        self.graph = graph
        self._sympy_expr = None

    def __repr__(self):
        return f"MaxCutProblem(nodes={list(self.graph.nodes())}, edges={list(self.graph.edges())})"

    def to_sympy_expr(self):
        if self._sympy_expr is not None:
            return self._sympy_expr

        node_syms = {node: symbols(f"s_{node}") for node in self.graph.nodes()}
        expr = sympify(0)
        for u, v, data in self.graph.edges(data=True):
            expr -= data.get("weight", 1) * node_syms[u] * node_syms[v]

        self._sympy_expr = expr
        return self._sympy_expr

    def eval(self, solution):
        expr = self.to_sympy_expr()
        node_syms = {node: symbols(f"s_{node}") for node in self.graph.nodes()}
        subs = {node_syms[node]: val for node, val in solution.items()}
        return float(expr.subs(subs))

    def generate_complete_search_space(self):
        nodes = list(self.graph.nodes())
        for combo in itertools.product([-1, 1], repeat=len(nodes)):
            yield {node: val for node, val in zip(nodes, combo)}

    def display_graph(self):
        pos = nx.spring_layout(self.graph, seed=42)
        nx.draw(self.graph, pos, with_labels=True,
                node_color="lightblue", node_size=700, font_size=12)
        edge_labels = nx.get_edge_attributes(self.graph, "weight")
        if edge_labels:
            nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
        plt.title("MaxCut Graph")
        plt.axis("off")
        plt.show()

    def display_solution(self, solution):
        pos = nx.spring_layout(self.graph, seed=42)
        color_map = ["red" if solution[n] == 1 else "blue" for n in self.graph.nodes()]
        nx.draw(self.graph, pos, with_labels=True,
                node_color=color_map, node_size=700, font_size=12)
        edge_labels = nx.get_edge_attributes(self.graph, "weight")
        if edge_labels:
            nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
        plt.title(f"MaxCut Solution (coût : {self.eval(solution):.2f})")
        plt.axis("off")
        plt.show()
