# Problème du Sac à dos (Knapsack 0/1).
# Maximiser la valeur totale des objets sélectionnés sans dépasser la capacité.
# Chaque objet est pris (1) ou non (0).

import itertools

import random


class KnapsackProblem:

    def __init__(self, weights, values, capacity):
        if len(weights) != len(values):
            raise ValueError("weights et values doivent avoir la même longueur.")
        self.weights = list(weights)
        self.values  = list(values)
        self.capacity = capacity
        self.n = len(weights)

    def __repr__(self):
        return f"KnapsackProblem(n={self.n}, capacity={self.capacity})"

    def is_feasible(self, solution):
        if isinstance(solution, dict):
            x = [solution[i] for i in range(self.n)]
        else:
            x = list(solution)
        return sum(x[i] * self.weights[i] for i in range(self.n)) <= self.capacity

    def eval(self, solution):
        if not self.is_feasible(solution):
            return float('-inf')
        if isinstance(solution, dict):
            x = [solution[i] for i in range(self.n)]
        else:
            x = list(solution)
        return sum(x[i] * self.values[i] for i in range(self.n))

    def generate_complete_search_space(self):
        for combo in itertools.product([0, 1], repeat=self.n):
            yield list(combo)

    @staticmethod
    def generate_random(n_items, capacity=None, weight_range=(1, 20), value_range=(1, 50), seed=None):
        if seed is not None:
            random.seed(seed)

        weights = [random.randint(*weight_range) for _ in range(n_items)]
        values = [random.randint(*value_range) for _ in range(n_items)]

        if capacity is None:
            capacity = sum(weights) // 2
        return KnapsackProblem(weights, values, capacity)

