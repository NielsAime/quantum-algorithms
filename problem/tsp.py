# Problème du Voyageur de Commerce (TSP).
# Trouver la tournée la plus courte visitant toutes les villes exactement une fois.
# La solution est une liste ordonnée de noms de villes (la boucle se referme automatiquement).

import itertools
import numpy as np
import matplotlib.pyplot as plt


class TspProblem:

    def __init__(self, cities: dict, distances: dict = None):
        """
        Parameters
        ----------
        cities    : {nom_ville: (x, y)}
        distances : {(ville_i, ville_j): distance}  — calculées si non fournies
        """
        self.cities     = cities
        self.city_names = list(cities.keys())

        if distances is not None:
            self.distances = distances
        else:
            self.distances = {}
            for a, b in itertools.combinations(self.city_names, 2):
                xa, ya = cities[a]
                xb, yb = cities[b]
                d = np.sqrt((xa - xb) ** 2 + (ya - yb) ** 2)
                self.distances[(a, b)] = d
                self.distances[(b, a)] = d

    def __repr__(self):
        return f"TspProblem(n_cities={len(self.city_names)})"

    def eval(self, solution: list) -> float:
        total = 0.0
        n = len(solution)
        for i in range(n):
            total += self.distances[(solution[i], solution[(i + 1) % n])]
        return total

    def generate_complete_search_space(self):
        fixed  = self.city_names[0]
        others = self.city_names[1:]
        for perm in itertools.permutations(others):
            yield [fixed] + list(perm)

    def display_solution(self, solution: list):
        total_dist = self.eval(solution)
        route = solution + [solution[0]]

        xs = [self.cities[c][0] for c in route]
        ys = [self.cities[c][1] for c in route]

        plt.figure(figsize=(7, 5))
        plt.plot(xs, ys, '-o', color='steelblue', linewidth=2, markersize=8)
        for city, (x, y) in self.cities.items():
            plt.annotate(city, (x, y), textcoords="offset points",
                         xytext=(8, 8), fontsize=12, fontweight='bold')
        plt.scatter(*self.cities[solution[0]], color='red', zorder=5, s=120, label='Départ')
        plt.title(f"Solution TSP — Distance totale : {total_dist:.2f}", fontsize=13)
        plt.legend()
        plt.axis('equal')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()
        print(f"Route    : {' → '.join(route)}")
        print(f"Distance : {total_dist:.4f}")
