# Algorithme Branch and Bound pour le problème du Sac à dos 0/1.
#
# Ce solveur est spécifique à KnapsackProblem : il exploite la structure du problème
# (poids, valeurs, capacité) pour calculer une borne supérieure par relaxation
# fractionnaire, ce qui n'est pas généralisable à d'autres types de problèmes.
#
# On explore un arbre binaire de décisions : pour chaque objet, on choisit
# de le prendre (1) ou non (0). Avant d'explorer une branche, on calcule
# une borne supérieure (upper bound) : le meilleur score qu'on pourrait espérer
# dans cette branche. Si ce score ne peut pas battre la meilleure solution déjà
# trouvée, on coupe la branche (pruning). La borne est calculée par relaxation
# fractionnaire : on remplit le sac goulûment en autorisant de prendre une fraction
# du dernier objet. On explore toujours le noeud le plus prometteur en premier
# (Best-First Search via une file de priorité).

import heapq


class Node:

    def __init__(self, level, value, weight, upper_bound, decisions):
        self.level = level
        self.value = value
        self.weight = weight
        self.upper_bound = upper_bound
        self.decisions = decisions

    def __lt__(self, other):
        return self.upper_bound > other.upper_bound


class BranchAndBound:

    def __init__(self, problem):
        self.problem = problem
        self.sorted_indices = sorted(
            range(problem.n),
            key=lambda i: problem.values[i] / problem.weights[i],
            reverse=True
        )
        self.sorted_weights = [problem.weights[i] for i in self.sorted_indices]
        self.sorted_values  = [problem.values[i]  for i in self.sorted_indices]


    def compute_upper_bound(self, level, current_value, current_weight):
        if current_weight > self.problem.capacity:
            return 0
        borne = current_value
        restant = self.problem.capacity - current_weight
        for i in range(level, self.problem.n):
            if self.sorted_weights[i] <= restant:
                restant -= self.sorted_weights[i]
                borne   += self.sorted_values[i]
            else:
                borne += (restant / self.sorted_weights[i]) * self.sorted_values[i]

                break
            if self.sorted_weights[i] <= remaining:
                bound += self.sorted_values[i]
                remaining -= self.sorted_weights[i]
            else:
                bound += self.sorted_values[i] * (remaining / self.sorted_weights[i])
                break

        return bound

    def solve(self):
        meilleure_valeur = 0
        meilleures_decisions = []

        racine = Node(
            level=0,
            value=0,
            weight=0,
            upper_bound=self.compute_upper_bound(0, 0, 0),
            decisions=[]
        )

        file = [(-racine.upper_bound, racine)]

        while file:
            _, noeud = heapq.heappop(file)

            if noeud.upper_bound <= meilleure_valeur:
                continue

            level = noeud.level

            if level == self.problem.n:
                if noeud.value > meilleure_valeur:
                    meilleure_valeur     = noeud.value
                    meilleures_decisions = noeud.decisions
                continue

            nouveau_poids   = noeud.weight + self.sorted_weights[level]
            nouvelle_valeur = noeud.value  + self.sorted_values[level]

            if nouveau_poids <= self.problem.capacity:
                borne_inclus = self.compute_upper_bound(level + 1, nouvelle_valeur, nouveau_poids)

                if borne_inclus > meilleure_valeur:
                    enfant_inclus = Node(
                        level=level + 1,
                        value=nouvelle_valeur,
                        weight=nouveau_poids,
                        upper_bound=borne_inclus,
                        decisions=noeud.decisions + [1]
                    )

                    if nouvelle_valeur > meilleure_valeur:
                        meilleure_valeur     = nouvelle_valeur
                        meilleures_decisions = enfant_inclus.decisions

                    heapq.heappush(file, (-borne_inclus, enfant_inclus))

            borne_exclus = self.compute_upper_bound(level + 1, noeud.value, noeud.weight)

            if borne_exclus > meilleure_valeur:
                enfant_exclus = Node(
                    level=level + 1,
                    value=noeud.value,
                    weight=noeud.weight,
                    upper_bound=borne_exclus,
                    decisions=noeud.decisions + [0]
                )
                heapq.heappush(file, (-borne_exclus, enfant_exclus))

        return meilleure_valeur, self._decoder_solution(meilleures_decisions)

    def _decoder_solution(self, decisions_triees):

        inclusion = [False] * self.problem.n

        for idx_trie, decision in enumerate(decisions_triees):
            if decision == 1:
                inclusion[self.sorted_indices[idx_trie]] = True

        return inclusion