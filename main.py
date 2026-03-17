from problem.knapsack import KnapsackProblem
from solver.classical_solver.branch_and_bound import BranchAndBound

# Un exemple concret
problem = KnapsackProblem(
    weights=[2, 3, 4, 5],
    values= [3, 4, 5, 6],
    capacity=8
)

bb = BranchAndBound(problem)
valeur, items = bb.solve()

print(f"Valeur optimale : {valeur}")
print(f"Objets pris     : {items}")
print(f"Poids total     : {sum(w for w, inc in zip(problem.weights, items) if inc)}")