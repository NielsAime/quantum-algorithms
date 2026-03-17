import numpy as np
from scipy.optimize import minimize
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit
from solver.quantum_solver.qaoa_solver.qaoa_mixers import add_ising_problem_ham, add_ising_mixer_ham


class QAOALocalOptimizer:

    def __init__(self, simulator, gamma_bounds, beta_bounds, p, shots, opt_method='COBYLA'):
        self.simulator    = simulator
        self.gamma_bounds = gamma_bounds
        self.beta_bounds  = beta_bounds
        self.p            = p
        self.shots        = shots
        self.opt_method   = opt_method

    def _build_circuit(self, problem, p):
        n = len(problem['linear'])
        qc = QuantumCircuit(n, n)
        for i in range(n):
            qc.h(i)
        all_params = []
        for _ in range(p):
            qc, gammas = add_ising_problem_ham(qc, problem, n)
            qc, betas  = add_ising_mixer_ham(qc, problem, n)
            all_params += gammas + betas
        qc.measure(range(n), range(n))
        return qc, all_params

    def _run_circuit(self, angles, qc):
        param_dict = dict(zip(qc.parameters, angles))
        bound_qc = qc.assign_parameters(param_dict)
        result = self.simulator.run(bound_qc, shots=self.shots).result()
        return result.get_counts()

    def get_expectation_value(self, angles, qc, problem):
        counts = self._run_circuit(angles, qc)
        total_shots = sum(counts.values())
        expectation = 0
        for bitstring, count in counts.items():
            spins = [1 if b == '1' else -1 for b in bitstring]
            cost = 0
            for (i, j), J in problem['quadratic'].items():
                cost += J * spins[i] * spins[j]
            for i, h in problem['linear'].items():
                cost += h * spins[i]
            expectation += cost * count / total_shots
        return expectation

    def run_without_optimization(self, problem, p, angles):
        qc, _ = self._build_circuit(problem, p)
        exp_val = self.get_expectation_value(angles, qc, problem)
        counts = self._run_circuit(angles, qc)
        best_bitstring = max(counts, key=counts.get)
        best_solution = [1 if b == '1' else -1 for b in best_bitstring]
        return exp_val, best_solution, angles

    def optimize(self, problem, p):
        qc, _ = self._build_circuit(problem, p)
        best_solution = [None]
        best_counts   = [None]

        def objective(angles):
            # Pénalité si les angles sortent des bornes
            for k in range(p):
                if not (self.gamma_bounds[0] <= angles[2*k]   <= self.gamma_bounds[1]):
                    return 1e6
                if not (self.beta_bounds[0]  <= angles[2*k+1] <= self.beta_bounds[1]):
                    return 1e6
            counts = self._run_circuit(angles, qc)
            best_counts[0] = counts
            total = sum(counts.values())
            exp = 0
            for bitstring, count in counts.items():
                spins = [1 if b == '1' else -1 for b in bitstring]
                cost = 0
                for (i, j), J in problem['quadratic'].items():
                    cost += J * spins[i] * spins[j]
                for i, h in problem['linear'].items():
                    cost += h * spins[i]
                exp += cost * count / total
            best_bs = max(counts, key=counts.get)
            best_solution[0] = [1 if b == '1' else -1 for b in best_bs]
            return exp

        x0 = np.random.uniform(0, np.pi, 2 * p)
        result = minimize(objective, x0, method=self.opt_method)
        return result.fun, best_solution[0], result.x