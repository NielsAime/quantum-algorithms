import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from qiskit import QuantumCircuit
from solver.quantum_solver.qaoa_solver.qaoa_mixers import add_ising_mixer_ham, add_ising_problem_ham

ising_problem = {
    'linear':    {0: -1.0, 1: 0.5, 2: -0.8},
    'quadratic': {(0, 1): 0.5, (1, 2): -0.4}
}

class TestQAOAMixers(unittest.TestCase):

    def test_mixer_retourne_circuit(self):
        qc = QuantumCircuit(3)
        qc_out, betas = add_ising_mixer_ham(qc, ising_problem, n=3)
        # Le circuit doit avoir 3 portes Rx (une par qubit)
        self.assertIsInstance(qc_out, QuantumCircuit)
        self.assertEqual(len(betas), 1)

    def test_mixer_nb_portes(self):
        qc = QuantumCircuit(3)
        qc_out, _ = add_ising_mixer_ham(qc, ising_problem, n=3)
        ops = [inst.operation.name for inst in qc_out.data]
        self.assertEqual(ops.count('rx'), 3)

    def test_problem_ham_retourne_circuit(self):
        qc = QuantumCircuit(3)
        qc_out, gammas = add_ising_problem_ham(qc, ising_problem, n=3)
        self.assertIsInstance(qc_out, QuantumCircuit)
        self.assertEqual(len(gammas), 1)

    def test_problem_ham_nb_portes(self):
        qc = QuantumCircuit(3)
        qc_out, _ = add_ising_problem_ham(qc, ising_problem, n=3)
        ops = [inst.operation.name for inst in qc_out.data]
        # 3 termes linéaires → 3 Rz, 2 termes quadratiques → 2 Rz + 4 cx
        self.assertEqual(ops.count('rz'), 5)
        self.assertEqual(ops.count('cx'), 4)

if __name__ == '__main__':
    unittest.main()