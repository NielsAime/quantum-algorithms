import numpy as np
import matplotlib.pyplot as plt


class DwaveSimulator:

    def __init__(self):
        # 101 pas de temps : s=0 (tout H_init) → s=100 (tout H_final)
        # A(s) décroît de 1 à 0, B(s) croît de 0 à 1
        self.annealing_schedule = {
            'A': [1 - s/100 for s in range(101)],
            'B': [s/100 for s in range(101)]
        }

    def build_Hfinal(self, h, J):
        # Encode le problème Ising en matrice 2^n x 2^n
        # Chaque variable s_i → sigma_z à la position i (produits tensoriels)
        linear    = h
        quadratic = J
        n = len(linear)

        sigma_z = np.array([[1, 0], [0, -1]])
        I       = np.eye(2)
        H       = np.zeros((2**n, 2**n))

        # Termes linéaires : h_i * sigma_z_i
        for i, h in linear.items():
            op = 1
            for k in range(n):
                op = np.kron(op, sigma_z if k == i else I)
            H += h * op

        # Termes quadratiques : J_ij * sigma_z_i * sigma_z_j
        for (i, j), J in quadratic.items():
            op = 1
            for k in range(n):
                op = np.kron(op, sigma_z if k == i or k == j else I)
            H += J * op

        return H

    def build_Hinit(self, n):
        # H_init = somme des sigma_x sur chaque qubit
        # Son état fondamental est un état superposé facile à préparer
        sigma_x = np.array([[0, 1], [1, 0]])
        I       = np.eye(2)
        H       = np.zeros((2**n, 2**n))

        for i in range(n):
            op = 1
            for k in range(n):
                op = np.kron(op, sigma_x if k == i else I)
            H += op

        return H

    def simulate_evolution(self, h, J, nb_eigenvalues=3):
        # À chaque pas s : H(s) = A(s)*H_init + B(s)*H_final
        # On diagonalise H(s) et on garde les nb_eigenvalues plus basses énergies
        n       = len(h)
        H_init  = self.build_Hinit(n)
        H_final = self.build_Hfinal(h, J)

        all_eigenvalues = []

        for s in range(101):
            A = self.annealing_schedule['A'][s]
            B = self.annealing_schedule['B'][s]
            H = A * H_init + B * H_final

            # eigh : diagonalisation d'une matrice symétrique réelle → valeurs triées
            eigenvalues, _ = np.linalg.eigh(H)
            all_eigenvalues.append(eigenvalues[:nb_eigenvalues])

        return np.array(all_eigenvalues)  # shape : (101, nb_eigenvalues)

    def get_ground_state(self, h, J):
        """
        Retourne l'énergie et le vecteur propre du ground state de H_final.
        Returns: (gs_energy, gs_vec)
        """
        H_final = self.build_Hfinal(h, J)
        eigenvalues, eigenvectors = np.linalg.eigh(H_final)
        return eigenvalues[0], eigenvectors[:, 0]

    def plot_eigenvalues(self, all_eigenvalues):
        """
        Trace les nb_eigenvalues plus basses énergies en fonction du pas d'anneal.
        all_eigenvalues : tableau (101, nb_eigenvalues) retourné par simulate_evolution
        """
        nb_eigenvalues = all_eigenvalues.shape[1]
        steps = range(101)

        plt.figure(figsize=(8, 5))
        for i in range(nb_eigenvalues):
            plt.plot(steps, all_eigenvalues[:, i], marker='o', markersize=2,
                     label=f'Eigen {i+1}')

        plt.xlabel('Annealing fraction')
        plt.ylabel('Energy')
        plt.title('Eigenvalues during D-Wave annealing')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_spectral_gap(self, all_eigenvalues):
        """
        Trace le gap spectral à chaque pas : différence entre la 2e et la 1re valeur propre.
        Le gap minimum g_min = min(epsilon_1(s) - epsilon_0(s)) est mis en évidence.
        """
        # gap à chaque pas de temps
        gap = all_eigenvalues[:, 1] - all_eigenvalues[:, 0]

        # position et valeur du gap minimum
        min_idx = np.argmin(gap)
        g_min   = gap[min_idx]

        steps = range(101)

        plt.figure(figsize=(8, 5))
        plt.plot(steps, gap, marker='o', markersize=2, label='Delta lowest Eigen')
        plt.axvline(x=min_idx, color='red', linestyle='--', alpha=0.5,
                    label=f'g_min = {g_min:.4f} at step {min_idx}')
        plt.scatter([min_idx], [g_min], color='red', zorder=5)

        plt.xlabel('Annealing fraction')
        plt.ylabel('Energy')
        plt.title('Spectral gap during D-Wave annealing')
        plt.legend()
        plt.tight_layout()
        plt.show()

        print(f"Gap minimum : {g_min:.4f} au pas {min_idx}")
        return g_min

# test rapide
#if __name__ == "__main__":
 #   sim = DwaveSimulator()
 #   ising = {
 #       'linear':    {0: -1.0, 1: 0.5, 2: -0.8},
 #       'quadratic': {(0,1): 0.5, (1,2): -0.4}
 #   }
 #   eigs = sim.simulate_evolution(ising, nb_eigenvalues=3)
 #   sim.plot_eigenvalues(eigs)
 #   sim.plot_spectral_gap(eigs)