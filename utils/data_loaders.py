"""
Fonctions pour sauvegarder et charger des instances de problèmes.
Les matrices QUBO sont stockées au format .npy (NumPy).
"""

import os
import numpy as np
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from problem.qubo import QuboProblem


def save_qubo(problem, filepath):
    """Sauvegarde une instance QuboProblem dans un fichier .npy."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.save(filepath, problem.Q)


def load_qubo(filepath):
    """Charge une instance QuboProblem depuis un fichier .npy."""
    Q = np.load(filepath)
    return QuboProblem(Q)


def save_qubo_pool(problems, directory, prefix="instance"):
    """Sauvegarde une liste d'instances QuboProblem dans un répertoire."""
    os.makedirs(directory, exist_ok=True)
    for i, prob in enumerate(problems):
        path = os.path.join(directory, f"{prefix}_{i}.npy")
        np.save(path, prob.Q)


def load_qubo_pool(directory, prefix="instance"):
    """Charge toutes les instances sauvegardées dans un répertoire."""
    problems = []
    i = 0
    while True:
        path = os.path.join(directory, f"{prefix}_{i}.npy")
        if not os.path.exists(path):
            break
        problems.append(load_qubo(path))
        i += 1
    return problems
