# Wrapper générique pour une fonction continue à n variables.
#
# Expose l'interface commune du projet (.n, .eval) pour permettre
# l'utilisation directe avec GradientDescent sans modification du solveur.
#
# Contrairement aux problèmes combinatoires, l'espace des solutions est
# continu : les variables prennent des valeurs réelles dans [lo, hi].
# On utilise GradientDescent avec round_solution=False pour obtenir
# la solution continue optimale sans arrondi binaire final.

import numpy as np


class ContinuousFunction:

    def __init__(self, fn, bounds, n=2):
        """
        Parameters
        ----------
        fn     : callable ([x0, x1, ...]) → float
                 La fonction à minimiser (ou maximiser).
        bounds : (lo, hi) appliqué à toutes les variables,
                 ou liste de (lo_i, hi_i) une par variable.
        n      : nombre de variables (défaut 2 pour les fonctions 2D).
        """
        self.fn = fn
        self.n  = n

        if isinstance(bounds[0], (int, float)):
            # (lo, hi) unique → même borne pour toutes les variables
            self.bounds = [tuple(bounds)] * n
        else:
            if len(bounds) != n:
                raise ValueError(f"bounds doit avoir {n} entrées, reçu {len(bounds)}.")
            self.bounds = [tuple(b) for b in bounds]

    # Propriété de commodité : retourne (lo_global, hi_global) si toutes
    # les bornes sont identiques, sinon lève une erreur explicite.
    @property
    def domain(self):
        lo = min(b[0] for b in self.bounds)
        hi = max(b[1] for b in self.bounds)
        return (lo, hi)

    def eval(self, solution):
        """Évalue la fonction en un point. solution peut être list ou ndarray."""
        return float(self.fn(np.asarray(solution, dtype=float)))

    def __repr__(self):
        return f"ContinuousFunction(n={self.n}, bounds={self.bounds})"
