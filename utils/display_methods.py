"""
Fonctions graphiques réutilisables pour les notebooks d'analyse.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_convergence_runs(runs_values, title="Convergence du SA", ylabel="Valeur"):
    """
    Affiche la distribution des valeurs obtenues sur plusieurs runs indépendants.

    Parameters
    ----------
    runs_values : list of float  — valeur finale de chaque run
    title, ylabel : labels
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(runs_values, bins=15, color="steelblue", edgecolor="white")
    axes[0].axvline(np.mean(runs_values), color="red", linestyle="--", label=f"Moyenne = {np.mean(runs_values):.2f}")
    axes[0].set_title(f"{title} — Distribution")
    axes[0].set_xlabel(ylabel)
    axes[0].set_ylabel("Fréquence")
    axes[0].legend()

    axes[1].plot(runs_values, "o", alpha=0.6, color="steelblue")
    axes[1].axhline(np.mean(runs_values), color="red", linestyle="--", label="Moyenne")
    axes[1].axhline(np.min(runs_values), color="green", linestyle=":", label=f"Min = {np.min(runs_values):.2f}")
    axes[1].set_title(f"{title} — Valeurs par run")
    axes[1].set_xlabel("Run #")
    axes[1].set_ylabel(ylabel)
    axes[1].legend()

    plt.tight_layout()
    return fig


def plot_parameter_sweep(param_values, means, stds, param_name, title="Impact du paramètre", ylabel="Valeur"):
    """
    Courbe moyenne ± écart-type pour un sweep de paramètre.

    Parameters
    ----------
    param_values : valeurs du paramètre testé (axe x)
    means, stds  : moyennes et écarts-types des résultats
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(param_values, means, "o-", color="steelblue", label="Moyenne")
    ax.fill_between(param_values,
                    np.array(means) - np.array(stds),
                    np.array(means) + np.array(stds),
                    alpha=0.2, color="steelblue", label="±1 std")
    ax.set_xlabel(param_name)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_boxplot_comparison(data_dict, title="Comparaison", xlabel="Condition", ylabel="Valeur"):
    """
    Boîte à moustaches pour comparer plusieurs conditions.

    Parameters
    ----------
    data_dict : dict {label: [values]}  — une entrée par condition
    """
    fig, ax = plt.subplots(figsize=(max(6, len(data_dict) * 1.5), 5))
    labels = list(data_dict.keys())
    values = [data_dict[k] for k in labels]
    bp = ax.boxplot(values, patch_artist=True)
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(labels)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_time_vs_quality(times, qualities, labels=None, title="Temps vs Qualité"):
    """
    Scatter plot temps de calcul vs qualité de la solution.

    Parameters
    ----------
    times, qualities : listes de valeurs (une par configuration)
    labels           : liste de strings pour annoter les points
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(times, qualities, color="steelblue", zorder=5)
    if labels:
        for x, y, lbl in zip(times, qualities, labels):
            ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Qualité (valeur QUBO)")
    ax.set_title(title)
    plt.tight_layout()
    return fig
