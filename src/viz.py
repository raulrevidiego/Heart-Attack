"""
src/viz.py
Funciones de visualización reutilizables.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd

COLOR_BAJO = "#4C72B0"   # azul  → bajo riesgo
COLOR_ALTO = "#DD4444"   # rojo  → alto riesgo
PALETTE    = {0: COLOR_BAJO, 1: COLOR_ALTO}


def setup_style() -> None:
    """Aplica el estilo global de seaborn."""
    sns.set_theme(style="whitegrid", font_scale=1.15)


def plot_class_balance(df: pd.DataFrame, target: str = "num",
                       save_path: str | None = None) -> None:
    """Gráfico de barras con el balance de clases."""
    counts = df[target].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Bajo riesgo (0)", "Alto riesgo (1)"], counts.values,
           color=[COLOR_BAJO, COLOR_ALTO], edgecolor="white", linewidth=0.8)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 1, str(v), ha="center", fontweight="bold")
    ax.set_title("Balance de clases — Riesgo de Ataque Cardíaco")
    ax.set_ylabel("Número de pacientes")
    ax.set_ylim(0, counts.max() * 1.15)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_continuous_distributions(df: pd.DataFrame,
                                  features: list[str] | None = None,
                                  target: str = "num",
                                  save_path: str | None = None) -> None:
    """KDE plots de variables continuas segmentados por clase."""
    features = features or ["age", "thalach", "oldpeak", "chol"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    for ax, feat in zip(axes, features):
        sns.kdeplot(data=df, x=feat, hue=target, palette=PALETTE,
                    fill=True, common_norm=False, alpha=0.5, linewidth=2, ax=ax)
        ax.set_title(f"Distribución de {feat} por Nivel de Riesgo")
        ax.set_ylabel("Densidad")
        ax.grid(axis="x", alpha=0.3)
    sns.move_legend(axes[0], "upper right", title="Riesgo (num)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_categorical_counts(df: pd.DataFrame,
                             vars_cat: list[str] | None = None,
                             target: str = "num",
                             save_path: str | None = None) -> None:
    """Count plots de variables categóricas por clase."""
    vars_cat = vars_cat or ["sex", "cp", "fbs", "restecg"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    for ax, var in zip(axes, vars_cat):
        sns.countplot(data=df, x=var, hue=target,
                      palette=PALETTE, ax=ax)
        ax.set_title(f"Distribución de {var} por Riesgo")
        ax.set_ylabel("Cantidad de Pacientes")
        ax.legend(title="Riesgo", labels=["Bajo", "Alto"])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame,
                              save_path: str | None = None) -> None:
    """Mapa de calor de correlaciones (triángulo inferior)."""
    corr = df.select_dtypes(include="number").corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", center=0,
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title("Mapa de correlación entre variables")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_risk_score_distribution(df: pd.DataFrame,
                                  target: str = "num",
                                  save_path: str | None = None) -> None:
    """Distribución de la puntuación de riesgo heurística por clase."""
    if "risk_score" not in df.columns:
        print("[viz] 'risk_score' no encontrado. Ejecuta build_features primero.")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x="risk_score", hue=target,
                  palette=PALETTE, ax=ax)
    ax.set_title("Puntuación de riesgo heurística (cp + exang + oldpeak) por clase")
    ax.set_xlabel("Risk Score (0 = bajo, 3 = máximo)")
    ax.set_ylabel("Pacientes")
    ax.legend(title="Riesgo", labels=["Bajo", "Alto"])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_age_group_risk(df: pd.DataFrame,
                         target: str = "num",
                         save_path: str | None = None) -> None:
    """Proporción de alto riesgo por grupo de edad."""
    if "age_group" not in df.columns:
        print("[viz] 'age_group' no encontrado. Ejecuta build_features primero.")
        return
    prop = (df.groupby("age_group", observed=True)[target]
              .mean()
              .reset_index()
              .rename(columns={target: "prop_alto_riesgo"}))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(prop["age_group"].astype(str), prop["prop_alto_riesgo"],
           color=COLOR_ALTO, edgecolor="white")
    ax.set_title("Proporción de Alto Riesgo por Grupo de Edad")
    ax.set_xlabel("Grupo de edad")
    ax.set_ylabel("Proporción alto riesgo")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
