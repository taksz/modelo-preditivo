"""Gráficos usados no notebook e no script de treino (matplotlib puro)."""

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def grafico_comparacao(resultados, metrica: str = "roc_auc"):
    """Barras lado a lado: métrica no treino x no teste para cada modelo."""
    tabela = resultados.pivot(index="modelo", columns="conjunto", values=metrica)
    tabela = tabela.loc[resultados["modelo"].unique(), ["treino", "teste"]]  # mantém a ordem original
    ax = tabela.plot.barh(figsize=(8, 4), color=["#9bb7d4", "#1f4e79"])
    ax.set_xlabel(metrica.upper().replace("_", " "))
    ax.set_ylabel("")
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()
    ax.set_title(f"{metrica.upper().replace('_', ' ')} no treino e no teste")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=2, fontsize=8)
    plt.tight_layout()
    return ax.figure


def grafico_curva_profundidade(curva):
    """AUC de treino e teste por profundidade da árvore (diagnóstico de overfitting)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(curva["profundidade"], curva["auc_treino"], marker="o", label="treino")
    ax.plot(curva["profundidade"], curva["auc_teste"], marker="o", label="teste")
    melhor = curva.loc[curva["auc_teste"].idxmax()]
    ax.axvline(melhor["profundidade"], color="gray", linestyle="--",
               label=f"melhor no teste (profundidade {int(melhor['profundidade'])})")
    ax.set_xticks(curva["profundidade"])  # só números inteiros no eixo x
    ax.set_xlabel("profundidade máxima da árvore")
    ax.set_ylabel("ROC AUC")
    ax.set_title("Overfitting: o treino sempre melhora, o teste não")
    ax.legend()
    plt.tight_layout()
    return fig


def grafico_matriz_confusao(modelo, X_teste, y_teste, titulo: str):
    """Matriz de confusão no conjunto de teste."""
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(modelo, X_teste, y_teste, display_labels=["não", "sim"],
                                          cmap="Blues", values_format="d", ax=ax, colorbar=False)
    ax.set_title(titulo)
    plt.tight_layout()
    return fig


def grafico_importancia(importancias):
    """Barras horizontais com as variáveis mais importantes."""
    fig, ax = plt.subplots(figsize=(7, 4))
    importancias.iloc[::-1].plot.barh(ax=ax, color="#1f4e79")
    ax.set_xlabel("importância (Random Forest)")
    ax.set_title("Variáveis mais importantes")
    plt.tight_layout()
    return fig
