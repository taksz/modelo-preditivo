"""Executa todo o experimento sem abrir o Jupyter e salva os resultados.

Uso (a partir da raiz do projeto):
    python -m src.treinar

Saídas:
    resultados/metricas.csv         métricas de treino e teste de cada modelo
    resultados/validacao_cruzada.csv
    resultados/curva_profundidade.csv
    imgs/*.png                       gráficos usados no README
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # gera imagens sem precisar de janela (funciona em servidor/terminal)

import pandas as pd  # noqa: E402

from src import graficos  # noqa: E402
from src import modelo as m  # noqa: E402
from src.dados import carregar_dados  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
PASTA_RESULTADOS = RAIZ / "resultados"
PASTA_IMGS = RAIZ / "imgs"


def main() -> None:
    PASTA_RESULTADOS.mkdir(exist_ok=True)
    PASTA_IMGS.mkdir(exist_ok=True)
    pd.set_option("display.width", 140)

    df = carregar_dados()
    print(f"Base: {len(df)} clientes, {df['y'].mean():.1%} contrataram (classe positiva)")

    X_treino, X_teste, y_treino, y_teste = m.separar_treino_teste(df)
    print(f"Treino: {len(X_treino)} | Teste: {len(X_teste)}")

    modelos = m.criar_modelos()
    resultados = m.treinar_e_avaliar(modelos, X_treino, X_teste, y_treino, y_teste)
    resultados.round(4).to_csv(PASTA_RESULTADOS / "metricas.csv", index=False)
    print("\nMétricas (treino x teste):")
    print(resultados.round(3).to_string(index=False))

    cv = m.validacao_cruzada({k: v for k, v in modelos.items() if "sem limite" not in k},
                             X_treino, y_treino)
    cv.round(4).to_csv(PASTA_RESULTADOS / "validacao_cruzada.csv", index=False)
    print("\nValidação cruzada (5 folds, só treino):")
    print(cv.round(3).to_string(index=False))

    curva = m.curva_profundidade(X_treino, X_teste, y_treino, y_teste)
    curva.round(4).to_csv(PASTA_RESULTADOS / "curva_profundidade.csv", index=False)

    floresta = modelos["Random Forest"]
    graficos.grafico_comparacao(resultados).savefig(PASTA_IMGS / "comparacao_modelos.png", dpi=120)
    graficos.grafico_curva_profundidade(curva).savefig(PASTA_IMGS / "curva_overfitting.png", dpi=120)
    graficos.grafico_matriz_confusao(floresta, X_teste, y_teste, "Random Forest - teste").savefig(
        PASTA_IMGS / "matriz_confusao_rf.png", dpi=120)
    graficos.grafico_importancia(m.importancia_variaveis(floresta)).savefig(
        PASTA_IMGS / "importancia_variaveis.png", dpi=120)

    print(f"\nResultados em {PASTA_RESULTADOS} | gráficos em {PASTA_IMGS}")


if __name__ == "__main__":
    main()
