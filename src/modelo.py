"""Funções de modelagem: separação treino/teste, modelos, avaliação e curva de overfitting.

Ficam fora do notebook para poderem ser testadas (pytest) e reutilizadas pelo
script src/treinar.py. O notebook apenas chama estas funções e discute os resultados.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEMENTE = 42  # fixa a aleatoriedade: quem rodar de novo obtém os mesmos números

COLUNAS_NUMERICAS = ["age", "balance", "day", "campaign", "pdays", "previous"]
COLUNAS_CATEGORICAS = ["job", "marital", "education", "default", "housing",
                       "loan", "contact", "month", "poutcome"]


def separar_treino_teste(df: pd.DataFrame, alvo: str = "y", tamanho_teste: float = 0.2):
    """Separa 80% treino / 20% teste.

    stratify=y mantém a mesma proporção de "sim" (~11,7%) nos dois conjuntos.
    Sem isso, por azar, o teste poderia ter bem mais ou bem menos positivos.
    """
    X = df.drop(columns=[alvo])
    y = df[alvo]
    return train_test_split(X, y, test_size=tamanho_teste, stratify=y, random_state=SEMENTE)


def criar_preprocessador() -> ColumnTransformer:
    """Numéricas: padronização (média 0, desvio 1). Categóricas: one-hot.

    O pré-processamento fica DENTRO do Pipeline, então ele "aprende" (média,
    desvio, categorias) só com os dados de treino — evita vazamento do teste.
    """
    return ColumnTransformer([
        ("num", StandardScaler(), COLUNAS_NUMERICAS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), COLUNAS_CATEGORICAS),
    ])


def criar_modelos() -> dict:
    """Modelos comparados, do mais simples ao mais complexo."""
    return {
        # Baseline: ignora as variáveis e sempre responde a classe mais comum ("não").
        "Baseline (sempre 'não')": DummyClassifier(strategy="most_frequent"),
        # Modelo linear, interpretável. class_weight="balanced" dá mais peso aos
        # poucos "sim", senão o modelo tende a imitar o baseline.
        "Regressão Logística": Pipeline([
            ("prep", criar_preprocessador()),
            ("modelo", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
        # Árvore sem limite de profundidade: incluída de propósito para MOSTRAR overfitting.
        "Árvore (sem limite)": Pipeline([
            ("prep", criar_preprocessador()),
            ("modelo", DecisionTreeClassifier(random_state=SEMENTE)),
        ]),
        # Floresta: muitas árvores com folhas de no mínimo 20 exemplos -> menos overfitting.
        "Random Forest": Pipeline([
            ("prep", criar_preprocessador()),
            ("modelo", RandomForestClassifier(n_estimators=300, min_samples_leaf=20,
                                              class_weight="balanced", n_jobs=-1,
                                              random_state=SEMENTE)),
        ]),
    }


def calcular_metricas(modelo, X, y) -> dict:
    """Métricas de classificação. Com classes desbalanceadas, acurácia sozinha engana."""
    previsto = modelo.predict(X)
    probabilidade = modelo.predict_proba(X)[:, 1]  # chance estimada de "sim"
    return {
        "acuracia": accuracy_score(y, previsto),
        "acuracia_balanceada": balanced_accuracy_score(y, previsto),
        "precisao": precision_score(y, previsto, zero_division=0),
        "recall": recall_score(y, previsto, zero_division=0),
        "f1": f1_score(y, previsto, zero_division=0),
        "roc_auc": roc_auc_score(y, probabilidade),
    }


def treinar_e_avaliar(modelos: dict, X_treino, X_teste, y_treino, y_teste) -> pd.DataFrame:
    """Treina cada modelo e mede no treino E no teste (a diferença indica overfitting)."""
    linhas = []
    for nome, modelo in modelos.items():
        modelo.fit(X_treino, y_treino)
        for conjunto, X, y in [("treino", X_treino, y_treino), ("teste", X_teste, y_teste)]:
            linhas.append({"modelo": nome, "conjunto": conjunto, **calcular_metricas(modelo, X, y)})
    return pd.DataFrame(linhas)


def validacao_cruzada(modelos: dict, X_treino, y_treino, folds: int = 5) -> pd.DataFrame:
    """ROC AUC em validação cruzada estratificada (só com dados de TREINO).

    Divide o treino em 5 partes; treina em 4 e avalia na 5ª, girando 5 vezes.
    O desvio padrão mostra se o desempenho é estável ou depende da sorte da divisão.
    """
    divisao = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEMENTE)
    linhas = []
    for nome, modelo in modelos.items():
        notas = cross_val_score(modelo, X_treino, y_treino, cv=divisao, scoring="roc_auc")
        linhas.append({"modelo": nome, "auc_media": notas.mean(), "auc_desvio": notas.std()})
    return pd.DataFrame(linhas)


def curva_profundidade(X_treino, X_teste, y_treino, y_teste, profundidades=range(1, 21)) -> pd.DataFrame:
    """ROC AUC de uma árvore de decisão para cada profundidade máxima.

    Mostra o padrão clássico: treino só melhora com mais profundidade, mas o
    teste melhora até um ponto e depois piora -> o modelo passou a decorar o treino.
    """
    linhas = []
    for profundidade in profundidades:
        arvore = Pipeline([
            ("prep", criar_preprocessador()),
            ("modelo", DecisionTreeClassifier(max_depth=profundidade, random_state=SEMENTE)),
        ]).fit(X_treino, y_treino)
        linhas.append({
            "profundidade": profundidade,
            "auc_treino": roc_auc_score(y_treino, arvore.predict_proba(X_treino)[:, 1]),
            "auc_teste": roc_auc_score(y_teste, arvore.predict_proba(X_teste)[:, 1]),
        })
    return pd.DataFrame(linhas)


def importancia_variaveis(pipeline_rf: Pipeline, top: int = 10) -> pd.Series:
    """Importância das variáveis da Random Forest (após o one-hot)."""
    nomes = pipeline_rf.named_steps["prep"].get_feature_names_out()
    importancias = pipeline_rf.named_steps["modelo"].feature_importances_
    serie = pd.Series(importancias, index=[n.split("__", 1)[1] for n in nomes])
    return serie.sort_values(ascending=False).head(top)
