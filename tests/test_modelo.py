"""Testes rápidos e sem internet, usando uma base sintética com as mesmas colunas da UCI."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import modelo as m  # noqa: E402
from src.dados import carregar_dados  # noqa: E402


@pytest.fixture
def base_falsa() -> pd.DataFrame:
    """600 clientes inventados; quem teve sucesso antes ('success') tende a contratar."""
    rng = np.random.default_rng(0)
    n = 600
    poutcome = rng.choice(["success", "failure", "unknown"], n, p=[0.1, 0.2, 0.7])
    df = pd.DataFrame({
        "age": rng.integers(18, 90, n),
        "balance": rng.normal(1500, 3000, n).round(),
        "day": rng.integers(1, 32, n),
        "campaign": rng.integers(1, 10, n),
        "pdays": rng.integers(-1, 400, n),
        "previous": rng.integers(0, 5, n),
        "job": rng.choice(["admin.", "technician", "student"], n),
        "marital": rng.choice(["married", "single"], n),
        "education": rng.choice(["primary", "secondary", "tertiary"], n),
        "default": rng.choice(["yes", "no"], n),
        "housing": rng.choice(["yes", "no"], n),
        "loan": rng.choice(["yes", "no"], n),
        "contact": rng.choice(["cellular", "unknown"], n),
        "month": rng.choice(["may", "jun", "oct"], n),
        "poutcome": poutcome,
    })
    chance = np.where(poutcome == "success", 0.7, 0.08)
    df["y"] = (rng.random(n) < chance).astype(int)
    return df


def test_separacao_estratificada_mantem_proporcao(base_falsa):
    X_treino, X_teste, y_treino, y_teste = m.separar_treino_teste(base_falsa)
    assert len(X_teste) == 120  # 20% de 600
    assert "y" not in X_treino.columns
    assert abs(y_treino.mean() - y_teste.mean()) < 0.02


def test_baseline_sempre_preve_classe_majoritaria(base_falsa):
    X_treino, X_teste, y_treino, y_teste = m.separar_treino_teste(base_falsa)
    baseline = m.criar_modelos()["Baseline (sempre 'não')"].fit(X_treino, y_treino)
    assert set(baseline.predict(X_teste)) == {0}


def test_modelos_superam_baseline_em_auc(base_falsa):
    X_treino, X_teste, y_treino, y_teste = m.separar_treino_teste(base_falsa)
    resultados = m.treinar_e_avaliar(m.criar_modelos(), X_treino, X_teste, y_treino, y_teste)
    teste = resultados[resultados["conjunto"] == "teste"].set_index("modelo")

    assert teste.loc["Baseline (sempre 'não')", "roc_auc"] == 0.5
    assert teste.loc["Regressão Logística", "roc_auc"] > 0.6
    assert set(resultados.columns) >= {"acuracia", "precisao", "recall", "f1", "roc_auc"}


def test_arvore_sem_limite_decora_o_treino(base_falsa):
    X_treino, X_teste, y_treino, y_teste = m.separar_treino_teste(base_falsa)
    resultados = m.treinar_e_avaliar({"arvore": m.criar_modelos()["Árvore (sem limite)"]},
                                     X_treino, X_teste, y_treino, y_teste)
    auc = resultados.set_index("conjunto")["roc_auc"]
    assert auc["treino"] == 1.0
    assert auc["teste"] < auc["treino"]


def test_categoria_nova_no_teste_nao_quebra(base_falsa):
    X_treino, X_teste, y_treino, _ = m.separar_treino_teste(base_falsa)
    modelo = m.criar_modelos()["Regressão Logística"].fit(X_treino, y_treino)
    X_novo = X_teste.head(1).copy()
    X_novo["job"] = "astronauta"  # categoria nunca vista: handle_unknown="ignore"
    assert modelo.predict(X_novo).shape == (1,)


def test_curva_profundidade_tem_colunas_esperadas(base_falsa):
    X_treino, X_teste, y_treino, y_teste = m.separar_treino_teste(base_falsa)
    curva = m.curva_profundidade(X_treino, X_teste, y_treino, y_teste, profundidades=[1, 3])
    assert list(curva.columns) == ["profundidade", "auc_treino", "auc_teste"]
    assert len(curva) == 2


def test_carregar_dados_remove_duration_e_converte_alvo(tmp_path):
    csv = tmp_path / "mini.csv"
    csv.write_text("age;duration;y\n30;100;yes\n40;5;no\n", encoding="utf-8")
    df = carregar_dados(csv)
    assert "duration" not in df.columns
    assert df["y"].tolist() == [1, 0]
