"""Download e carga da base pública "Bank Marketing" (UCI Machine Learning Repository).

Fonte: Moro, S., Rita, P. & Cortez, P. (2014). Bank Marketing [Dataset].
UCI Machine Learning Repository. https://doi.org/10.24432/C5K306 (licença CC BY 4.0)

Cada linha é um cliente de um banco português contatado por telefone numa
campanha de marketing. O alvo "y" diz se o cliente contratou um depósito a prazo.
"""

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL_UCI = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_CSV = RAIZ / "data" / "bank-full.csv"

# "duration" (duração da ligação) só é conhecida DEPOIS que a ligação termina,
# e quase define o resultado (ligação de 0s = cliente não contratou). Usá-la
# seria "vazamento de dados" (data leakage): o modelo pareceria ótimo no teste,
# mas seria inútil para decidir QUEM ligar. A própria documentação da base
# recomenda descartá-la para um modelo preditivo realista.
COLUNAS_DESCARTADAS = ["duration"]

ALVO = "y"


def baixar_dados(destino: Path = CAMINHO_CSV) -> Path:
    """Baixa o zip da UCI (só na primeira vez) e extrai o bank-full.csv."""
    if destino.exists():
        return destino

    destino.parent.mkdir(parents=True, exist_ok=True)
    print(f"Baixando {URL_UCI} ...")
    with urllib.request.urlopen(URL_UCI, timeout=120) as resposta:
        zip_externo = zipfile.ZipFile(io.BytesIO(resposta.read()))

    # O arquivo da UCI tem um zip dentro do outro: bank+marketing.zip -> bank.zip -> bank-full.csv
    zip_interno = zipfile.ZipFile(io.BytesIO(zip_externo.read("bank.zip")))
    destino.write_bytes(zip_interno.read("bank-full.csv"))
    print(f"Base salva em {destino}")
    return destino


def carregar_dados(caminho: Path = CAMINHO_CSV) -> pd.DataFrame:
    """Lê o CSV, converte o alvo para 0/1 e remove a coluna com vazamento."""
    if not Path(caminho).exists():
        baixar_dados(caminho)

    df = pd.read_csv(caminho, sep=";")  # a UCI usa ";" como separador
    df[ALVO] = (df[ALVO] == "yes").astype(int)  # 1 = contratou, 0 = não contratou
    return df.drop(columns=COLUNAS_DESCARTADAS)
