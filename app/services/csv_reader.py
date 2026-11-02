"""
Módulo de leitura e identificação de colunas de arquivos CSV.

Responsável por:
- Ler o conteúdo de arquivos CSV recebido em bytes.
- Converter os dados para um DataFrame do pandas.
- Padronizar nomes de colunas para facilitar sua identificação.
- Localizar colunas a partir de uma lista de nomes possíveis.

Este módulo é utilizado pela API FairReport para preparar os
arquivos enviados antes da etapa de avaliação das métricas.
"""
from io import BytesIO
import pandas as pd

# Leitura do arquivo CSV
def read_csv_bytes(content: bytes) -> pd.DataFrame:
    if not content:
        raise ValueError("O arquivo CSV está vazio.")

    try:
        df = pd.read_csv(BytesIO(content))
    except Exception as exc:
        raise ValueError(f"Não foi possível ler o CSV: {exc}") from exc

    if df.empty:
        raise ValueError("O arquivo CSV não possui registros.")

    return df

# Padroniza o nome de uma coluna para facilitar sua comparação.
def normalize_column_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


# Localização de uma coluna por nomes alternativos
def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    # Cria um dicionário relacionando cada nome normalizado
    # ao nome original da coluna no DataFrame.
    normalized = {
        normalize_column_name(column): column
        for column in df.columns
    }

    # Percorre os nomes possíveis na ordem recebida.
    for candidate in candidates:

        # Padroniza o nome candidato para compará-lo
        # com os nomes padronizados do DataFrame.
        candidate_normalized = normalize_column_name(candidate)

        # Se houver correspondência, retorna o nome original.
        if candidate_normalized in normalized:
            return normalized[candidate_normalized]

    # Retorna None quando nenhuma coluna correspondente existe.
    return None
