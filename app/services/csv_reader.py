from io import BytesIO

import pandas as pd


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


def normalize_column_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {
        normalize_column_name(column): column
        for column in df.columns
    }

    for candidate in candidates:
        candidate_normalized = normalize_column_name(candidate)

        if candidate_normalized in normalized:
            return normalized[candidate_normalized]

    return None
