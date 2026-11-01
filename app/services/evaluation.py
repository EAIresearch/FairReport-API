import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    equal_opportunity_difference,
)

from app.services.csv_reader import find_column


PERFORMANCE_ALIASES = {
    "accuracy": [
        "accuracy",
        "acuracia",
        "acurácia",
    ],
    "micro-f1": [
        "micro-f1",
        "micro_f1",
        "microf1",
    ],
    "macro-f1": [
        "macro-f1",
        "macro_f1",
        "macrof1",
    ],
}

JUSTICE_ALIASES = {
    "dpd": [
        "dpd",
        "spd",
        "demographic_parity_difference",
    ],
    "eod": [
        "eod",
        "equalized_odds_difference",
    ],
    "eopd": [
        "eopd",
        "equal_opportunity_difference",
    ],
}


def _clean_selection(values: list[str]) -> list[str]:
    return [str(value).strip().lower() for value in values]


def _find_selected_metric(
    selected: list[str],
    aliases: list[str],
) -> bool:
    normalized = _clean_selection(selected)

    return any(alias in normalized for alias in aliases)


def _to_json_number(value):
    if value is None:
        return None

    value = float(value)

    if not np.isfinite(value):
        return None

    return value


def _prepare_data(
    dados_teste: pd.DataFrame,
    atributo_protegido: pd.DataFrame,
):
    if len(dados_teste) != len(atributo_protegido):
        raise ValueError(
            "Os arquivos de dados de teste e atributo protegido "
            "devem possuir a mesma quantidade de linhas."
        )

    if len(dados_teste.columns) != 2:
        raise ValueError(
            "O CSV de dados de teste deve possuir exatamente duas colunas: "
            "rótulo real e rótulo predito."
        )

    if len(atributo_protegido.columns) != 1:
        raise ValueError(
            "O CSV de atributo protegido deve possuir exatamente uma coluna."
        )

    # Aceita nomes como:
    # actual / predicted
    # real / predicted
    # y_true / y_pred
    # rotulo_real / rotulo_predito
    true_column = find_column(
        dados_teste,
        [
            "actual",
            "real",
            "y_true",
            "true",
            "rotulo_real",
            "rótulo_real",
        ],
    )

    pred_column = find_column(
        dados_teste,
        [
            "predicted",
            "predito",
            "y_pred",
            "prediction",
            "rotulo_predito",
            "rótulo_predito",
        ],
    )

    # Se os nomes não forem reconhecidos, usa a ordem das colunas.
    if true_column is None:
        true_column = dados_teste.columns[0]

    if pred_column is None:
        pred_column = dados_teste.columns[1]

    protected_column = atributo_protegido.columns[0]

    y_true = dados_teste[true_column]
    y_pred = dados_teste[pred_column]
    sensitive_features = atributo_protegido[protected_column]

    if y_true.isna().any() or y_pred.isna().any():
        raise ValueError(
            "Os dados de teste não podem possuir valores vazios."
        )

    if sensitive_features.isna().any():
        raise ValueError(
            "O atributo protegido não pode possuir valores vazios."
        )

    return (
        y_true.reset_index(drop=True),
        y_pred.reset_index(drop=True),
        sensitive_features.reset_index(drop=True),
    )


def calculate_performance_metrics(
    y_true,
    y_pred,
    selected_metrics: list[str],
):
    result = {}

    if _find_selected_metric(
        selected_metrics,
        PERFORMANCE_ALIASES["accuracy"],
    ):
        result["accuracy"] = _to_json_number(
            accuracy_score(y_true, y_pred)
        )

    if _find_selected_metric(
        selected_metrics,
        PERFORMANCE_ALIASES["micro-f1"],
    ):
        result["micro_f1"] = _to_json_number(
            f1_score(
                y_true,
                y_pred,
                average="micro",
                zero_division=0,
            )
        )

    if _find_selected_metric(
        selected_metrics,
        PERFORMANCE_ALIASES["macro-f1"],
    ):
        result["macro_f1"] = _to_json_number(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        )

    return result


def calculate_fairness_metrics(
    y_true,
    y_pred,
    sensitive_features,
    selected_metrics: list[str],
):
    result = {}

    if _find_selected_metric(
        selected_metrics,
        JUSTICE_ALIASES["dpd"],
    ):
        result["dpd"] = _to_json_number(
            demographic_parity_difference(
                y_true,
                y_pred,
                sensitive_features=sensitive_features,
            )
        )

    if _find_selected_metric(
        selected_metrics,
        JUSTICE_ALIASES["eod"],
    ):
        result["eod"] = _to_json_number(
            equalized_odds_difference(
                y_true,
                y_pred,
                sensitive_features=sensitive_features,
            )
        )

    if _find_selected_metric(
        selected_metrics,
        JUSTICE_ALIASES["eopd"],
    ):
        result["eopd"] = _to_json_number(
            equal_opportunity_difference(
                y_true,
                y_pred,
                sensitive_features=sensitive_features,
            )
        )

    return result


def calculate_group_results(
    y_true,
    y_pred,
    sensitive_features,
):
    """
    Retorna resultados básicos por grupo.
    Isso será útil para montar gráficos/tabelas no frontend.
    """

    data = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "grupo": sensitive_features,
        }
    )

    groups = {}

    for group, group_df in data.groupby("grupo", dropna=False):
        group_name = str(group)

        groups[group_name] = {
            "quantidade": int(len(group_df)),
            "accuracy": _to_json_number(
                accuracy_score(
                    group_df["y_true"],
                    group_df["y_pred"],
                )
            ),
        }

    return groups


def evaluate_model(
    dados_teste: pd.DataFrame,
    atributo_protegido: pd.DataFrame,
    metricas_performance: list[str],
    metricas_justica: list[str],
):
    y_true, y_pred, sensitive_features = _prepare_data(
        dados_teste,
        atributo_protegido,
    )

    if len(metricas_justica) == 0:
        raise ValueError(
            "Selecione pelo menos uma métrica de justiça algorítmica."
        )

    performance = calculate_performance_metrics(
        y_true,
        y_pred,
        metricas_performance,
    )

    fairness = calculate_fairness_metrics(
        y_true,
        y_pred,
        sensitive_features,
        metricas_justica,
    )

    groups = calculate_group_results(
        y_true,
        y_pred,
        sensitive_features,
    )

    return {
        "performance": performance,
        "fairness": fairness,
        "groups": groups,
        "quantidade_registros": int(len(y_true)),
    }
