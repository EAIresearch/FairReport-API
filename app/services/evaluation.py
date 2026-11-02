"""
Módulo de avaliação de performance e justiça algorítmica.

Responsável por:
- Identificar as métricas selecionadas pelo usuário.
- Validar e preparar os dados de teste e o atributo protegido.
- Calcular métricas de performance do modelo.
- Calcular métricas de justiça algorítmica utilizando Fairlearn.
- Calcular a acurácia separadamente para cada grupo protegido.
- Organizar os resultados para serem retornados pela API FairReport.

As métricas são calculadas a partir dos rótulos reais, dos rótulos
preditos e dos grupos definidos no atributo protegido.
"""

import numpy as np
import pandas as pd

# Métricas de desempenho disponibilizadas pelo scikit-learn.
from sklearn.metrics import accuracy_score, f1_score

# Métricas de justiça algorítmica disponibilizadas pelo Fairlearn.
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    equal_opportunity_difference,
)

# Função auxiliar para localizar colunas por nomes alternativos.
from app.services.csv_reader import find_column


# Nomes alternativos das métricas de performance
# Permite reconhecer diferentes formas de escrever o nome
# de uma métrica selecionada no frontend.
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

# Nomes alternativos das métricas de justiça algorítmica
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

# Normalização dos nomes das métricas selecionadas
def _clean_selection(values: list[str]) -> list[str]:
    return [str(value).strip().lower() for value in values]


# Verificação da seleção de uma métrica
def _find_selected_metric(
    selected: list[str],
    aliases: list[str],
) -> bool:
    # Padroniza os valores recebidos do frontend.
    normalized = _clean_selection(selected)

    # Verifica se pelo menos um nome alternativo foi selecionado.
    return any(alias in normalized for alias in aliases)

# Conversão de resultados para números compatíveis com JSON
def _to_json_number(value):
    # Preserva valores ausentes como None.
    if value is None:
        return None

    #Converte o resultado para um numero float
    value = float(value)

    # Verifica se o número é finito.
    # Valores NaN e infinito são representados como None.
    if not np.isfinite(value):
        return None

    return value

# Validação e preparação dos dados
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
    # Procura nomes conhecidos que representem o rótulo real.
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

    # Procura nomes conhecidos que representem a previsão.
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

# Cálculo das métricas de performance
def calculate_performance_metrics(
    y_true,
    y_pred,
    selected_metrics: list[str],
):

    """
    Calcula as métricas de desempenho selecionadas pelo usuário.

    Parâmetros:
        y_true:
            Série ou vetor contendo os rótulos reais.

        y_pred:
            Série ou vetor contendo os rótulos preditos pelo modelo.

        selected_metrics (list[str]):
            Lista de métricas de performance selecionadas.

    Retorna:
        dict:
            Dicionário contendo somente as métricas solicitadas.
            As métricas não selecionadas não são incluídas.

    Métricas disponíveis:
        accuracy:
            Proporção de previsões iguais aos rótulos reais.

        micro_f1:
            F1 calculado com média micro, que agrega as
            contribuições das classes antes de calcular a métrica.

        macro_f1:
            Média aritmética dos valores F1 de cada classe,
            atribuindo o mesmo peso a cada classe.

    Observação:
        O valor de cada métrica normalmente fica entre 0 e 1.
        A interpretação depende da métrica e dos dados avaliados.
    """

    # Dicionário que armazenará as métricas calculadas.
    result = {}

    # Calcula a acurácia somente se ela tiver sido selecionada.
    if _find_selected_metric(
        selected_metrics,
        PERFORMANCE_ALIASES["accuracy"],
    ):
        result["accuracy"] = _to_json_number(
            accuracy_score(y_true, y_pred)
        )
    # Calcula o F1 com média micro quando solicitado.
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
    # Calcula o F1 com média macro quando solicitado.
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

    # Retorna somente as métricas calculadas.
    return result

# Cálculo das métricas de justiça algorítmica
def calculate_fairness_metrics(
    y_true,
    y_pred,
    sensitive_features,
    selected_metrics: list[str],
):
    """
       Calcula as métricas de justiça algorítmica selecionadas.

       Parâmetros:
           y_true:
               Rótulos reais dos registros.

           y_pred:
               Rótulos preditos pelo modelo.

           sensitive_features:
               Série contendo os grupos do atributo protegido,
               como "homem" e "mulher".

           selected_metrics (list[str]):
               Lista de métricas de justiça selecionadas.

       Retorna:
           dict:
               Dicionário contendo as métricas solicitadas.

       Métricas disponíveis:
           dpd:
               Demographic Parity Difference. Mede a diferença
               entre as taxas de previsões positivas dos grupos.

           eod:
               Equalized Odds Difference. Compara as diferenças
               nas taxas de verdadeiro positivo e falso positivo
               entre os grupos.

           eopd:
               Equal Opportunity Difference. Compara as taxas
               de verdadeiro positivo entre os grupos.

       Observação:
           Essas métricas são calculadas pela biblioteca Fairlearn.
           Seus resultados precisam ser interpretados considerando
           a definição de cada métrica e os dados utilizados.
       """

    result = {}

    # Calcula a diferença de paridade demográfica quando solicitada.
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
    # Calcula a diferença de equalized odds quando solicitada.
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
    # Calcula a diferença de equal opportunity quando solicitada.
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

# Cálculo dos resultados individuais por grupo
def calculate_group_results(
    y_true,
    y_pred,
    sensitive_features,
):
    """
    Retorna resultados a acuracia por atributo protegido
    além da quantidade de itens analidos
    """

    # Cria uma tabela temporária que reúne os rótulos reais,
    # os rótulos preditos e o grupo correspondente a cada registro.
    data = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "grupo": sensitive_features,
        }
    )

    groups = {}

    # Separa os registros pelo valor do atributo protegido.
    # dropna=False mantém grupos nulos caso existam.
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

# Função principal de avaliação
def evaluate_model(
    dados_teste: pd.DataFrame,
    atributo_protegido: pd.DataFrame,
    metricas_performance: list[str],
    metricas_justica: list[str],
):
    """
        Executa o processo completo de avaliação do modelo.

        Parâmetros:
            dados_teste (pd.DataFrame):
                DataFrame com os rótulos reais e preditos.

            atributo_protegido (pd.DataFrame):
                DataFrame com uma coluna que identifica os grupos.

            metricas_performance (list[str]):
                Lista das métricas de performance selecionadas.

            metricas_justica (list[str]):
                Lista das métricas de justiça selecionadas.

        Retorna:
            dict:
                Dicionário com quatro elementos principais:
                - performance: métricas de desempenho solicitadas.
                - fairness: métricas de justiça solicitadas.
                - groups: resultados individuais por grupo.
                - quantidade_registros: total de registros avaliados.

        Raises:
            ValueError:
                Se os dados forem inválidos ou nenhuma métrica
                de justiça for selecionada.

        Funcionamento:
            1. Valida e prepara os dados.
            2. Verifica a seleção das métricas de justiça.
            3. Calcula as métricas de performance.
            4. Calcula as métricas de justiça algorítmica.
            5. Calcula a acurácia e a quantidade de registros por grupo.
            6. Organiza e retorna os resultados.
        """

    # Preparação dos dados
    y_true, y_pred, sensitive_features = _prepare_data(
        dados_teste,
        atributo_protegido,
    )
    # Validação das métricas de justiça
    # Exige pelo menos uma métrica de justiça algorítmica.
    # A verificação considera se a lista está vazia.
    if len(metricas_justica) == 0:
        raise ValueError(
            "Selecione pelo menos uma métrica de justiça algorítmica."
        )

    # Calcula as métricas de desempenho selecionadas no frontend.
    performance = calculate_performance_metrics(
        y_true,
        y_pred,
        metricas_performance,
    )
    # Calcula as métricas de justiça selecionadas pelo usuário.
    fairness = calculate_fairness_metrics(
        y_true,
        y_pred,
        sensitive_features,
        metricas_justica,
    )
    # Calcula a quantidade de registros e a acurácia de cada grupo
    # identificado no arquivo de atributo protegido.
    groups = calculate_group_results(
        y_true,
        y_pred,
        sensitive_features,
    )
    # Retorna os resultados em um dicionário que poderá ser
    # convertido para JSON pela API e exibido no frontend Vue.
    return {
        "performance": performance,
        "fairness": fairness,
        "groups": groups,
        "quantidade_registros": int(len(y_true)),
    }
