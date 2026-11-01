# FairReport Backend

Backend em Python para o projeto FairReport.

## Tecnologias

- FastAPI
- Fairlearn
- Pandas
- NumPy
- Scikit-learn

## 1. Criar ambiente virtual

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 2. Instalar dependências

```bash
pip install -r requirements.txt
```

## 3. Executar

```bash
uvicorn app.main:app --reload
```

O backend ficará disponível em:

http://127.0.0.1:8000

Documentação Swagger:

http://127.0.0.1:8000/docs

## 4. Endpoint principal

```text
POST /avaliar
```

Campos:

- modelo
- metricas_performance
- metricas_justica
- dados_teste
- atributo_protegido
- observacoes

### Exemplo

```text
modelo = Modelo Exemplo
metricas_performance = accuracy,micro-f1,macro-f1
metricas_justica = dpd,eod,eopd
dados_teste = dados_teste.csv
atributo_protegido = atributo_protegido.csv
observacoes = Avaliação inicial
```

## 5. Formato dos CSVs

### dados_teste.csv

O arquivo deve possuir duas colunas:

```csv
actual,predicted
1,1
0,0
1,0
1,1
0,1
```

O backend também reconhece nomes como:

- actual / predicted
- real / predito
- y_true / y_pred
- rotulo_real / rotulo_predito

Se os nomes não forem reconhecidos, a primeira coluna será considerada o rótulo real e a segunda o rótulo predito.

### atributo_protegido.csv

Deve possuir uma única coluna:

```csv
grupo
mulher
homem
mulher
homem
mulher
```

A ordem das linhas dos dois arquivos deve ser a mesma.

## 6. Resposta

Exemplo:

```json
{
  "modelo": "Modelo Exemplo",
  "observacoes": "Avaliação inicial",
  "resultado": {
    "performance": {
      "accuracy": 0.8,
      "micro_f1": 0.8,
      "macro_f1": 0.79
    },
    "fairness": {
      "dpd": 0.12,
      "eod": 0.18,
      "eopd": 0.15
    },
    "groups": {
      "homem": {
        "quantidade": 50,
        "accuracy": 0.84
      },
      "mulher": {
        "quantidade": 50,
        "accuracy": 0.76
      }
    },
    "quantidade_registros": 100
  }
}
```
