
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from app.services.csv_reader import read_csv_bytes
from app.services.evaluation import evaluate_model


# Cria a aplicação FastAPI e define as informações exibidas
# na documentação automática da API, disponível em /docs.
app = FastAPI(
    title="FairReport API",
    description="API para avaliação de performance e justiça algorítmica.",
    version="1.0.0",
)

# Lista de origens autorizadas a acessar a API.
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]

# Permite requisições das origens listadas, com credenciais,
# métodos HTTP e cabeçalhos.
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Verifica o acesso básico à API.
@app.get("/")
def root():
    return {
        "message": "FairReport API funcionando.",
        "docs": "/docs",
    }

# Endpoint de verificação de conexão com a API
@app.get("/health")
def health():
    return {"status": "ok"}

# Endpoint de avaliação do modelo
@app.post("/avaliar")
async def avaliar(
    # Nome utilizado para identificar o modelo avaliado.
    modelo: str = Form(...),
    metricas_performance: str = Form(""),
    metricas_justica: str = Form(...),
    dados_teste: UploadFile = File(...),
    atributo_protegido: UploadFile = File(...),
    observacoes: str = Form(""),
):
    # Impede que o formulário seja enviado com o nome vazio
    if not modelo.strip():
        raise HTTPException(
            status_code=400,
            detail="O campo 'modelo' é obrigatório.",
        )

    # Verifica se o arquivo de teste possui extensão CSV.
    if not dados_teste.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="O arquivo de dados de teste deve ser CSV.",
        )
    # Verifica se o arquivo do atributo protegido possui extensão CSV.
    if not atributo_protegido.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="O arquivo de atributo protegido deve ser CSV.",
        )

    try:
        # Leitura dos arquivos enviados
        dados_teste_bytes = await dados_teste.read()
        atributo_protegido_bytes = await atributo_protegido.read()

        # Converte os bytes dos CSVs em DataFrames do pandas,
        # por meio da função definida no serviço csv_reader.
        dados_df = read_csv_bytes(dados_teste_bytes)
        protegido_df = read_csv_bytes(atributo_protegido_bytes)

        # Transforma a string de métricas de performance em
        # uma lista, separando os itens pelas vírgulas.
        # Remove espaços extras e ignora itens vazios.
        performance = [
            item.strip()
            for item in metricas_performance.split(",")
            if item.strip()
        ]

        justica = [
            item.strip()
            for item in metricas_justica.split(",")
            if item.strip()
        ]
        # Encaminha os DataFrames e as métricas selecionadas
        # para a função responsável pelos cálculos.
        resultado = evaluate_model(
            dados_teste=dados_df,
            atributo_protegido=protegido_df,
            metricas_performance=performance,
            metricas_justica=justica,
        )
        # Retorna os dados em formato JSON para o frontend.
        return {
            "modelo": modelo.strip(),
            "observacoes": observacoes,
            "resultado": resultado,
        }
    # Tratamento de erros de validação
    except ValueError as exc:
        # Erros de validação ou de conteúdo dos dados são
        # retornados ao cliente com o código HTTP 400.
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        # Captura erros não previstos durante o processamento
        # e retorna o código HTTP 500.
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao processar a avaliação: {exc}",
        ) from exc
