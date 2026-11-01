from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from app.services.csv_reader import read_csv_bytes
from app.services.evaluation import evaluate_model


app = FastAPI(
    title="FairReport API",
    description="API para avaliação de performance e justiça algorítmica.",
    version="1.0.0",
)

# Ajuste as origens conforme o endereço utilizado pelo seu frontend.
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "FairReport API funcionando.",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/avaliar")
async def avaliar(
    modelo: str = Form(...),
    metricas_performance: str = Form(""),
    metricas_justica: str = Form(...),
    dados_teste: UploadFile = File(...),
    atributo_protegido: UploadFile = File(...),
    observacoes: str = Form(""),
):
    if not modelo.strip():
        raise HTTPException(
            status_code=400,
            detail="O campo 'modelo' é obrigatório.",
        )

    if not dados_teste.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="O arquivo de dados de teste deve ser CSV.",
        )

    if not atributo_protegido.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="O arquivo de atributo protegido deve ser CSV.",
        )

    try:
        dados_teste_bytes = await dados_teste.read()
        atributo_protegido_bytes = await atributo_protegido.read()

        dados_df = read_csv_bytes(dados_teste_bytes)
        protegido_df = read_csv_bytes(atributo_protegido_bytes)

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

        resultado = evaluate_model(
            dados_teste=dados_df,
            atributo_protegido=protegido_df,
            metricas_performance=performance,
            metricas_justica=justica,
        )

        return {
            "modelo": modelo.strip(),
            "observacoes": observacoes,
            "resultado": resultado,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao processar a avaliação: {exc}",
        ) from exc
