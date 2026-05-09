# main.py
from fastapi import FastAPI, HTTPException
from config import load_settings
from llm_client import LLMClient
from schemas import ChatRequest, ChatResponse, ChatJsonRequest, RagJsonAnswer
from services.chat_service import ChatService
from utils.errors import AppError
from fastapi import Request
from utils.trace import generate_trace_id
from schemas import ErrorResponse
from fastapi.responses import JSONResponse

app = FastAPI(
    title="Mini LLM Service",
    description="一个最小的大模型服务封装示例",
    version="0.1.0",
)

settings = load_settings()
llm_client = LLMClient(settings)
chat_service = ChatService(llm_client)
rag_service = RagService(
    llm_client=llm_client,
    document_path="data/policies.md",
)




@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "mini-llm-service",
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        return chat_service.chat(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat-json", response_model=RagJsonAnswer)
def chat_json(request: ChatJsonRequest):
    try:
        return chat_service.chat_json(request)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.post("/rag/ask", response_model=RagAskResponse)
def rag_ask(request_body: RagAskRequest, request: Request):
    trace_id = request.state.trace_id
    return rag_service.ask(request_body, trace_id=trace_id)
    

@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    trace_id = getattr(request.state, "trace_id", "unknown")

    response = ErrorResponse(
        error_code=exc.error_code,
        message=exc.message,
        detail=exc.detail,
        trace_id=trace_id,
    )

    return JSONResponse(
        status_code=400,
        content=response.model_dump(),
    )
    
    

@app.middleware("http")
async def trace_middleware(request: Request, call_next):
    trace_id = request.headers.get("X-Trace-Id") or generate_trace_id()
    request.state.trace_id = trace_id

    response = await call_next(request)
    response.headers["X-Trace-Id"] = trace_id

    return response