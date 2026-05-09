# main.py
from fastapi import FastAPI, HTTPException
from config import load_settings
from llm_client import LLMClient
from schemas import ChatRequest, ChatResponse, ChatJsonRequest, RagJsonAnswer
from services.chat_service import ChatService

app = FastAPI(
    title="Mini LLM Service",
    description="一个最小的大模型服务封装示例",
    version="0.1.0",
)

settings = load_settings()
llm_client = LLMClient(settings)
chat_service = ChatService(llm_client)
prompt_manager = PromptManager()

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