# services/chat_service.py
from llm_client import LLMClient
from schemas import ChatRequest, ChatResponse, ChatJsonRequest, RagJsonAnswer
from utils.trace import generate_trace_id
from prompt_templates import build_rag_qa_prompt
from prompt_manager import PromptManager


class ChatService:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def chat(self, request: ChatRequest) -> ChatResponse:
        trace_id = generate_trace_id()

        response = self.llm_client.chat(
            system_prompt=request.system_prompt,
            user_prompt=request.question,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        return ChatResponse(
            answer=response.answer,
            model=response.model,
            trace_id=trace_id,
        )

    def chat_json(self, request: ChatJsonRequest) -> RagJsonAnswer:
        context_text = request.context or "无"

        prompt = pro(
            question=request.question,
            contexts=context_text,
        )

        return self.llm_client.chat_json(
            user_prompt=prompt,
            output_schema=RagJsonAnswer,
            system_prompt="你是一名严谨的企业知识库问答助手。",
            temperature=0.1,
            max_tokens=1024,
        )
