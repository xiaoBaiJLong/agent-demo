# services/chat_service.py
from llm_client import LLMClient
from schemas import ChatRequest, ChatResponse, ChatJsonRequest, RagJsonAnswer
from utils.trace import generate_trace_id


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

        prompt = f"""
        请根据上下文回答用户问题。

        用户问题：
        {request.question}

        上下文：
        {context_text}

        请严格输出 JSON，字段包括：
        - answer: 字符串
        - citations: 字符串数组
        - confidence: 0到1之间的小数
        """

        return self.llm_client.chat_json(
            user_prompt=prompt,
            output_schema=RagJsonAnswer,
            system_prompt="你是一名严谨的企业知识库问答助手。",
            temperature=0.1,
            max_tokens=1024,
        )