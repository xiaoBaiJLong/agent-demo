# schemas.py
from typing import Optional, List
from pydantic import BaseModel, Field
from typing import Literal, Dict, Any

class ChatRequest(BaseModel):
    question: str = Field(..., description="用户问题")
    system_prompt: Optional[str] = Field(default=None, description="系统提示词")
    temperature: float = Field(default=0.2, ge=0, le=2, description="采样温度")
    max_tokens: int = Field(default=1024, ge=1, le=8192, description="最大输出 token 数")


class ChatResponse(BaseModel):
    answer: str = Field(..., description="模型回答")
    model: str = Field(..., description="使用的模型")
    trace_id: str = Field(..., description="链路追踪ID")
    
class RagAskRequest(BaseModel):
    question: str = Field(..., description="用户问题")
    top_k: int = Field(default=3, ge=1, le=10, description="检索返回数量")


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    score: float


class RagAskResponse(BaseModel):
    answer: str
    citations: List[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0, le=1)
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list)
    trace_id: str


class RagAnswer(BaseModel):
    answer: str = Field(..., description="最终答案")
    citations: List[str] = Field(default_factory=list, description="引用来源")
    confidence: float = Field(..., ge=0, le=1, description="置信度")


class RagJsonAnswer(BaseModel):
    answer: str = Field(..., description="最终答案")
    citations: List[str] = Field(default_factory=list, description="引用来源")
    confidence: float = Field(..., ge=0, le=1, description="置信度")


class ChatJsonRequest(BaseModel):
    question: str = Field(..., description="用户问题")
    context: Optional[str] = Field(default=None, description="可选上下文资料")
    
class AgentDecision(BaseModel):
    action_type: Literal["CALL_TOOL", "FINAL_ANSWER", "ASK_USER", "FALLBACK"]
    tool_name: Optional[str] = None
    arguments: Dict[str, Any] = Field(default_factory=dict)
    final_answer: Optional[str] = None
    thought_summary: Optional[str] = None
    is_final: bool = False
    
class RagAnswerResponse(BaseModel):
    success: bool
    answer: Optional[str] = None
    citations: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0, ge=0, le=1)
    error_code: Optional[Literal[
        "INSUFFICIENT_EVIDENCE",
        "INVALID_INPUT",
        "MODEL_OUTPUT_INVALID"
    ]] = None
    message: Optional[str] = None
    
class ErrorResponse(BaseModel):
    success: bool = False
    error_code: str
    message: str
    trace_id: str
    detail: Optional[str] = None