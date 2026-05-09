# schemas.py
from typing import Optional, List
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., description="用户问题")
    system_prompt: Optional[str] = Field(default=None, description="系统提示词")
    temperature: float = Field(default=0.2, ge=0, le=2, description="采样温度")
    max_tokens: int = Field(default=1024, ge=1, le=8192, description="最大输出 token 数")


class ChatResponse(BaseModel):
    answer: str = Field(..., description="模型回答")
    model: str = Field(..., description="使用的模型")
    trace_id: str = Field(..., description="链路追踪ID")


class RagJsonAnswer(BaseModel):
    answer: str = Field(..., description="最终答案")
    citations: List[str] = Field(default_factory=list, description="引用来源")
    confidence: float = Field(..., ge=0, le=1, description="置信度")


class ChatJsonRequest(BaseModel):
    question: str = Field(..., description="用户问题")
    context: Optional[str] = Field(default=None, description="可选上下文资料")
    
class RagAnswer(BaseModel):
    answer: str
    citations: List[str]
    confidence: float = Field(..., ge=0, le=1)