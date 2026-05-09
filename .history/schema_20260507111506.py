# schemas.py
from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    question: str = Field(..., description="用户问题")
    system_prompt: Optional[str] = Field(default=None, description="系统提示词")
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_tokens: int = Field(default=1024, ge=1, le=8192)


class ChatResponse(BaseModel):
    answer: str
    model: str
    raw_response: Optional[dict] = None
    
    
class RagJsonAnswer(BaseModel):
    answer: str = Field(..., description="最终答案")
    citations: List[str] = Field(default_factory=list, description="引用来源")
    confidence: float = Field(..., ge=0, le=1, description="置信度")