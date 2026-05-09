# prompt_manager.py

from dataclasses import dataclass


@dataclass
class PromptTemplate:
    prompt_id: str
    version: str
    template: str


class PromptManager:
    def __init__(self):
        self.templates = {
            "rag_qa:v1": PromptTemplate(
                prompt_id="rag_qa",
                version="v1",
                template="""
你是一名严谨的企业知识库问答助手。

任务：
请根据参考资料回答用户问题。

用户问题：
{question}

参考资料：
{contexts}

规则：
1. 只能依据参考资料回答。
2. 如果资料不足，请返回“未找到足够依据”。
3. 必须附引用。

输出 JSON：
{{
  "answer": "字符串",
  "citations": ["引用来源ID"],
  "confidence": 0到1之间的小数
}}
"""
            )
        }

    def render(self, prompt_id: str, version: str, variables: dict) -> str:
        key = f"{prompt_id}:{version}"

        if key not in self.templates:
            raise ValueError(f"Prompt template not found: {key}")

        template = self.templates[key].template
        return template.format(**variables)