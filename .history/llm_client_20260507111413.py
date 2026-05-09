# llm_client.py
import time
import requests
from typing import Optional
from pydantic import BaseModel
from config import Settings
from schemas import ChatResponse
from typing import Type, TypeVar


T = TypeVar("T", bound=BaseModel)

class LLMClient:
    
    def __init__(self, settings: Settings):
        self.settings = settings

    def chat(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> ChatResponse:
        payload = self._build_payload(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        raw = self._post_with_retry(payload)
        answer = self._extract_answer(raw)

        return ChatResponse(
            answer=answer,
            model=self.settings.model,
            raw_response=raw,
        )

    def _build_payload(
        self,
        user_prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> dict:
        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt,
            })

        messages.append({
            "role": "user",
            "content": user_prompt,
        })

        return {
            "model": self.settings.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

    def _post_with_retry(self, payload: dict) -> dict:
        last_error = None

        for attempt in range(self.settings.max_retries + 1):
            try:
                response = requests.post(
                    self.settings.base_url,
                    headers={
                        "Authorization": f"Bearer {self.settings.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.settings.timeout_seconds,
                )

                response.raise_for_status()
                return response.json()

            except requests.Timeout as e:
                last_error = e
                print(f"LLM request timeout, attempt={attempt + 1}")

            except requests.HTTPError as e:
                status_code = e.response.status_code if e.response else None

                # 4xx 通常是参数或权限问题，不建议盲目重试
                if status_code and 400 <= status_code < 500:
                    raise RuntimeError(f"LLM request failed with client error: {status_code}") from e

                last_error = e
                print(f"LLM server error, attempt={attempt + 1}")

            except requests.RequestException as e:
                last_error = e
                print(f"LLM request failed, attempt={attempt + 1}")

            if attempt < self.settings.max_retries:
                time.sleep(0.5 * (attempt + 1))

        raise RuntimeError("LLM request failed after retries") from last_error

    def _extract_answer(self, raw: dict) -> str:
        """
        不同模型供应商返回结构可能不同。
        这里用一种常见 chat completions 风格示例。
        """
        try:
            return raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"Unexpected LLM response format: {raw}") from e
        
    def chat_with_system(self, system_prompt: str, user_prompt: str) -> ChatResponse:
        return self.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.2,
            max_tokens=1024,
        )
        
    def chat_json(
    self,
    user_prompt: str,
    output_schema: Type[T],
    system_prompt: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 1024,
    ) -> T:
        json_instruction = """
        请严格输出 JSON。
        不要输出 JSON 之外的任何解释、Markdown、代码块标记。
        """
        final_prompt = f"{user_prompt}\n\n{json_instruction}"

        response = self.chat(
            user_prompt=final_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        try:
            data = json.loads(response.answer)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM did not return valid JSON: {response.answer}") from e

        try:
            return output_schema.model_validate(data)
        except Exception as e:
            raise ValueError(f"LLM JSON did not match schema: {data}") from e
    
    def chat_json_with_retry(
    self,
    user_prompt: str,
    output_schema: Type[T],
    system_prompt: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 1024,
    ) -> T:
        try:
            return self.chat_json(
                user_prompt=user_prompt, 
                output_schema=output_schema, 
                system_prompt=system_prompt, 
                temperature=temperature, 
                max_tokens=max_tokens,
                )
        except ValueError as first_error:
            repair_prompt = f"""
            上一次输出不是合法 JSON，错误信息：
            {str(first_error)}

            请重新输出严格合法 JSON，不要包含任何额外文本。
            原始任务：
            {user_prompt}
            """
        return self.chat_json(
            user_prompt=repair_prompt,
            output_schema=output_schema,
            system_prompt=system_prompt,
            temperature=0,
            max_tokens=max_tokens,
        )