# test_llm_client.py
from config import load_settings
from llm_client import LLMClient


def main():
    settings = load_settings()
    client = LLMClient(settings)

    response = client.chat(
        system_prompt="你是一名通俗易懂的大模型老师。",
        user_prompt="用高中生能听懂的话解释什么是 RAG。",
        temperature=0.2,
        max_tokens=800,
    )

    print("模型：", response.model)
    print("回答：")
    print(response.answer)


if __name__ == "__main__":
    main()