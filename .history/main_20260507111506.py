from config import load_settings
from llm_client import LLMClient
from schemas import RagJsonAnswer


def main():
    client = LLMClient(load_settings())

    prompt = """
    请根据以下资料回答问题。

    问题：
    差旅报销超过500元是否需要审批？

    资料：
    制度第4.3条：单笔差旅支出超过500元，需附审批单。

    请输出字段：
    answer, citations, confidence
    """

    result = client.chat_json(
        user_prompt=prompt,
        output_schema=RagJsonAnswer,
        system_prompt="你是一名严谨的企业制度问答助手。",
    )

    print(result.model_dump())


if __name__ == "__main__":
    main()