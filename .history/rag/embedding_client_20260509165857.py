import math


class SimpleEmbeddingClient:
    """
    教学用简化 embedding。
    不是生产方案。
    用固定关键词构造一个简单向量。
    """

    def __init__(self):
        self.vocabulary = [
            "报销",
            "差旅",
            "发票",
            "审批",
            "500",
            "年假",
            "请假",
            "考勤",
            "迟到",
            "申请",
            "行程单",
            "付款记录",
        ]

    def embed(self, text: str) -> list[float]:
        vector = []

        for word in self.vocabulary:
            vector.append(float(text.count(word)))

        return self._normalize(vector)

    def _normalize(self, vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(x * x for x in vector))

        if norm == 0:
            return vector

        return [x / norm for x in vector]