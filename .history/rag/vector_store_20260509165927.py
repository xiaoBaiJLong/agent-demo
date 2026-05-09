import math
from dataclasses import dataclass
from rag.chunker import Chunk


@dataclass
class VectorRecord:
    chunk: Chunk
    embedding: list[float]


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


class InMemoryVectorStore:
    def __init__(self):
        self.records: list[VectorRecord] = []

    def add(self, chunk: Chunk, embedding: list[float]):
        self.records.append(
            VectorRecord(
                chunk=chunk,
                embedding=embedding,
            )
        )

    def search(self, query_embedding: list[float], top_k: int = 3) -> list[SearchResult]:
        results = []

        for record in self.records:
            score = cosine_similarity(query_embedding, record.embedding)
            results.append(
                SearchResult(
                    chunk=record.chunk,
                    score=score,
                )
            )

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Vector dimensions do not match")

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)