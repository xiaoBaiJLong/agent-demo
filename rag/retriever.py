from rag.embedding_client import SimpleEmbeddingClient
from rag.vector_store import InMemoryVectorStore, SearchResult


class Retriever:
    def __init__(
        self,
        embedding_client: SimpleEmbeddingClient,
        vector_store: InMemoryVectorStore,
    ):
        self.embedding_client = embedding_client
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 3) -> list[SearchResult]:
        query_embedding = self.embedding_client.embed(query)
        return self.vector_store.search(query_embedding, top_k=top_k)