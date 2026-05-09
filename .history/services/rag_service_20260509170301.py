from schemas import RagAnswer, RagAskRequest, RagAskResponse, RetrievedChunk
from llm_client import LLMClient
from rag.document_loader import load_text_file
from rag.chunker import split_by_paragraphs
from rag.embedding_client import SimpleEmbeddingClient
from rag.vector_store import InMemoryVectorStore
from rag.retriever import Retriever
from utils.trace import generate_trace_id


class RagService:
    def __init__(self, llm_client: LLMClient, document_path: str):
        self.llm_client = llm_client

        self.embedding_client = SimpleEmbeddingClient()
        self.vector_store = InMemoryVectorStore()
        self.retriever = Retriever(
            embedding_client=self.embedding_client,
            vector_store=self.vector_store,
        )

        self._build_index(document_path)
        

    def _build_index(self, document_path: str):
        doc_id = "policy_001"

        text = load_text_file(document_path)
        chunks = split_by_paragraphs(text, doc_id=doc_id)

        for chunk in chunks:
            embedding = self.embedding_client.embed(chunk.text)
            self.vector_store.add(chunk, embedding)