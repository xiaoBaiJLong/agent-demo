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
            
    def ask(self, request: RagAskRequest) -> RagAskResponse:
        trace_id = generate_trace_id()

        search_results = self.retriever.retrieve(
            query=request.question,
            top_k=request.top_k,
        )

        contexts = self._build_contexts(search_results)

        prompt = build_rag_ask_prompt(
            question=request.question,
            contexts=contexts,
        )

        rag_answer = self.llm_client.chat_json(
            user_prompt=prompt,
            output_schema=RagAnswer,
            system_prompt="你是一名严谨的企业知识库问答助手。",
            temperature=0.1,
            max_tokens=1024,
        )

        retrieved_chunks = [
            RetrievedChunk(
                chunk_id=result.chunk.chunk_id,
                doc_id=result.chunk.doc_id,
                text=result.chunk.text,
                score=result.score,
            )
            for result in search_results
        ]

        return RagAskResponse(
            answer=rag_answer.answer,
            citations=rag_answer.citations,
            confidence=rag_answer.confidence,
            retrieved_chunks=retrieved_chunks,
            trace_id=trace_id,
        )

    def _build_contexts(self, search_results) -> str:
        lines = []

        for result in search_results:
            lines.append(f"chunk_id: {result.chunk.chunk_id}")
            lines.append(f"doc_id: {result.chunk.doc_id}")
            lines.append(f"score: {result.score:.4f}")
            lines.append(f"text: {result.chunk.text}")
            lines.append("")

        return "\n".join(lines)