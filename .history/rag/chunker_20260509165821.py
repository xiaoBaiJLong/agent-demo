from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str


def split_by_paragraphs(text: str, doc_id: str) -> list[Chunk]:
    paragraphs = [
        p.strip()
        for p in text.split("\n\n")
        if p.strip()
    ]

    chunks = []

    for index, paragraph in enumerate(paragraphs):
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}_chunk_{index + 1}",
                doc_id=doc_id,
                text=paragraph,
            )
        )

    return chunks