import pypdf
import tiktoken
from typing import Generator, Dict

try:
    encoding = tiktoken.get_encoding("cl100k_base")
except Exception:
    encoding = tiktoken.get_encoding("p50k_base")

def chunk_pdf(file_path: str, doc_name: str, chunk_size: int = 500, overlap: int = 50) -> Generator[Dict, None, None]:
    """
    Read a PDF and split it into token‑based chunks ready for vector storage.
    """
    reader = pypdf.PdfReader(file_path)
    chunk_counter = 0

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if not text.strip():
            continue

        tokens = encoding.encode(text)
        step = chunk_size - overlap
        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i : i + chunk_size]
            chunk_text = encoding.decode(chunk_tokens)
            metadata = {
                "doc_name": doc_name,
                "page": page_num + 1,
                "chunk": chunk_counter,
            }
            yield {"text": chunk_text, "metadata": metadata}
            chunk_counter += 1
