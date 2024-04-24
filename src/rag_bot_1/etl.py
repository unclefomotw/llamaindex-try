# export OPENAI_API_KEY and RAG_DATA_DIR first

import os

from llama_index.core import Settings
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding

from src.rag_bot_1.config import INDEX_PERSIST_DIR, RAG_DATA_DIR
from src.rag_bot_1.logger import get_logger

EMBEDDING_MODEL = OpenAIEmbedding(model="text-embedding-3-small")
Settings.embed_model = EMBEDDING_MODEL

logger = get_logger(__name__)


def build_index():
    documents = SimpleDirectoryReader(
        input_dir=RAG_DATA_DIR,
        recursive=True,
        required_exts=[".md"]
    ).load_data()

    node_parser = SentenceSplitter(chunk_size=512,
                                   chunk_overlap=128)

    transformations = [node_parser]

    index = VectorStoreIndex.from_documents(
        documents=documents,
        transformations=transformations
    )

    index.storage_context.persist(persist_dir=INDEX_PERSIST_DIR)

    return index


if __name__ == '__main__':
    if os.path.exists(INDEX_PERSIST_DIR):
        logger.error(f"{INDEX_PERSIST_DIR} exists when building the index.  Consider removing it.")
    else:
        build_index()
