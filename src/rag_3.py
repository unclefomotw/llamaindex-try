# export OPENAI_API_KEY and RAG_DATA_DIR first
# Set up a local Qdrant by https://qdrant.tech/documentation/quick-start/
# Start via run-qdrant.sh, and inspect at http://localhost:6333/dashboard
# Also, pip install llama-index-vector-stores-qdrant
# (which also installs qdrant-client)

# Use vector database as the storage

import logging
import os
import sys

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

RAG_DATA_DIR = os.environ["RAG_DATA_DIR"]

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


def ingest_to_db():
    """
    One time ETL.  This leverages llama-index (which is not necessary) to store data
    into the DB.
    """

    # "Crawl" the interesting raw files into Documents
    documents = SimpleDirectoryReader(
        input_dir=RAG_DATA_DIR,
        recursive=True,
        required_exts=[".md"]
    ).load_data()

    # Start db via my "run-qdrant.sh"
    # This is from qdrant, not llama-index
    db_client = QdrantClient(host="localhost", port=6333)

    # Storage objects of llama-index
    # pass the DB client to the vector store
    vector_store = QdrantVectorStore(
        collection_name="rag_3",
        client=db_client
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Feed data into db (the collection is not created before this step)
    # By calling `.from_documents` with storage_context/vector store
    # the `build_index_from_nodes` is called, which takes its _vector_store and add nodes
    # Hence using the db_client to insert data into Qdrant DB
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context
    )


def query_db():
    # Declare the Qdrant VectorStore; assuming the data is ingested
    db_client = QdrantClient(host="localhost", port=6333)
    vector_store = QdrantVectorStore(
        collection_name="rag_3",
        client=db_client
    )

    # Build via `.from_vector_store` instead of `.from_documents`
    # The index itself does NOT hold any nodes (see below)
    index = VectorStoreIndex.from_vector_store(vector_store)
    query_engine = index.as_query_engine()

    # During inference, the major operations from the query_engine
    # happen in its retriever, where the vector store's `query()` is called
    response = query_engine.query("CFOP 要怎麼學？有建議的順序嗎？")
    print(response)


if __name__ == '__main__':
    # Only execute this once; can be done separately
    # ingest_to_db()

    # Can execute separately AFTER the DB is filled with the data
    query_db()
