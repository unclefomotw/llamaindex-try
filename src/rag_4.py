# export OPENAI_API_KEY and RAG_DATA_DIR first

# Create the query engine in a slightly lower level way,
# including setting retriever, node post processors, and synthesis
# Also, customize the embedding model here

import logging
import os
import sys

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    get_response_synthesizer,
    load_index_from_storage
)
# from llama_index.core import Settings
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

RAG_DATA_DIR = os.environ["RAG_DATA_DIR"]
PERSIST_DIR = "./storage/rag_4"

# The same embedding model is required in both creation and load
# ("How" data is embedded is not serialized)
EMBEDDING_MODEL = OpenAIEmbedding(model="text-embedding-3-small")

# To avoid excessive arg passing (see `build_index()` + `load_index()`)
# Another way is just to specify in global setting level
# so that you don't need to specify when loading the index
# Settings.embed_model = EMBEDDING_MODEL

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


def build_index():
    # https://docs.llamaindex.ai/en/stable/module_guides/loading/simpledirectoryreader/
    # it seems .md reader will split a file into multiple Documents by h3 (###)
    documents = SimpleDirectoryReader(
        input_dir=RAG_DATA_DIR,
        recursive=True,
        required_exts=[".md"]
    ).load_data()

    # Specify embedding model when creating index
    # (Or maybe use ServiceContext ??)
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=EMBEDDING_MODEL
    )

    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)

    return index


def load_index():
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    # When loading, need to specify the embedding model information again
    index = load_index_from_storage(
        storage_context,
        embed_model=EMBEDDING_MODEL
    )

    return index


def main():
    if not os.path.exists(PERSIST_DIR):
        index = build_index()
    else:
        index = load_index()

    # retriever takes index
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=10,
    )

    node_postprocessors = [
        SimilarityPostprocessor(similarity_cutoff=0.55)
    ]

    # LLM is with synthesizer, so are prompt templates and response mode
    synthesis_llm = OpenAI(temperature=0.0001)
    response_synthesizer = get_response_synthesizer(
        llm=synthesis_llm
    )

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=node_postprocessors
    )

    response = query_engine.query("CFOP 要怎麼學？有建議的順序嗎？")
    print(response)

    response = query_engine.query("學魔術方塊有哪些網站或影片可以看？")
    print(response)


if __name__ == '__main__':
    main()
