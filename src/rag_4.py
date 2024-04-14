# export OPENAI_API_KEY and RAG_DATA_DIR first

# Create the query engine in a slightly lower level way,
# including setting retriever, node post processors, and synthesis

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
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI

RAG_DATA_DIR = os.environ["RAG_DATA_DIR"]
PERSIST_DIR = "./storage"

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

    # make the index in the memory
    index = VectorStoreIndex.from_documents(documents)

    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)

    return index


def load_index():
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

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
        SimilarityPostprocessor(similarity_cutoff=0.85)
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


if __name__ == '__main__':
    main()
