# export OPENAI_API_KEY and RAG_DATA_DIR first
# The most simple and wrapped example from the tutorial
import logging
import os
import sys

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

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

    query_engine = index.as_query_engine()
    response = query_engine.query("推薦給新手哪個型號的魔術方塊？")
    print(response)


if __name__ == '__main__':
    main()
