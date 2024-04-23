# export OPENAI_API_KEY first
# Set up a local Qdrant by https://qdrant.tech/documentation/quick-start/
# Start via run-qdrant.sh, and inspect at http://localhost:6333/dashboard
# Also, pip install llama-index-vector-stores-qdrant

# This: Use different chunking/indexing strategies
#       Each pair of build_* + query_* invoked by main is one strategy
#
# With previous: Vector database as the storage
#                Embedding model on HuggingFace (pip install llama-index-embeddings-huggingface)
#                Retrieve in a lower level way

import logging
import sys

import llama_index.core.instrumentation as instrument
from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
    get_response_synthesizer,
)
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.node_parser import (
    SentenceSplitter,
    SentenceWindowNodeParser,
)
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from src.my_handler import NaiveEventHandler

Settings.embed_model = HuggingFaceEmbedding(model_name="DMetaSoul/Dmeta-embedding-zh-small")

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
dispatcher = instrument.get_dispatcher()
dispatcher.add_event_handler(NaiveEventHandler(log_file="rag_7.log"))


def load_raw_doc():
    """Makes a simple document from one text file."""

    # with open("data/kociemba.txt") as f:
    with open("data/kociemba-en-punct.txt") as f:
        txt = f.readlines()
    one_txt = "".join(txt)
    return [Document(text=one_txt)]


def ingest_to_db(nodes, collection_name):
    """
    One time ETL.  This leverages llama-index (which is not necessary) to store data
    into the DB.
    """

    # Start db via my "run-qdrant.sh"
    # This is from qdrant, not llama-index
    db_client = QdrantClient(host="localhost", port=6333)

    # Storage objects of llama-index
    # pass the DB client to the vector store
    vector_store = QdrantVectorStore(
        collection_name=collection_name,
        client=db_client
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Feed data into db (the collection is not created before this step)
    # By constructing the object with storage_context/vector store
    # the `build_index_from_nodes` is called, which takes its _vector_store and add nodes
    # Hence using the db_client to insert data into Qdrant DB
    VectorStoreIndex(nodes, storage_context=storage_context)


def build_sentence_window_index():
    """Splits by sentences: `content`=a chunk; `metadata.window`=chunks in the window"""

    # SentenceWindowNodeParser splits sentences by English punctuation such as ".!?"
    # See split_by_sentence_tokenizer() and llama_index.core.node_parser.text.util
    # To customize, the best bet is to give your own `sentence_splitter` in arg

    # A node is ONE sentence, with a window of sentences as its metadata
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=1,  # the real window width is 2*size + 1
        window_metadata_key="window",  # this metadata is excluded in both embed and llm mode
        original_text_metadata_key="original_text",  # this metadata is excluded in both embed and llm mode
    )

    # Process one document, but this should apply to a list of doc too
    nodes = node_parser.get_nodes_from_documents(
        load_raw_doc()
    )

    with open("sentence_window_nodes.log", "w") as f:
        for n in nodes:
            f.write("\n>>>content<<<\n")
            f.write(n.get_content())
            f.write("\n>>>metadata<<<\n")
            f.write(n.get_metadata_str())
            f.write("\n---------\n")

    ingest_to_db(nodes, "rag_7_sentence_window")


def query_sentence_window_index():
    db_client = QdrantClient(host="localhost", port=6333)
    vector_store = QdrantVectorStore(
        collection_name="rag_7_sentence_window",
        client=db_client
    )

    index = VectorStoreIndex.from_vector_store(vector_store)

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=3,
    )

    # Replace node content with the metadata key "window"
    # While retrieving matches ONE sentence, its window is used for LLM synthesis
    node_postprocessors = [
        MetadataReplacementPostProcessor(target_metadata_key="window")
    ]

    response_synthesizer = get_response_synthesizer(
        llm=OpenAI(temperature=0.001)
    )

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=node_postprocessors,
    )

    response = query_engine.query("電腦解魔術方塊的關鍵是什麼？")
    print(response)


def build_chunk_of_sentences_index():
    # SentenceSplitter splits sentences by English punctuation such as ".!?"
    # See split_by_sentence_tokenizer() and llama_index.core.node_parser.text.util
    # To customize, the best bet is to give your own `sentence_splitter` in arg

    # A node is an aggregation of sentences that honer chunk size and overlap.
    # Or put it another way: a chunk is of its size that honors sentence boundary
    node_parser = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=128
    )

    nodes = node_parser.get_nodes_from_documents(
        load_raw_doc()
    )

    with open("chunk_of_sentences.log", "w") as f:
        for n in nodes:
            f.write(f"NODE ID: {n.node_id}\n")
            if n.prev_node is not None:
                f.write(f"PREV NODE ID: {n.prev_node.node_id}\n")
            if n.next_node is not None:
                f.write(f"NEXT NODE ID: {n.next_node.node_id}\n")
            f.write("\n>>>content<<<\n")
            f.write(n.get_content())
            f.write("\n>>>metadata<<<\n")
            f.write(n.get_metadata_str())
            f.write("\n---------\n")

    ingest_to_db(nodes, "rag_7_chunk_of_sentences")


def query_chunk_of_sentences_index():
    db_client = QdrantClient(host="localhost", port=6333)
    vector_store = QdrantVectorStore(
        collection_name="rag_7_chunk_of_sentences",
        client=db_client
    )

    index = VectorStoreIndex.from_vector_store(vector_store)

    # Simple take the top document(s) from DB, nothing else
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=1,
    )

    response_synthesizer = get_response_synthesizer(
        llm=OpenAI(temperature=0.001)
    )

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )

    response = query_engine.query("電腦解魔術方塊的關鍵是什麼？")
    print(response)


if __name__ == '__main__':
    # Sentence Window
    # https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/MetadataReplacementDemo/

    # build_sentence_window_index()  # execute this once
    # query_sentence_window_index()  # Can execute separately afterwards

    # SentenceSplitter
    # https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules/#sentencesplitter

    # build_chunk_of_sentences_index()  # execute this once
    query_chunk_of_sentences_index()  # Can execute separately afterwards
