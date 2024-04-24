import os

import llama_index.core.instrumentation as instrument
import streamlit as st
from llama_index.core import (
    Settings,
    StorageContext,
    get_response_synthesizer,
    load_index_from_storage,
)
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.prompts.base import ChatPromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from src.my_handler import NaiveEventHandler
from src.rag_bot_1.config import INDEX_PERSIST_DIR

dispatcher = instrument.get_dispatcher()
dispatcher.add_event_handler(NaiveEventHandler(log_file="logs/rag_bot_1.log"))

EMBEDDING_MODEL = OpenAIEmbedding(model="text-embedding-3-small")
Settings.embed_model = EMBEDDING_MODEL


TEXT_QA_SYSTEM_PROMPT = ChatMessage(
    content=(
        "You are an expert Q&A system that is trusted around the world.\n"
        "Always answer the query using the provided context information, "
        "and not prior knowledge.\n"
        "Some rules to follow:\n"
        "1. Never directly reference the given context in your answer.\n"
        "2. Avoid statements like 'Based on the context, ...' or "
        "'The context information ...' or anything along "
        "those lines.\n"
        "3. Detect the language of the question, and always answer in that language"
    ),
    role=MessageRole.SYSTEM,
)

TEXT_QA_PROMPT_TMPL_MSGS = [
    TEXT_QA_SYSTEM_PROMPT,
    ChatMessage(
        content=(
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, "
            "answer the query.\n"
            "Query: {query_str}\n"
            "Answer: "
        ),
        role=MessageRole.USER,
    ),
]

MY_QA_PROMPT = ChatPromptTemplate(message_templates=TEXT_QA_PROMPT_TMPL_MSGS)


@st.cache_resource(show_spinner=False)
def load_index():
    if not os.path.exists(INDEX_PERSIST_DIR):
        raise IOError("Need to build the index first.")

    with st.spinner(text="Loading my knowledge..."):
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_PERSIST_DIR)
        index = load_index_from_storage(storage_context)

    return index


def get_query_engine():
    index = load_index()

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=5,
    )
    synthesis_llm = OpenAI(temperature=0.0001)
    response_synthesizer = get_response_synthesizer(
        llm=synthesis_llm,
        text_qa_template=MY_QA_PROMPT,
        # streaming=True  # Somehow streaming conflicts with instrumentation
    )

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer
    )
    return query_engine
