# export OPENAI_API_KEY and RAG_DATA_DIR first
# Based on rag_1, customize LLM and prompts
import logging
import os
import sys

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.llms.openai import OpenAI

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.prompts.base import ChatPromptTemplate


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


def main():
    if not os.path.exists(PERSIST_DIR):
        index = build_index()
    else:
        index = load_index()

    # Can use a different model
    synthesis_llm = OpenAI(model="gpt-4-turbo", temperature=0)

    # Customize my own prompt
    # ref: from retriever_query_engine.py to chat_prompts.py
    query_engine = index.as_query_engine(
        llm=synthesis_llm,
        text_qa_template=MY_QA_PROMPT,  # magic kwargs defined in retriever_query_engine.py
    )
    response = query_engine.query("推薦給新手哪個型號的魔術方塊？")
    print(response)

    # Another way to set the prompt of a query engine:
    # query_engine.update_prompts({
    #    "response_synthesizer:text_qa_template": MY_QA_PROMPT
    # })


if __name__ == '__main__':
    main()
