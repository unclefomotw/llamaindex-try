# export OPENAI_API_KEY and RAG_DATA_DIR first

# Use EventHandler to capture various events from start to finish
# This helps inspect and debug
# Need to "pip install treelib" to print tree-like traces

import os
from typing import Optional

import llama_index.core.instrumentation as instrument
from llama_index.core import Settings
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    get_response_synthesizer,
    load_index_from_storage
)
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.events import BaseEvent
from llama_index.core.instrumentation.span_handlers import SimpleSpanHandler
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

RAG_DATA_DIR = os.environ["RAG_DATA_DIR"]
PERSIST_DIR = "./storage/rag_4"

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")


class NaiveEventHandler(BaseEventHandler):
    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "NaiveEventHandler"

    @classmethod
    def _get_pretty_dict_str(
            cls,
            _dict: dict,
            skip_keys: Optional[list] = None,
            indent_str: str = ""
    ) -> str:
        _skip_keys = skip_keys or []

        ret = ""
        for _k, _v in _dict.items():
            if _k in _skip_keys:
                continue
            ret += f"{indent_str}{_k}: {_v}\n"
        return ret

    @classmethod
    def _get_pretty_even_str(cls, event: BaseEvent) -> str:
        _indent = "    "
        ret = ""

        for ek, ev in event.dict().items():
            if ek == "model_dict":
                # dict
                ret += f"{ek}:\n"
                ret += cls._get_pretty_dict_str(
                    ev, skip_keys=["api_key"], indent_str=_indent
                )
            elif ek == "embeddings":
                # List[List[float]]
                ret += f"{ek}: "
                ret += ",".join([f"<{len(_embedding)}-dim>" for _embedding in ev])
                ret += "\n"
            elif ek == "nodes":
                # List[NodeWithScore]
                # NodeWithScore is still too long; cannot think of a good repr in pure text
                ret += f"{ek}:\n"
                for _n in ev:
                    ret += f"{_indent}{_n}\n"
            elif ek == "messages":
                # List[ChatMessage]
                ret += f"{ek}:\n"
                for _n in ev:
                    ret += f"{_indent}{_n}\n"
            else:
                ret += f"{ek}: {ev}\n"

        return ret

    def handle(self, event: BaseEvent, **kwargs):
        """Logic for handling event."""
        with open("log.txt", "a") as f:
            f.write(self._get_pretty_even_str(event))
            f.write("\n")


dispatcher = instrument.get_dispatcher()  # only works in the root dispatcher ??
dispatcher.add_event_handler(NaiveEventHandler())
span_handler = SimpleSpanHandler()
dispatcher.add_span_handler(span_handler)


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
    index = VectorStoreIndex.from_documents(documents)

    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)

    return index


def load_index():
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    # When loading, need to specify the embedding model information again
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

    # Need to pip install treelib
    print("\n---vvv Time Used vvv---")
    span_handler.print_trace_trees()


if __name__ == '__main__':
    main()
