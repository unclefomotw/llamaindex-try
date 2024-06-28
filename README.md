# llamaindex-try
random attempts in learning llamaindex

## Prerequisites

### Python dependencies

1. (Optional) set up virtual environment.  There are many ways.  Here's an example
```bash
$ python -m venv venv
$ source venv/bin/activate
```
2. Install dependencies
```bash
$ pip install -r requirements.txt
```

### Install Qdrant database

1. Install Docker
2. Follow https://qdrant.tech/documentation/quick-start/ to install Qdrant

### Environment variables

1. Set up your OpenAI API key

```bash
$ export OPENAI_API_KEY="sk-..."
```

2. Prepare your own dataset to be retrieved, and then set the environment variable pointing to the directory

```bash
$ export RAG_DATA_DIR="<path to data>"
```

My code assumes a bunch of .md files, but you can customize by messing around with `SimpleDirectoryReader` in the code

## Execution

This repo is not a package.  This repo contains separated codes, each of which representing a use case or a scenario.

To run it, simply run in the project directory and

```bash
$ PYTHONPATH=$(pwd) python src/<code.py>
```

### Run Qdrant

Some codes require Qdrant.  After you install Qdrant, you can use `./run-qdrant.sh` to start it.

### Run "rag_bot_1"

To run `src/rag_bot_1`, which runs a RAG Chat bot using **streamlit**, simple run `./run-rag_bot_1.sh`

---

## Try RAG
* rag_1.py - use the [start example](https://docs.llamaindex.ai/en/stable/getting_started/starter_example/) with my own documents in markdown
* rag_2.py - Customize LLM and prompts
* rag_3.py - Use a vector database (Qdrant) + Data ingestion
* rag_4.py - Create the query engine in a slightly lower level way
* rag_5.py - Instrumentation and customized event handler
* rag_6.py - Use an embedding model on HuggingFace
* rag_7.py - Use different chunking/indexing strategies
* rag_8.py - Use IngestionPipeline to parse doc and feed nodes into vector db
* rag_bot_1 - The most simple RAG chat app using Streamlit as UI

