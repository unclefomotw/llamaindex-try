#!/usr/bin/env bash
set -o nounset
set -o errexit
set -o pipefail

### run only once
# python src/rag_bot_1/etl.py

### launch streamlit for UI chat interaction
PYTHONPATH=$(pwd) streamlit run src/rag_bot_1/app.py 
