# export OPENAI_API_KEY and RAG_DATA_DIR first
# Do `pip install streamlit` (currently using 1.33.0)

# RAG QA (no history) + Streamlit chat UI
#   * Need to execute etl.py once to build index before everything
#   * Run ./run-rag_bot_1.sh to launch Streamlit Chat UI

import streamlit as st

from src.rag_bot_1.rag import get_query_engine


def main():
    # LlamaIndex setup

    # Streamlit
    st.title("My Chat!")

    # Init LlamaIndex RAG query engine
    if "query_engine" not in st.session_state.keys():
        st.session_state.query_engine = get_query_engine()

    # Init chat message history
    if "messages" not in st.session_state.keys():
        st.session_state.messages = []

    # Re-draw history / all messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("任何魔方的問題都可以問我唷"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        query_engine = st.session_state.query_engine
        with st.chat_message("assistant"):
            response = query_engine.query(prompt)
            # Somehow streaming conflicts with instrumentation
            # st.write_stream(streaming_response.response_gen)
            st.write(response.response)
        st.session_state.messages.append({"role": "assistant", "content": response.response})


if __name__ == '__main__':
    main()
