# app.py — Streamlit GUI for LangChain chatbot
# -------------------------------------------
# Prereqs:
#   pip install streamlit langchain langchain-openai python-dotenv

from dotenv import load_dotenv
load_dotenv()                 # <─ pulls OPENAI_API_KEY from .env

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ── 0 · Page basics ────────────────────────────────────────────
st.set_page_config(page_title="Chitty Chatty Bot", page_icon="💬")  # title + favicon
st.title("💬 Chitty Chatty Bot")                                    # big header

# ── 1 · Sidebar controls (model, temperature, system prompt) ──
with st.sidebar:
    st.header("⚙️ Settings")

    model_name = st.selectbox(
        "Model",
        ("gpt-4o-mini", "gpt-4.1-mini", "gpt-4o", "o3-mini"),          # add more as you like
        index=0
    )

    temperature = st.slider(
        "Creativity",
        min_value=0.0,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Fuck you, you want some of this? Come and get it."
    )

    system_prompt = st.text_area(
        "System prompt",
        "You are a helpful assistant.",
        height=80
    )

# ── 2 · LangChain setup (uses sidebar values) ─────────────────
llm = ChatOpenAI(
    model=model_name,
    temperature=temperature,
    streaming=True
)

prompt_tmpl = ChatPromptTemplate.from_messages(
    [system_prompt, MessagesPlaceholder("history"), ("human", "{user_input}")]
)

# ── 3 · Session-state chat history ────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []        # list of {"role","content"}

# ── 4 · Replay past turns so they show in the UI ──────────────
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── 5 · Handle a new user message ─────────────────────────────
if user_input := st.chat_input("Ask me anything…"):
    st.chat_message("user").markdown(user_input)   # echo user bubble

    # Build & stream the assistant’s reply
    chain = prompt_tmpl | llm
    partial_reply = ""
    with st.chat_message("assistant"):
        stream_area = st.empty()
        for chunk in chain.stream(
            {"user_input": user_input, "history": st.session_state.history}
        ):
            partial_reply += chunk.content or ""
            stream_area.markdown(partial_reply + "▌")   # live cursor
        stream_area.markdown(partial_reply)             # final text

    # Save turn to history so it persists on the next rerun
    st.session_state.history.extend([
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": partial_reply}
    ])