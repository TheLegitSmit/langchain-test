#!/usr/bin/env python3
"""
multitool_demo.py — Streamlit GUI for LangChain Multitool Demo

Demonstrates combined LangChain features via multiple demo modes:
- Simple Chain
- Retrieval QA (RAG)
- Memory + Retrieval
- Memory + Agent
- Memory + RAG Augment (retrieval augments general knowledge)

Prerequisites:
    pip install streamlit langchain-openai langchain-core python-dotenv chromadb requests
"""

from dotenv import load_dotenv
load_dotenv()  # pull OPENAI_API_KEY from .env

import os
import streamlit as st
import requests

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import LLMChain, RetrievalQA, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool, AgentType

# --- Config and state ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
st.set_page_config(page_title="LangChain Multitool Demo", page_icon="🤖")
st.title("🤖 LangChain Multitool Demo")

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.selectbox("Model", ("gpt-4o-mini","gpt-4.1-mini","gpt-4o","o3-mini"), index=0)
    temperature = st.slider("Creativity", 0.0, 2.0, 0.7, 0.1)
    system_prompt = st.text_area("System prompt","You are an AI consultant.", height=100)
    demo_mode = st.selectbox("Demo Mode", (
        "Simple Chain",
        "Retrieval QA",
        "Memory + Retrieval",
        "Memory + Agent",
        "Memory + RAG Augment"
    ))

# Initialize model
llm = ChatOpenAI(
    model=model_name,
    temperature=temperature,
    openai_api_key=OPENAI_API_KEY,
    streaming=False
)

# 1. Simple chain
simple_prompt = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}"),
    ("human", "{user_input}")
])
simple_chain = LLMChain(llm=llm, prompt=simple_prompt)

# 2. Retrieval QA setup (RAG)
if not os.path.exists("data.txt"):
    with open("data.txt","w") as f:
        f.write("LangChain is a framework for developing LLM apps.")
loader = TextLoader("data.txt")
docs = loader.load()
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectordb = Chroma.from_documents(docs, embeddings)
retriever = vectordb.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

# 3. Conversation + memory
if "conv_memory" not in st.session_state:
    st.session_state.conv_memory = ConversationBufferMemory()
conv_chain = ConversationChain(llm=llm, memory=st.session_state.conv_memory)

# 4. Agent + tool

def fetch_cat_fact(text: str) -> str:
    res = requests.get("https://catfact.ninja/fact")
    return res.json().get("fact","Error fetching cat fact.")
cat_tool = Tool(name="cat_fact", func=fetch_cat_fact, description="Get a random cat fact.")
agent = initialize_agent(
    tools=[cat_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False
)

# --- Handle user input ---
if user_input := st.text_input("Your input:"):
    st.markdown(f"**You:** {user_input}")

    if demo_mode == "Simple Chain":
        response = simple_chain.run({"system_prompt": system_prompt, "user_input": user_input})

    elif demo_mode == "Retrieval QA":
        response = qa_chain.run(user_input)

    elif demo_mode == "Memory + Retrieval":
        # use memory for context and retrieval for facts
        # first retrieval result
        fact = qa_chain.run(user_input)
        # then conversation with memory
        response = conv_chain.predict(input=f"{fact}\nUser: {user_input}")

    elif demo_mode == "Memory + Agent":
        # agent run inside a conversation chain
        tool_resp = agent.run(user_input)
        response = conv_chain.predict(input=f"{tool_resp}")

    elif demo_mode == "Memory + RAG Augment":
        # fetch relevant docs
        docs = retriever.get_relevant_documents(user_input)
        context = "\n".join([d.page_content for d in docs])
        # augment the question
        augmented = f"Context:\n{context}\n\n{user_input}"
        response = conv_chain.predict(input=augmented)

    else:
        response = "Unknown demo mode."

    st.markdown(f"**Bot:** {response}")
