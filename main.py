#!/usr/bin/env python3
"""
main.py — Streamlit GUI for LangChain Proof of Concept

Prereqs:
    pip install streamlit langchain langchain-community langchain-openai \
                python-dotenv chromadb requests google-search-results
"""

# ── 0 · env vars ─────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()  # loads OPENAI_API_KEY and SERPAPI_API_KEY from .env

import os
import requests
import streamlit as st

# ── 1 · LangChain imports ───────────────────────────────
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import LLMChain, RetrievalQA, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_community.utilities import SerpAPIWrapper  # for internet search :contentReference[oaicite:1]{index=1}

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ── 2 · Streamlit page config ────────────────────────────
st.set_page_config(page_title="LangChain POC", page_icon="🤖")
st.title("🤖 LangChain Proof of Concept")

with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.selectbox(
        "Model",
        ("gpt-4o-mini", "gpt-4.1-mini", "gpt-4o", "o3-mini"),
        index=0
    )
    temperature = st.slider("Creativity", 0.0, 2.0, 0.7, 0.1)
    system_prompt = st.text_area(
        "System prompt",
        "You are a helpful assistant that provides concise yet accurate information.",
        height=100
    )
    demo_mode = st.selectbox(
        "Demo Mode",
        (
            "Simple Chain",
            "Retrieval QA",
            "Chat with Memory",
            "Agent (Cat Fact)",
            "Internet Search + Memory",
            "Site Search + Memory"
        )
    )

# ── 3 · Chat model ───────────────────────────────────────
llm = ChatOpenAI(
    model=model_name,
    temperature=temperature,
    openai_api_key=OPENAI_API_KEY,
    streaming=False
)

# ── 4 · Simple chain ─────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}"),
    ("human", "{user_input}")
])
simple_chain = LLMChain(llm=llm, prompt=prompt)

# ── 5 · Retrieval-QA setup ──────────────────────────────
for fname, sample in [("data.txt", "LangChain is a framework for developing applications powered by language models."),
                      ("file.txt", "File.txt sample content for second document."),
                      ("story.txt", "File.txt sample content for third document.")]:
    if not os.path.exists(fname):
        with open(fname, "w") as f:
            f.write(sample)

docs = []
for fname in ["data.txt", "file.txt", "story.txt"]:
    docs += TextLoader(fname).load()

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectordb = Chroma.from_documents(docs, embeddings)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectordb.as_retriever()
)

# ── 6 · Conversation w/ memory ──────────────────────────
if "memory_buffer" not in st.session_state:
    st.session_state.memory_buffer = ConversationBufferMemory()
conv_chain = ConversationChain(
    llm=llm,
    memory=st.session_state.memory_buffer
)

# ── 7 · Agent demo (cat fact) ───────────────────────────
def get_cat_fact(_: str) -> str:
    res = requests.get("https://catfact.ninja/fact")
    return (
        res.json().get("fact", "Could not fetch a cat fact.")
        if res.ok
        else "API error."
    )

cat_tool = Tool(
    name="cat_fact",
    func=get_cat_fact,
    description="Get a random cat fact."
)
agent = initialize_agent(
    tools=[cat_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False
)

# ── 8 · Internet Search + Memory setup ──────────────────
search = SerpAPIWrapper(serpapi_api_key=os.getenv("SERPAPI_API_KEY"))
search_tool = Tool(
    name="Search",
    func=search.run,
    description="Useful for answering questions about current events."
)
memory_search = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True
)
internet_agent = initialize_agent(
    tools=[search_tool],
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory_search,
    verbose=False
)

# ── 9 · Site-Specific Search + Memory setup ─────────────
SITE = "https://howyoutravel.com/"  # ← replace with your site
site_search_tool = Tool(
    name="SiteSearch",
    func=lambda q: search.run(f"site:{SITE} {q}"),
    description=f"Search only the {SITE} website."
)
memory_site = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True
)
site_agent = initialize_agent(
    tools=[site_search_tool],
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory_site,
    verbose=False
)

# ── 10 · Main UI input/output ────────────────────────────
if user_input := st.text_input("Your input:"):
    st.markdown(f"**You:** {user_input}")

    if demo_mode == "Simple Chain":
        response = simple_chain.run({
            "system_prompt": system_prompt,
            "user_input": user_input
        })
    elif demo_mode == "Retrieval QA":
        response = qa_chain.run(user_input)
    elif demo_mode == "Chat with Memory":
        response = conv_chain.predict(input=user_input)
    elif demo_mode == "Agent (Cat Fact)":
        response = agent.run(user_input)
    elif demo_mode == "Internet Search + Memory":
        response = internet_agent.run(user_input)
    elif demo_mode == "Site Search + Memory":
        response = site_agent.run(user_input)
    else:
        response = "Demo mode not recognized."

    st.markdown(f"**Bot:** {response}")
