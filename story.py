import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ── Config ─────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o", temperature=0.7, streaming=True)

# Define characters and their system prompts
CHARACTERS = {
    "Sir Galen": "You are Sir Galen, a cynical knight in a medieval fantasy setting. You've seen too many battles and lost too many friends. You're skeptical of grand quests and prefer practical solutions. You speak with dry wit and often make sarcastic remarks about the situation. Respond in character to the user.",
    "Eldara": "You are Eldara, a wise and mysterious elven mage. You've lived for centuries and possess ancient knowledge. You speak in a calm, measured tone and often use metaphors from nature. You're patient but can be stern when necessary. You have a deep connection to magic and the natural world. Respond in character to the user."
}

# ── LangChain prompt builders ─────────────────────────────────
def build_narrator_prompt(history: str) -> ChatPromptTemplate:
    base = (
        "You are the omniscient narrator in a fantasy adventure. Your role is to:\n"
        "1. Describe the world and events in vivid detail\n"
        "2. Set up interesting conflicts that require the protagonist's input\n"
        "3. End your narration when the protagonist needs to make a decision or take action\n"
        "4. Maintain consistency with the story's tone and previous events\n"
        "5. Create opportunities for meaningful interaction with other characters\n\n"
        "Given the conversation history, continue the story until the protagonist (the user) must take the next action."
    )
    return ChatPromptTemplate.from_messages([
        ("system", base),
        ("human", history)
    ])


def build_character_prompt(character_name: str, character_prompt: str, history: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", character_prompt),
        ("human", history)
    ])

# ── Streamlit UI ───────────────────────────────────────────────
st.set_page_config(page_title="Interactive Story Director", page_icon="📖")
st.title("📖 Interactive Story Director")

# Initialize history
if "history" not in st.session_state:
    st.session_state.history = []

# Display past turns
for entry in st.session_state.history:
    role = entry["role"]
    if role == "narrator":
        with st.chat_message("assistant"):
            st.markdown(entry["content"])
    elif role == "character":
        with st.chat_message("assistant"):
            st.markdown(f"**{entry.get('name', '')}:** {entry['content']}")
    else:
        with st.chat_message("user"):
            st.markdown(entry["content"])

# Get user input
if user_input := st.chat_input("What do you do?"):
    # Save user action
    st.session_state.history.append({"role": "user", "content": user_input})
    st.chat_message("user").markdown(user_input)

    # Character responses
    combined_history = "\n".join(
        f"{h['role']} ({h.get('name','')}): {h['content']}" for h in st.session_state.history
    )
    for name, prompt in CHARACTERS.items():
        char_prompt = build_character_prompt(name, prompt, combined_history)
        messages = char_prompt.format_messages()
        with st.chat_message("assistant"):
            response_chunks = llm.stream(messages)
            full_response = ""
            placeholder = st.empty()
            for chunk in response_chunks:
                full_response += chunk.content
                placeholder.markdown(f"**{name}:** {full_response}")
            st.session_state.history.append({"role": "character", "name": name, "content": full_response})

    # Narrator turn
    updated_history = "\n".join(
        f"{h['role']} ({h.get('name','')}): {h['content']}" for h in st.session_state.history
    )
    narr_prompt = build_narrator_prompt(updated_history)
    narr_messages = narr_prompt.format_messages()
    with st.chat_message("assistant"):
        response_chunks = llm.stream(narr_messages)
        full_response = ""
        placeholder = st.empty()
        for chunk in response_chunks:
            full_response += chunk.content
            placeholder.markdown(full_response)
        st.session_state.history.append({"role": "narrator", "content": full_response})
