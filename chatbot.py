# chatbot.py
# CLI chatbot using LangChain + GPT-4o-mini
# ----------------------------------------

from dotenv import load_dotenv

# 1 · Load environment variables from .env (OPENAI_API_KEY, etc.)
load_dotenv()                     # looks for a file named .env in this folder

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage


def main() -> None:
    # 2 · Model wrapper
    llm = ChatOpenAI(
        model="gpt-4o-mini",      # change to "gpt-4.1-mini" or "o3-mini" if you like
        temperature=0.7,          # for o3-mini set to 1 or 0
        streaming=True            # stream tokens as they arrive
    )

    # 3 · Prompt template
    system_msg = SystemMessage(content="You are a helpful assistant.")
    history_placeholder = MessagesPlaceholder(variable_name="history")

    prompt = ChatPromptTemplate.from_messages(
        [system_msg, history_placeholder, ("human", "{user_input}")]
    )

    # 4 · Chat loop with simple in-memory history
    chat_history: list[tuple[str, str]] = []

    print("Chatbot ready (type 'exit' or 'quit' to leave).")
    while True:
        user_input = input("> ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        chain = prompt | llm
        response = chain.invoke(
            {"user_input": user_input, "history": chat_history}
        )

        print(response.content)

        chat_history.extend(
            [("human", user_input), ("ai", response.content)]
        )


if __name__ == "__main__":
    # Ensure the key is set; give a clear error if not.
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is missing. "
            "Add it to a .env file or export it in the shell."
        )
    main()
