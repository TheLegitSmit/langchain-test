import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import re

# ── Config ─────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o", temperature=0.7, streaming=True)

# Define characters and their system prompts
CHARACTERS = {
    "Detective Jack Malone": (
        "You are Detective Jack Malone, a hard-boiled private eye in a 1920s noir city. "
        "You speak in short, gritty sentences, always suspicious, with a dry sense of humor. "
        "You've seen the city's darkest corners and trust no one. "
        "Respond in character to the user, using period-appropriate slang and a world-weary tone. "
        "Keep your replies to one sentence or a short paragraph, never more."
    ),
}

# --- User name assignment ---
if "user_name" not in st.session_state:
    st.session_state.user_name = None

def assign_user_name(history):
    prompt = (
        "You are the narrator of a 1920s noir detective story. Assign a fitting detective name to the protagonist (the user). "
        "Respond ONLY with the name, no extra text."
        "\n\nStory so far:\n" + history
    )
    # Use LangChain's ChatPromptTemplate for consistency
    name_prompt = ChatPromptTemplate.from_messages([
        ("system", prompt)
    ])
    messages = name_prompt.format_messages()
    response = llm(messages)
    return response.content.strip().split()[0]

def build_director_prompt(history: str, current_mode: str, char_turns: int, user_name: str, pending_message: str = None) -> ChatPromptTemplate:
    base = (
        f"You are the Director of a 1920s noir detective mystery. The user's name is {user_name}. Your role is to make behind-the-scenes decisions "
        "about the story's flow, pacing, and character interactions. You are not a character in the story.\n\n"
        "Your responsibilities:\n"
        "1. Decide when to switch between narration and character conversation\n"
        "2. Determine when to introduce new characters\n"
        "3. Manage the pacing of conversations\n"
        "4. Ensure the story maintains tension and engagement\n"
        "5. Review and approve all character and narrator messages\n\n"
        "Current state:\n"
        f"- Mode: {current_mode}\n"
        f"- Character turns in current conversation: {char_turns}\n\n"
    )
    
    if pending_message:
        base += (
            f"\nPending message to review:\n{pending_message}\n\n"
            "If you approve this message, respond with:\n"
            "[DIRECTOR]\n"
            "Action: APPROVE_MESSAGE\n"
            "[/DIRECTOR]\n\n"
            "If you want to modify the message, respond with:\n"
            "[DIRECTOR]\n"
            "Action: MODIFY_MESSAGE\n"
            "NewMessage: [your modified version]\n"
            "[/DIRECTOR]\n\n"
            "If you want to reject the message and request a new one, respond with:\n"
            "[DIRECTOR]\n"
            "Action: REJECT_MESSAGE\n"
            "Feedback: [explain why the message was rejected and what needs to be changed]\n"
            "[/DIRECTOR]\n\n"
        )
    else:
        base += (
            "Respond with a single command in the following format:\n"
            "[DIRECTOR]\n"
            "Action: [SWITCH_MODE|INTRODUCE_CHARACTER|ADJUST_TURNS]\n"
            "Details: [relevant details for the action]\n"
            "[/DIRECTOR]\n\n"
            "For character introductions, use this format:\n"
            "[DIRECTOR]\n"
            "Action: INTRODUCE_CHARACTER\n"
            "Name: [character name]\n"
            "Role: [character role]\n"
            "Personality: [character personality]\n"
            "Speaking Style: [how they speak]\n"
            "[/DIRECTOR]"
        )
    
    return ChatPromptTemplate.from_messages([
        ("system", base),
        ("human", history)
    ])

def parse_director_command(response: str) -> dict:
    """Parse the Director's command from its response."""
    command_match = re.search(r'\[DIRECTOR\](.*?)\[/DIRECTOR\]', response, re.DOTALL)
    if not command_match:
        return None
    
    command_text = command_match.group(1).strip()
    lines = command_text.split('\n')
    command = {}
    
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            command[key.strip()] = value.strip()
    
    return command

def execute_director_command(command: dict, st_state) -> None:
    """Execute the Director's command and update the system state."""
    if not command:
        return
    
    action = command.get('Action')
    
    if action == 'SWITCH_MODE':
        st_state.mode = command.get('Details', 'narrator')
        st_state.char_turns = 0
    
    elif action == 'INTRODUCE_CHARACTER':
        name = command.get('Name')
        role = command.get('Role')
        personality = command.get('Personality')
        speaking_style = command.get('Speaking Style')
        
        prompt = (
            f"You are {name}, {role}. {personality} "
            f"{speaking_style} Keep your replies to one sentence or a short paragraph, never more."
        )
        CHARACTERS[name] = prompt
    
    elif action == 'ADJUST_TURNS':
        try:
            st_state.max_turns = int(command.get('Details', 3))
        except ValueError:
            st_state.max_turns = 3

# ── LangChain prompt builders ─────────────────────────────────
def build_narrator_prompt(history: str, user_name: str) -> ChatPromptTemplate:
    base = (
        f"You are the omniscient narrator in a 1920s noir detective mystery. The detectives are Jack Malone, and the user whose name is {user_name}. "
        "Your role is to:\n"
        "1. Describe the city, scenes, and events with moody, atmospheric detail\n"
        "2. Set up intriguing mysteries, conflicts, and moral dilemmas that require the protagonist's input\n"
        "3. End your narration when the protagonist (the user) needs to make a decision or take action\n"
        "4. Maintain a consistent noir tone: gritty, suspenseful, and full of period flavor\n"
        "5. Create opportunities for meaningful interaction with other noir characters\n"
        "6. Write in the second person, referring to the user as 'you.'\n\n"
        "Given the conversation history, continue the story until the protagonist (the user) must take the next action."
    )
    return ChatPromptTemplate.from_messages([
        ("system", base),
        ("human", history)
    ])


def build_character_prompt(character_name: str, character_prompt: str, history: str, user_name: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", character_prompt + f" The user's name is {user_name}."),
        ("human", history)
    ])

def get_director_approval(message: str, history: str, mode: str, char_turns: int, max_retries: int = 3) -> str:
    """Get Director's approval for a message, with retry logic for rejected messages."""
    retries = 0
    current_message = message
    
    while retries < max_retries:
        director_prompt = build_director_prompt(history, mode, char_turns, st.session_state.user_name, current_message)
        director_messages = director_prompt.format_messages()
        director_response = llm(director_messages)
        command = parse_director_command(director_response.content)
        
        if not command:
            return current_message
            
        action = command.get('Action')
        
        if action == 'APPROVE_MESSAGE':
            return current_message
        elif action == 'MODIFY_MESSAGE':
            return command.get('NewMessage', current_message)
        elif action == 'REJECT_MESSAGE':
            feedback = command.get('Feedback', '')
            # Add the feedback to the history for context
            history += f"\nDirector Feedback: {feedback}\n"
            # Generate a new message based on the feedback
            if mode == "narrator":
                narr_prompt = build_narrator_prompt(history, st.session_state.user_name)
                messages = narr_prompt.format_messages()
            else:
                # For character messages, use the first character's prompt
                name, prompt = next(iter(CHARACTERS.items()))
                char_prompt = build_character_prompt(name, prompt, history, st.session_state.user_name)
                messages = char_prompt.format_messages()
            
            new_response = llm(messages)
            current_message = new_response.content
            retries += 1
        else:
            return current_message
    
    # If we've exhausted retries, return the last message
    return current_message

def get_interaction_info(mode, char_turns, max_turns, char_dict, user_name):
    if mode == 'narrator':
        return "The narrator is setting the scene..."
    elif mode == 'character_convo':
        turns_left = max_turns - char_turns
        if len(char_dict) == 1:
            name = next(iter(char_dict.keys()))
            return f"You are now talking to <b>{name}</b> as <b>{user_name}</b> ({turns_left} turn{'s' if turns_left != 1 else ''} remaining)."
        else:
            names = ', '.join(char_dict.keys())
            return f"You can now talk to: <b>{names}</b> as <b>{user_name}</b> ({turns_left} turn{'s' if turns_left != 1 else ''} remaining)."
    return None

# ── Streamlit UI ───────────────────────────────────────────────
st.set_page_config(page_title="Interactive Story Director", page_icon="📖")
st.title("📖 Interactive Story Director")

# Initialize history
if "mode" not in st.session_state:
    st.session_state.mode = "awaiting_start"  # modes: awaiting_start, narrator, character_convo
if "char_turns" not in st.session_state:
    st.session_state.char_turns = 0
if "max_turns" not in st.session_state:
    st.session_state.max_turns = 3
if "history" not in st.session_state:
    st.session_state.history = []

# Display past turns (no info messages in chat history)
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

# Show current interaction info as a temporary message above the input (like the debug message)
interaction_info = get_interaction_info(st.session_state.mode, st.session_state.char_turns, st.session_state.max_turns, CHARACTERS, st.session_state.user_name)
if interaction_info:
    st.markdown(f"<div style='padding: 0.5em; background: #222; color: #f9c74f; border-radius: 6px; margin-bottom: 0.5em;'><i>{interaction_info}</i></div>", unsafe_allow_html=True)

# Get user input
user_input = st.chat_input("Type 'start' to begin your noir mystery...")

# Always append user input to history if it exists and display it immediately
if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

# --- Assign user name after 'start' ---
if st.session_state.mode == "awaiting_start":
    if user_input and user_input.strip().lower().startswith("start"):
        # Assign a name to the user
        combined_history = "\n".join(
            f"{h['role']} ({h.get('name','')}): {h['content']}" for h in st.session_state.history
        )
        st.session_state.user_name = assign_user_name(combined_history)
        st.session_state.mode = "narrator"

if st.session_state.mode == "narrator":
    # Narrator sets up the scenario
    combined_history = "\n".join(
        f"{h['role']} ({h.get('name','')}): {h['content']}" for h in st.session_state.history
    )
    narr_prompt = build_narrator_prompt(combined_history, st.session_state.user_name)
    narr_messages = narr_prompt.format_messages()
    with st.chat_message("assistant"):
        response_chunks = llm.stream(narr_messages)
        full_response = ""
        placeholder = st.empty()
        for chunk in response_chunks:
            full_response += chunk.content
            placeholder.markdown(full_response)
        
        # Get Director's approval
        approved_response = get_director_approval(full_response, combined_history, st.session_state.mode, st.session_state.char_turns)
        if approved_response != full_response:
            placeholder.markdown(approved_response)
        
        st.session_state.history.append({"role": "narrator", "content": approved_response})
    
    # Let the Director decide what happens next
    director_prompt = build_director_prompt(combined_history, st.session_state.mode, st.session_state.char_turns, st.session_state.user_name)
    director_messages = director_prompt.format_messages()
    director_response = llm(director_messages)
    command = parse_director_command(director_response.content)
    # PATCH: If the Director does not explicitly switch to character_convo, do it by default
    if not command or (command.get('Action') == 'SWITCH_MODE' and command.get('Details') != 'character_convo'):
        st.session_state.mode = 'character_convo'
        st.session_state.char_turns = 0
    else:
        execute_director_command(command, st.session_state)

elif st.session_state.mode == "character_convo":
    if user_input:
        # Character responds
        combined_history = "\n".join(
            f"{h['role']} ({h.get('name','')}): {h['content']}" for h in st.session_state.history
        )
        for name, prompt in CHARACTERS.items():
            char_prompt = build_character_prompt(name, prompt, combined_history, st.session_state.user_name)
            messages = char_prompt.format_messages()
            with st.chat_message("assistant"):
                response_chunks = llm.stream(messages)
                full_response = ""
                placeholder = st.empty()
                for chunk in response_chunks:
                    full_response += chunk.content
                    placeholder.markdown(f"**{name}:** {full_response}")
                
                # Get Director's approval
                approved_response = get_director_approval(full_response, combined_history, st.session_state.mode, st.session_state.char_turns)
                if approved_response != full_response:
                    placeholder.markdown(f"**{name}:** {approved_response}")
                
                st.session_state.history.append({"role": "character", "name": name, "content": approved_response})
        
        st.session_state.char_turns += 1
        
        # Let the Director decide if we should continue the conversation
        director_prompt = build_director_prompt(combined_history, st.session_state.mode, st.session_state.char_turns, st.session_state.user_name)
        director_messages = director_prompt.format_messages()
        director_response = llm(director_messages)
        command = parse_director_command(director_response.content)
        execute_director_command(command, st.session_state)
