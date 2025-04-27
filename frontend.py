#!/usr/bin/env python3
"""
frontend.py — Streamlit UI for LangChain Multitool Demo

Contains all the UI elements and styling for the Streamlit interface.
Imports the backend logic from backend.py.
"""

import streamlit as st
import time
import json
import base64
from datetime import datetime

# IMPORTANT: Set page config must be the first Streamlit command
st.set_page_config(
    page_title="AI Assistant Hub",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import the backend class
from backend import LangChainBackend

# --- App Configuration ---
APP_TITLE = "AI Assistant Hub"  # Customize with your brand name
APP_LOGO = "🧠"  # Customize with your preferred emoji or path to logo
APP_VERSION = "1.0.0"

# --- Session State Management ---
def init_session_state():
    """Initialize all session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "backend" not in st.session_state:
        st.session_state.backend = LangChainBackend()
    
    if "demo_mode" not in st.session_state:
        st.session_state.demo_mode = "Simple Chain"

# --- Custom CSS for UI Styling ---
def apply_custom_css():
    """Apply custom CSS styling to the app"""
    st.markdown("""
    <style>
    /* Custom fonts and styling */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Main container styling */
    .main {
        background-color: #0F0F15;
        color: #F8F8F8;
        font-family: 'Inter', sans-serif;
    }
    
    /* Custom header styling */
    .stApp header {
        background-color: #1B1B26;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Custom chat container */
    .chat-container {
        margin-bottom: 100px;
        padding: 10px;
    }
    
    /* Custom user message styling */
    .user-message {
        background-color: #FF4A4A;
        color: white;
        border-radius: 20px 20px 0 20px;
        padding: 15px;
        margin: 15px 0;
        max-width: 80%;
        float: right;
        clear: both;
    }
    
    /* Custom bot message styling */
    .bot-message {
        background-color: #1B1B26;
        color: white;
        border-radius: 20px 20px 20px 0;
        padding: 15px;
        margin: 15px 0;
        max-width: 80%;
        float: left;
        clear: both;
    }
    
    /* Avatar images */
    .avatar-img {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        object-fit: cover;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #FF4A4A;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #E63E3E;
        transform: translateY(-2px);
    }
    
    /* Input field styling */
    .stTextInput>div>div>input {
        border-radius: 25px;
        border: 1px solid #1B1B26;
        padding: 15px 20px;
        background-color: #1B1B26;
        color: white;
    }
    
    /* Selectbox styling */
    .stSelectbox>div>div {
        background-color: #1B1B26;
        border-radius: 8px;
    }
    
    /* Text area styling */
    .stTextArea>div>div>textarea {
        background-color: #1B1B26;
        border-radius: 8px;
    }
    
    /* Chat input fixed at bottom */
    .chat-input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #1B1B26;
        padding: 15px 30px;
        border-top: 1px solid rgba(255,255,255,0.1);
        z-index: 100;
    }
    
    /* Mobile responsive adjustments */
    @media (max-width: 768px) {
        .user-message, .bot-message {
            max-width: 90%;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# --- UI Components ---
def create_sidebar():
    """Create and configure the sidebar with all settings"""
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection with descriptions
        st.subheader("Select AI Model")
        model_name = st.selectbox(
            "Model",
            options=("gpt-4o-mini", "gpt-4.1-mini", "gpt-4o", "o3-mini"),
            index=0,
            help="Choose the AI model powering your assistant"
        )
        
        # Model descriptions
        model_descriptions = {
            "gpt-4o-mini": "Fast and cost-effective for simple tasks",
            "gpt-4.1-mini": "Balanced performance for general use",
            "gpt-4o": "High performance for complex reasoning",
            "o3-mini": "Optimized for specialized knowledge"
        }
        st.caption(model_descriptions[model_name])
        
        # Temperature slider
        st.subheader("Creativity Level")
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="Higher values = more creative, lower = more focused"
        )
        
        # Temperature visualization
        if temperature < 0.5:
            st.caption("🧠 Precise & factual responses")
        elif temperature < 1.2:
            st.caption("💡 Balanced creativity")
        else:
            st.caption("🎨 Highly creative responses")
        
        # System prompt
        st.subheader("System Prompt")
        system_prompt = st.text_area(
            "Instructions for the AI",
            value="You are a helpful AI consultant.",
            height=100,
            help="Set instructions for how the AI should behave"
        )
        
        # Demo mode selection
        st.subheader("AI Processing Mode")
        demo_mode = st.selectbox(
            "Choose Processing Method",
            options=(
                "Simple Chain",
                "Retrieval QA",
                "Memory + Retrieval",
                "Memory + Agent",
                "Memory + RAG Augment"
            ),
            index=0,
            help="Different modes process your questions in different ways"
        )
        
        # Update session state
        st.session_state.demo_mode = demo_mode
        
        # Mode descriptions
        if demo_mode == "Simple Chain":
            st.caption("Basic Q&A without memory or external data")
        elif demo_mode == "Retrieval QA":
            st.caption("Answers questions using knowledge base")
        elif demo_mode == "Memory + Retrieval":
            st.caption("Remembers conversation + uses knowledge base")
        elif demo_mode == "Memory + Agent":
            st.caption("Remembers conversation + uses tools")
        elif demo_mode == "Memory + RAG Augment":
            st.caption("Enhanced with both memory and knowledge")
        
        # Update backend when settings change
        if st.button("Apply Settings"):
            st.session_state.backend.update_model_settings(model_name, temperature)
            st.success("Settings applied!")
        
        # Add export chat button
        if st.session_state.messages:
            if st.button("💾 Export Chat History"):
                export_chat_history()
        
        # Add clear conversation button
        if st.session_state.messages:
            if st.button("🗑️ Clear Conversation"):
                st.session_state.messages = []
                st.session_state.backend.reset_memory()
                st.rerun()
        
        # Branding in sidebar footer
        st.sidebar.markdown("---")
        st.sidebar.caption(f"© {datetime.now().year} Your Brand | v{APP_VERSION}")
        
        return model_name, temperature, system_prompt, demo_mode

def display_chat_messages():
    """Display all chat messages with custom styling"""
    # Define default avatars (replace with your own images)
    DEFAULT_USER_AVATAR = "https://via.placeholder.com/40/FF4A4A/FFFFFF?text=You"
    DEFAULT_BOT_AVATAR = "https://via.placeholder.com/40/1B1B26/FFFFFF?text=AI"
    
    # Container for messages
    with st.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Display all messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                # User message
                st.markdown(f"""
                <div style="display:flex; justify-content:flex-end; align-items:flex-start; margin-bottom:10px;">
                    <div class="user-message">
                        {message["content"]}
                    </div>
                    <div style="margin-left:10px;">
                        <img src="{DEFAULT_USER_AVATAR}" class="avatar-img">
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Bot message
                st.markdown(f"""
                <div style="display:flex; justify-content:flex-start; align-items:flex-start; margin-bottom:10px;">
                    <div style="margin-right:10px;">
                        <img src="{DEFAULT_BOT_AVATAR}" class="avatar-img">
                    </div>
                    <div class="bot-message">
                        {message["content"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

def create_chat_input():
    """Create the chat input area at the bottom of the screen"""
    # Add some space before input to prevent overlap with messages
    st.markdown("<div style='height: 80px;'></div>", unsafe_allow_html=True)
    
    # Create the chat input at the bottom
    st.markdown("""
    <div class="chat-input-container">
        <div style="max-width: 1200px; margin: 0 auto;">
            <form>
                <div style="display: flex; align-items: center;">
                    <div style="flex-grow: 1;">
                        <!-- This is just for UI styling, actual input comes from Streamlit -->
                        <input type="text" placeholder="Ask your question..." 
                               style="width: 100%; padding: 12px 20px; border-radius: 25px; border: none; 
                                      background-color: #272734; color: white; font-size: 16px;">
                    </div>
                    <div style="margin-left: 10px;">
                        <button type="submit" disabled
                                style="background-color: #FF4A4A; color: white; border: none; border-radius: 50%; 
                                       width: 40px; height: 40px; display: flex; align-items: center; justify-content: center;">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M22 2L11 13" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                <path d="M22 2L15 22L11 13L2 9L22 2Z" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>
                        </button>
                    </div>
                </div>
            </form>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Actual functional Streamlit chat input (visually hidden but working)
    return st.chat_input("Ask your question...", key="user_input")

def process_user_input(user_input, system_prompt):
    """Process user input and get a response from the backend"""
    if not user_input:
        return
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Create a placeholder for the assistant's response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Show typing indicator
        message_placeholder.markdown("""
        <div style="display: flex; gap: 5px;">
            <div style="width: 10px; height: 10px; background-color: #FF4A4A; border-radius: 50%; animation: pulse 1s infinite;"></div>
            <div style="width: 10px; height: 10px; background-color: #FF4A4A; border-radius: 50%; animation: pulse 1s infinite 0.2s;"></div>
            <div style="width: 10px; height: 10px; background-color: #FF4A4A; border-radius: 50%; animation: pulse 1s infinite 0.4s;"></div>
        </div>
        <style>
        @keyframes pulse {
            0% { opacity: 0.3; transform: scale(0.8); }
            50% { opacity: 1; transform: scale(1); }
            100% { opacity: 0.3; transform: scale(0.8); }
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Get response from backend
        response = st.session_state.backend.process_input(
            user_input, 
            st.session_state.demo_mode,
            system_prompt
        )
        
        # Simulate typing effect (optional)
        full_response = ""
        for word in response.split():
            full_response += word + " "
            time.sleep(0.01)  # Adjust speed as needed
            message_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
        
        # Display final response
        message_placeholder.markdown(response)
        
        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def export_chat_history():
    """Export chat history to a downloadable JSON file"""
    # Prepare the export data
    chat_export = {
        "messages": st.session_state.messages,
        "timestamp": datetime.now().isoformat(),
        "demo_mode": st.session_state.demo_mode
    }
    
    # Convert to JSON
    chat_json = json.dumps(chat_export, indent=4)
    
    # Encode to base64 for download
    b64 = base64.b64encode(chat_json.encode()).decode()
    
    # Create a download link
    download_filename = f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    href = f'<a href="data:file/json;base64,{b64}" download="{download_filename}">Click to download chat history</a>'
    st.sidebar.markdown(href, unsafe_allow_html=True)
    
    # Show success message
    st.sidebar.success("Chat history ready for download!")

# --- Main App Function ---
def main():
    """Main application entry point"""
    # Initialize session state
    init_session_state()
    
    # Apply custom CSS
    apply_custom_css()
    
    # Create header with branding
    col1, col2 = st.columns([1, 5])
    with col1:
        st.markdown(f"<h1 style='font-size: 48px; margin-bottom: 0;'>{APP_LOGO}</h1>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<h1 style='margin-top: 10px;'>{APP_TITLE}</h1>", unsafe_allow_html=True)
    
    # Add a subtle separator line
    st.markdown("<hr style='margin: 0; padding: 0; opacity: 0.3;'>", unsafe_allow_html=True)
    
    # Setup sidebar and get settings
    _, _, system_prompt, _ = create_sidebar()
    
    # Display chat history
    display_chat_messages()
    
    # Handle user input
    user_input = create_chat_input()
    if user_input:
        process_user_input(user_input, system_prompt)
        st.rerun()  # Rerun to update chat display

if __name__ == "__main__":
    main()