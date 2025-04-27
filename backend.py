#!/usr/bin/env python3
"""
backend.py — LangChain business logic for Multitool Demo

Contains all the processing logic, chains, and models for the LangChain demo,
separated from the Streamlit UI code for cleaner architecture.
"""

import os
import requests
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import LLMChain, RetrievalQA, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool, AgentType

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class LangChainBackend:
    """Backend class that handles all LangChain functionality"""
    
    def __init__(self, model_name="gpt-4o-mini", temperature=0.7, memory=None):
        """Initialize backend with model settings"""
        self.model_name = model_name
        self.temperature = temperature
        self.memory = memory if memory else ConversationBufferMemory()
        self.initialize_components()
    
    def initialize_components(self):
        """Initialize all LangChain components"""
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            openai_api_key=OPENAI_API_KEY,
            streaming=True
        )
        
        # Setup Simple Chain
        simple_prompt = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}"),
            ("human", "{user_input}")
        ])
        self.simple_chain = LLMChain(llm=self.llm, prompt=simple_prompt)
        
        # Setup Retrieval QA
        self._ensure_data_file_exists()
        loader = TextLoader("data.txt")
        docs = loader.load()
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectordb = Chroma.from_documents(docs, embeddings)
        self.retriever = vectordb.as_retriever()
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever
        )
        
        # Setup Conversation Chain
        self.conv_chain = ConversationChain(llm=self.llm, memory=self.memory)
        
        # Setup Agent with Tools
        cat_tool = Tool(
            name="cat_fact",
            func=self._fetch_cat_fact,
            description="Get a random cat fact."
        )
        self.agent = initialize_agent(
            tools=[cat_tool],
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False
        )
    
    def _ensure_data_file_exists(self):
        """Make sure data.txt exists for the retriever"""
        if not os.path.exists("data.txt"):
            with open("data.txt", "w") as f:
                f.write("LangChain is a framework for developing LLM apps.")
    
    def _fetch_cat_fact(self, text: str) -> str:
        """Tool function to fetch a random cat fact"""
        res = requests.get("https://catfact.ninja/fact")
        return res.json().get("fact", "Error fetching cat fact.")
    
    def update_model_settings(self, model_name, temperature):
        """Update the model settings and reinitialize components"""
        self.model_name = model_name
        self.temperature = temperature
        self.initialize_components()
    
    def process_input(self, user_input, demo_mode, system_prompt="You are an AI consultant."):
        """Process user input based on selected demo mode"""
        try:
            if demo_mode == "Simple Chain":
                return self.simple_chain.run({
                    "system_prompt": system_prompt,
                    "user_input": user_input
                })
            
            elif demo_mode == "Retrieval QA":
                return self.qa_chain.run(user_input)
            
            elif demo_mode == "Memory + Retrieval":
                # First retrieval result
                fact = self.qa_chain.run(user_input)
                # Then conversation with memory
                return self.conv_chain.predict(input=f"{fact}\nUser: {user_input}")
            
            elif demo_mode == "Memory + Agent":
                # Agent run inside a conversation chain
                tool_resp = self.agent.run(user_input)
                return self.conv_chain.predict(input=f"{tool_resp}")
            
            elif demo_mode == "Memory + RAG Augment":
                # Fetch relevant docs
                docs = self.retriever.get_relevant_documents(user_input)
                context = "\n".join([d.page_content for d in docs])
                # Augment the question
                augmented = f"Context:\n{context}\n\n{user_input}"
                return self.conv_chain.predict(input=augmented)
            
            else:
                return "Unknown demo mode."
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    def reset_memory(self):
        """Reset the conversation memory"""
        self.memory = ConversationBufferMemory()
        self.conv_chain = ConversationChain(llm=self.llm, memory=self.memory)