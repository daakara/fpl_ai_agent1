import streamlit as st
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from llm_integration import fpl_qa_with_openai, cohere_client

@dataclass
class ChatMessage:
    role: str
    content: str

class AIResponseGenerator:
    """Handles AI response generation with fallback logic"""
    
    def __init__(self):
        self.cohere_available = bool(os.getenv("COHERE_API_KEY"))
        self.openai_available = bool(os.getenv("OPENAI_API_KEY"))
    
    def generate_response(self, prompt: str, sample_data: List[Dict], team_info: Dict) -> tuple[str, str]:
        """Generate AI response and return (response, provider_used)"""
        if self.cohere_available:
            try:
                cohere_prompt = f"""You are an expert FPL assistant analyzing current Fantasy Premier League data.
User's Current Team: {team_info}
Sample Player Data: {sample_data[:10]}
Question: {prompt}
Provide specific, actionable FPL advice in 2-3 paragraphs."""
                
                response = cohere_client.chat(
                    model="command", 
                    message=cohere_prompt, 
                    max_tokens=400, 
                    temperature=0.6
                )
                return response.text, "Cohere"
                
            except Exception as cohere_error:
                st.warning(f"⚠️ Cohere failed, switching to OpenAI... ({str(cohere_error)[:50]})")
                
        if self.openai_available:
            try:
                response = fpl_qa_with_openai(prompt, sample_data, team_info)
                return response, "OpenAI"
            except Exception as openai_error:
                st.error(f"❌ OpenAI failed: {openai_error}")
                
        return "❌ No AI service available. Please configure API keys.", "None"

class FeedbackManager:
    """Handles user feedback for AI responses"""
    
    @staticmethod
    def render_feedback_buttons(question_hash: str) -> None:
        """Render feedback buttons with unique keys"""
        col1, col2 = st.columns([1, 10])
        with col1:
            if st.button("👍", key=f"thumbs_up_{question_hash}"):
                st.success("Thanks for your feedback!")
                FeedbackManager._store_feedback(question_hash, "positive")
        with col2:
            if st.button("👎", key=f"thumbs_down_{question_hash}"):
                st.error("We'll work on improving the AI's responses.")
                FeedbackManager._store_feedback(question_hash, "negative")
    
    @staticmethod
    def _store_feedback(question_hash: str, feedback_type: str) -> None:
        """Store feedback in session state for analytics"""
        if "feedback_data" not in st.session_state:
            st.session_state.feedback_data = []
        
        st.session_state.feedback_data.append({
            "question_hash": question_hash,
            "feedback": feedback_type,
            "timestamp": st.session_state.get("current_timestamp")
        })

class ChatManager:
    """Manages the chat interface and conversation flow"""
    
    def __init__(self, ai_generator: AIResponseGenerator):
        self.ai_generator = ai_generator
        self.example_questions = [
            "Who should I captain this gameweek?",
            "Which players have the best fixtures in the next 5 gameweeks?",
            "Should I use my wildcard now?",
            "Who are the best differential picks under 10% ownership?",
        ]
    
    def render_example_questions(self) -> None:
        """Render example questions with unique keys"""
        with st.expander("💡 Example Questions"):
            for i, question in enumerate(self.example_questions):
                if st.button(question, key=f"example_question_{i}"):
                    st.session_state.user_question_to_submit = question
                    st.rerun()
    
    def render_chat_messages(self) -> None:
        """Render existing chat messages"""
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    def handle_user_input(self, sample_data: List[Dict], team_info: Dict) -> None:
        """Handle user input and generate AI response"""
        prompt = st.chat_input("Ask your FPL question...")
        
        # Handle example question submission
        if "user_question_to_submit" in st.session_state:
            prompt = st.session_state.user_question_to_submit
            del st.session_state.user_question_to_submit

        if prompt:
            # Add user message
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    ai_response, provider = self.ai_generator.generate_response(
                        prompt, sample_data, team_info
                    )
                    
                    st.markdown(ai_response)
                    
                    # Show provider info
                    if provider == "Cohere":
                        st.success("Responded using Cohere AI.", icon="✅")
                    elif provider == "OpenAI":
                        st.info("Responded using OpenAI.", icon="🤖")
                    else:
                        st.error("No AI service available.", icon="❌")
            
            # Add AI response to chat history
            st.session_state.chat_messages.append({"role": "assistant", "content": ai_response})
            
            # Render feedback buttons
            question_hash = str(hash(prompt))
            FeedbackManager.render_feedback_buttons(question_hash)