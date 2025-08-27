"""
Enhanced AI Integration System for FPL Analytics
Provides robust error handling, sophisticated prompt engineering, and multiple AI provider support
"""

import streamlit as st
from openai import OpenAI
import cohere
import os
import httpx
import time
import json
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import Dict, List, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIProviderError(Exception):
    """Custom exception for AI provider errors"""
    pass

class FPLAIAssistant:
    """Enhanced FPL AI Assistant with robust error handling and prompt engineering"""
    
    def __init__(self):
        self.openai_client = None
        self.cohere_client = None
        self.available_providers = []
        self.current_provider = None
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize AI providers with proper error handling"""
        # Initialize OpenAI
        openai_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if openai_key:
            try:
                self.openai_client = OpenAI(api_key=openai_key)
                # Test connection
                self.openai_client.models.list()
                self.available_providers.append("openai")
                logger.info("✅ OpenAI initialized successfully")
            except Exception as e:
                logger.error(f"❌ OpenAI initialization failed: {e}")
                st.warning(f"⚠️ OpenAI setup issue: {str(e)[:100]}...")
        
        # Initialize Cohere
        cohere_key = st.secrets.get("COHERE_API_KEY") or os.getenv("COHERE_API_KEY")
        if cohere_key:
            try:
                unverified_client = httpx.Client(verify=False, timeout=30.0)
                self.cohere_client = cohere.Client(
                    api_key=cohere_key,
                    client_name="fpl-analytics-app",
                    httpx_client=unverified_client,
                )
                # Test connection
                self.cohere_client.chat(
                    model="command-light",
                    message="Test",
                    max_tokens=10
                )
                self.available_providers.append("cohere")
                logger.info("✅ Cohere initialized successfully")
            except Exception as e:
                logger.error(f"❌ Cohere initialization failed: {e}")
                st.warning(f"⚠️ Cohere setup issue: {str(e)[:100]}...")
        
        # Set current provider preference
        if "cohere" in self.available_providers:
            self.current_provider = "cohere"
        elif "openai" in self.available_providers:
            self.current_provider = "openai"
        else:
            st.error("❌ No AI providers available. Please configure API keys.")
    
    @retry(
        stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=1, min=2, max=8),
        retry=retry_if_exception_type((AIProviderError, Exception))
    )
    def _call_openai(self, messages: List[Dict], **kwargs) -> str:
        """Call OpenAI with retry logic"""
        if not self.openai_client:
            raise AIProviderError("OpenAI client not initialized")
        
        try:
            response = self.openai_client.chat.completions.create(
                model=kwargs.get("model", "gpt-3.5-turbo"),
                messages=messages,
                max_tokens=kwargs.get("max_tokens", 600),
                temperature=kwargs.get("temperature", 0.7),
                presence_penalty=0.1,
                frequency_penalty=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise AIProviderError(f"OpenAI error: {str(e)}")
    
    @retry(
        stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=1, min=2, max=8),
        retry=retry_if_exception_type((AIProviderError, Exception))
    )
    def _call_cohere(self, prompt: str, **kwargs) -> str:
        """Call Cohere with retry logic"""
        if not self.cohere_client:
            raise AIProviderError("Cohere client not initialized")
        
        try:
            response = self.cohere_client.chat(
                model=kwargs.get("model", "command"),
                message=prompt,
                max_tokens=kwargs.get("max_tokens", 600),
                temperature=kwargs.get("temperature", 0.7),
                p=0.9
            )
            return response.text
        except Exception as e:
            logger.error(f"Cohere API call failed: {e}")
            raise AIProviderError(f"Cohere error: {str(e)}")
    
    def generate_response(self, prompt: str, context: Dict = None, **kwargs) -> Dict:
        """Generate AI response with fallback providers and comprehensive error handling"""
        context = context or {}
        
        # Prepare enhanced prompt
        enhanced_prompt = self._enhance_prompt(prompt, context)
        
        # Try providers in order of preference
        providers_to_try = self.available_providers.copy()
        if self.current_provider and self.current_provider in providers_to_try:
            providers_to_try.remove(self.current_provider)
            providers_to_try.insert(0, self.current_provider)
        
        last_error = None
        
        for provider in providers_to_try:
            try:
                st.info(f"🤖 Using {provider.upper()} AI assistant...")
                
                if provider == "openai":
                    messages = [
                        {"role": "system", "content": self._get_system_prompt(context)},
                        {"role": "user", "content": enhanced_prompt}
                    ]
                    response = self._call_openai(messages, **kwargs)
                    
                elif provider == "cohere":
                    full_prompt = f"{self._get_system_prompt(context)}\n\n{enhanced_prompt}"
                    response = self._call_cohere(full_prompt, **kwargs)
                
                else:
                    continue
                
                # Success! Update current provider and return
                self.current_provider = provider
                return {
                    "success": True,
                    "response": response,
                    "provider": provider,
                    "error": None
                }
                
            except Exception as e:
                last_error = e
                logger.warning(f"Provider {provider} failed: {e}")
                st.warning(f"⚠️ {provider.upper()} failed, trying next provider...")
                continue
        
        # All providers failed
        error_msg = f"All AI providers failed. Last error: {str(last_error)}" if last_error else "No AI providers available"
        return {
            "success": False,
            "response": "❌ AI services temporarily unavailable. Please try again later.",
            "provider": None,
            "error": error_msg
        }
    
    def _enhance_prompt(self, prompt: str, context: Dict) -> str:
        """Apply sophisticated prompt engineering techniques"""
        
        # Context-aware prompt enhancement
        enhanced_parts = []
        
        # Add context if available
        if context.get("current_gameweek"):
            enhanced_parts.append(f"Current Gameweek: {context['current_gameweek']}")
        
        if context.get("user_team_value"):
            enhanced_parts.append(f"User's Team Value: £{context['user_team_value']:.1f}m")
        
        if context.get("analysis_type"):
            enhanced_parts.append(f"Analysis Focus: {context['analysis_type']}")
        
        # Add structured thinking framework
        thinking_framework = """
        Please follow this analysis framework:
        1. SITUATION ANALYSIS: What is the current state?
        2. KEY FACTORS: What are the most important considerations?
        3. RECOMMENDATIONS: What specific actions should be taken?
        4. RATIONALE: Why are these the best options?
        5. RISKS: What could go wrong and how to mitigate?
        """
        
        # Combine all parts
        if enhanced_parts:
            context_str = "\n".join(enhanced_parts)
            return f"{context_str}\n\n{thinking_framework}\n\nUser Question: {prompt}"
        else:
            return f"{thinking_framework}\n\nUser Question: {prompt}"
    
    def _get_system_prompt(self, context: Dict) -> str:
        """Generate context-aware system prompt"""
        
        base_prompt = """You are an elite Fantasy Premier League (FPL) strategist and data analyst with deep expertise in:
        - Player performance analysis and statistical modeling
        - Fixture difficulty assessment and scheduling optimization
        - Transfer strategy and timing
        - Captaincy selection and differential picks
        - Chip strategy (Wildcard, Bench Boost, Free Hit, Triple Captain)
        - Game state management and rank improvement tactics

        Your responses must be:
        ✅ DATA-DRIVEN: Base recommendations on concrete statistics and trends
        ✅ ACTIONABLE: Provide specific, implementable advice
        ✅ STRATEGIC: Consider both short-term gains and long-term positioning
        ✅ RISK-AWARE: Highlight potential downsides and mitigation strategies
        ✅ CONCISE: Deliver maximum value in minimal words

        Communication style:
        - Use FPL terminology and metrics naturally
        - Structure responses with clear headings and bullet points
        - Include confidence levels for predictions (High/Medium/Low)
        - Provide alternative options when appropriate"""
        
        # Add context-specific enhancements
        if context.get("tab_context"):
            tab_context = context["tab_context"]
            if tab_context == "fixtures":
                base_prompt += "\n\nCURRENT FOCUS: Fixture difficulty analysis - prioritize fixture-based recommendations and scheduling insights."
            elif tab_context == "transfers":
                base_prompt += "\n\nCURRENT FOCUS: Transfer planning - emphasize player comparisons, value analysis, and timing strategies."
            elif tab_context == "team_analysis":
                base_prompt += "\n\nCURRENT FOCUS: Team optimization - focus on squad balance, formation, and performance maximization."
        
        return base_prompt
    
    def get_status(self) -> Dict:
        """Get current AI assistant status"""
        return {
            "available_providers": self.available_providers,
            "current_provider": self.current_provider,
            "total_providers": len(self.available_providers),
            "is_ready": len(self.available_providers) > 0
        }

# Global instance
_ai_assistant = None

def get_ai_assistant() -> FPLAIAssistant:
    """Get or create AI assistant instance"""
    global _ai_assistant
    if _ai_assistant is None:
        _ai_assistant = FPLAIAssistant()
    return _ai_assistant

def generate_fpl_analysis(prompt: str, context: Dict = None, **kwargs) -> str:
    """Main function for generating FPL analysis with AI"""
    assistant = get_ai_assistant()
    result = assistant.generate_response(prompt, context, **kwargs)
    
    if result["success"]:
        return result["response"]
    else:
        st.error(f"❌ AI Analysis Failed: {result['error']}")
        return result["response"]

def display_ai_status():
    """Display AI assistant status in sidebar"""
    assistant = get_ai_assistant()
    status = assistant.get_status()
    
    with st.sidebar:
        st.subheader("🤖 AI Assistant Status")
        
        if status["is_ready"]:
            st.success(f"✅ Ready ({status['total_providers']} provider{'s' if status['total_providers'] > 1 else ''})")
            if status["current_provider"]:
                st.info(f"🎯 Active: {status['current_provider'].upper()}")
        else:
            st.error("❌ No AI providers available")
            st.warning("Configure API keys in secrets or environment variables")
        
        # Test connection button
        if st.button("🧪 Test AI Connection"):
            test_prompt = "Briefly explain what FPL is."
            with st.spinner("Testing AI connection..."):
                result = assistant.generate_response(test_prompt, max_tokens=100)
                if result["success"]:
                    st.success("✅ AI is working correctly!")
                    with st.expander("Test Response"):
                        st.write(result["response"])
                else:
                    st.error("❌ AI test failed")
