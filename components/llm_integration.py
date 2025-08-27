import streamlit as st
from openai import OpenAI
import cohere
import os
import httpx
import time
from tenacity import retry, stop_after_attempt, wait_exponential

# Securely load API keys from Streamlit secrets or environment variables
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = st.secrets.get("COHERE_API_KEY") or os.getenv("COHERE_API_KEY")

# Initialize OpenAI client
if OPENAI_API_KEY:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        st.error(f"❌ Error initializing OpenAI: {e}")
        client = None
else:
    client = None
    st.warning("⚠️ OpenAI API key not found. Please set OPENAI_API_KEY in secrets or environment variables.")

# Initialize Cohere client
cohere_client = None  # Initialize to None
if COHERE_API_KEY:
    try:
        # In environments with SSL-inspecting proxies, verification needs to be disabled.
        unverified_client = httpx.Client(verify=False)
        cohere_client = cohere.Client(
            api_key=COHERE_API_KEY,
            client_name="streamlit-fpl-app",
            httpx_client=unverified_client,
        )
    except Exception as e:
        st.error(f"❌ Error initializing Cohere: {e}")
        cohere_client = None
else:
    st.warning("⚠️ Cohere API key not found. Please set COHERE_API_KEY in secrets or environment variables.")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_cohere_response(prompt):
    """Generate response from Cohere with retry."""
    if not cohere_client:
        raise ValueError("Cohere client not initialized.")
    response = cohere_client.chat(
        model="command",
        message=prompt,
        max_tokens=400,
        temperature=0.7
    )
    return response.text

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_openai_response(prompt):
    """Generate response from OpenAI with retry."""
    if not client:
        raise ValueError("OpenAI client not initialized.")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert FPL analyst. Provide specific, actionable advice."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=400,
        temperature=0.7
    )
    return response.choices[0].message.content

def fpl_qa_with_openai(user_question, player_data, team_info):
    if not client:
        return "❌ No AI service available. Please configure OpenAI API key."
    
    try:
        # Enhanced prompt engineering for more specific and actionable advice
        prompt = f"""You are an expert Fantasy Premier League (FPL) assistant with deep knowledge of player performance, fixtures, and strategy. Your goal is to provide concise, data-driven advice to FPL managers.

        Current User Team: {team_info}

        Available Player Data (sample): {player_data}

        User Question: {user_question}

        Instructions:
        1.  Analyze the user's question and the provided data.
        2.  Focus on specific player recommendations, considering their form, expected points, and upcoming fixtures.
        3.  Evaluate fixture difficulty using a simple scale (Easy, Medium, Hard).
        4.  Provide strategic advice on captaincy, transfers, and chip usage, if relevant.
        5.  Justify your recommendations with data and reasoning.
        6.  Keep your response concise (2-3 paragraphs maximum).

        Begin!
        """
        try:
            ai_response = generate_cohere_response(prompt)
            st.info("✅ Cohere used successfully!")  # Indicate Cohere was used
        except Exception as cohere_error:
            st.warning(f"⚠️ Cohere failed, switching to OpenAI... (Error: {cohere_error})")
            try:
                ai_response = generate_openai_response(prompt)
            except Exception as openai_error:
                st.error(f"❌ OpenAI failed: {openai_error}")
                return f"❌ Both Cohere and OpenAI failed. Please check API keys and network connectivity."

        # Add a feedback mechanism (thumbs up/down)
        col1, col2 = st.columns([1, 10])
        with col1:
            if st.button("👍", key=f"openai_thumbs_up_{hash(user_question)}"):
                st.success("Thanks for your feedback!")
        with col2:
            if st.button("👎", key=f"openai_thumbs_down_{hash(user_question)}"):
                st.error("We'll work on improving the AI's responses.")

        return ai_response
        
    except Exception as e:
        st.error(f"❌ Error generating FPL advice: {str(e)}")
        return f"❌ Error generating FPL advice: {str(e)}"

def test_cohere_connection():
    """Test if Cohere is working properly"""
    if st.button("🧪 Test Cohere Connection"):
        if cohere_client:
            try:
                with st.spinner("Testing Cohere..."):
                    response = cohere_client.chat(
                        model="command",
                        message="What is FPL?",
                        max_tokens=50
                    )
                st.success("✅ Cohere is working! Response:")
                st.write(response.text)
            except Exception as e:
                st.error(f"❌ Cohere test failed: {e}")
        else:
            st.error("❌ Cohere client not initialized")