# Create debug_app.py
import streamlit as st
import sys
import traceback

def main():
    st.set_page_config(page_title="FPL Debug", page_icon="🔍", layout="wide")
    st.title("🔍 FPL AI Agent - Debug Version")
    
    # Test imports first
    st.header("1. Testing Imports")
    
    try:
        from app_config import AppConfig, SessionStateManager
        st.success("✅ app_config imports working")
    except Exception as e:
        st.error(f"❌ app_config import failed: {e}")
        st.code(traceback.format_exc())
    
    try:
        from app2 import DataManager
        st.success("✅ DataManager import working")
    except Exception as e:
        st.error(f"❌ DataManager import failed: {e}")
        st.code(traceback.format_exc())
    
    try:
        from fpl_official import get_fixtures_data_async
        st.success("✅ fpl_official import working")
    except Exception as e:
        st.error(f"❌ fpl_official import failed: {e}")
        st.code(traceback.format_exc())
    
    # Test basic FPL API connection
    st.header("2. Testing FPL API Connection")
    
    if st.button("Test FPL API"):
        try:
            import requests
            response = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", timeout=10)
            if response.status_code == 200:
                data = response.json()
                st.success(f"✅ FPL API working - {len(data.get('elements', []))} players found")
                
                # Show sample data
                if data.get('elements'):
                    sample_player = data['elements'][0]
                    st.json({k: v for k, v in list(sample_player.items())[:10]})
            else:
                st.error(f"❌ FPL API error: {response.status_code}")
        except Exception as e:
            st.error(f"❌ FPL API connection failed: {e}")
            st.code(traceback.format_exc())
    
    # Test DataManager initialization
    st.header("3. Testing DataManager")
    
    if st.button("Test DataManager Creation"):
        try:
            from app2 import DataManager
            data_manager = DataManager()
            st.success("✅ DataManager created successfully")
            
            # Check what attributes it has
            st.write("DataManager attributes:")
            attrs = [attr for attr in dir(data_manager) if not attr.startswith('_')]
            st.write(attrs)
            
        except Exception as e:
            st.error(f"❌ DataManager creation failed: {e}")
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()