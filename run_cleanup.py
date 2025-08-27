# run_cleanup.py
import subprocess
import sys

def run_cleanup():
    """Execute the folder cleanup"""
    
    print("🧹 Starting FPL AI Agent folder cleanup...")
    
    # Step 1: Create the cleanup script
    with open("create_clean_structure.py", "w") as f:
        f.write(open("create_clean_structure.py").read())
    
    # Step 2: Run the cleanup
    try:
        subprocess.run([sys.executable, "create_clean_structure.py"], check=True)
        print("✅ File organization completed!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Cleanup failed: {e}")
        return False
    
    # Step 3: Create new files
    print("📝 Creating new configuration files...")
    
    # Create config template
    os.makedirs("config", exist_ok=True)
    
    # Create .env template
    with open(".env.template", "w") as f:
        f.write("""# FPL AI Agent Environment Variables
OPENAI_API_KEY=your_openai_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
FPL_TEAM_ID=your_fpl_team_id_here
""")
    
    print("✅ Cleanup completed!")
    print("\n📋 Next steps:")
    print("1. Copy .env.template to .env and add your API keys")
    print("2. Run: streamlit run core/simple_app.py")
    print("3. Check archive/ folder for moved files")
    
    return True

if __name__ == "__main__":
    run_cleanup()