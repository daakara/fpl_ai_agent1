# Create a test file: test_imports.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test 1: Try importing from the package (current behavior)
try:
    from team_recommender import TeamOptimizer
    print("✅ TeamOptimizer imported successfully from package")
    print(f"TeamOptimizer class: {TeamOptimizer}")
except ImportError as e:
    print(f"❌ Import from package failed: {e}")

# Test 2: Try importing the package and check its contents
try:
    import team_recommender
    print(f"✅ team_recommender package imported: {dir(team_recommender)}")
except ImportError as e:
    print(f"❌ Package import failed: {e}")

# Test 3: Try importing directly from the root-level file by renaming
try:
    # Temporarily remove the package from sys.modules to avoid conflicts
    if 'team_recommender' in sys.modules:
        del sys.modules['team_recommender']
    
    # Try to import the root-level file directly
    import importlib.util
    spec = importlib.util.spec_from_file_location("team_recommender_root", "team_recommender.py")
    team_recommender_root = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(team_recommender_root)
    
    TeamOptimizer = team_recommender_root.TeamOptimizer
    print("✅ TeamOptimizer imported successfully from root file")
    print(f"TeamOptimizer class: {TeamOptimizer}")
    
except Exception as e:
    print(f"❌ Root file import failed: {e}")

# Test 4: Check what's actually in the root team_recommender.py file
try:
    with open("team_recommender.py", "r") as f:
        content = f.read()
        if "class TeamOptimizer" in content:
            print("✅ TeamOptimizer class found in root team_recommender.py file")
        else:
            print("❌ TeamOptimizer class NOT found in root team_recommender.py file")
except FileNotFoundError:
    print("❌ Root team_recommender.py file not found")
except Exception as e:
    print(f"❌ Error reading root file: {e}")