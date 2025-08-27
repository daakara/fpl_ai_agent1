import sys
import os

# Get the parent directory (where the root team_recommender.py is located)
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Import the root team_recommender.py file directly
import importlib.util
spec = importlib.util.spec_from_file_location(
    "team_recommender_root", 
    os.path.join(parent_dir, "team_recommender.py")
)

if spec and spec.loader:
    team_recommender_root = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(team_recommender_root)
    
    # Export the classes and functions from the root file
    TeamOptimizer = team_recommender_root.TeamOptimizer
    get_latest_team_recommendations = team_recommender_root.get_latest_team_recommendations
    valid_team_formation = team_recommender_root.valid_team_formation
    
    # Export additional functions if they exist
    try:
        calculate_team_diversity = team_recommender_root.calculate_team_diversity
        calculate_fixture_difficulty_score = team_recommender_root.calculate_fixture_difficulty_score
        generate_key_player_explanations = team_recommender_root.generate_key_player_explanations
        render_team_recommendations_tab = team_recommender_root.render_team_recommendations_tab
    except AttributeError:
        # These functions might not exist in the root file
        def calculate_team_diversity(*args, **kwargs):
            return 0
        def calculate_fixture_difficulty_score(*args, **kwargs):
            return 0
        def generate_key_player_explanations(*args, **kwargs):
            return {}
        def render_team_recommendations_tab(*args, **kwargs):
            pass

else:
    # Fallback if the root file can't be loaded
    class TeamOptimizer:
        def __init__(self, *args, **kwargs):
            pass
        def optimize_team(self):
            return None
    
    def get_latest_team_recommendations(*args, **kwargs):
        return None
    
    def valid_team_formation(*args, **kwargs):
        return False
    
    def calculate_team_diversity(*args, **kwargs):
        return 0
    
    def calculate_fixture_difficulty_score(*args, **kwargs):
        return 0
    
    def generate_key_player_explanations(*args, **kwargs):
        return {}
    
    def render_team_recommendations_tab(*args, **kwargs):
        pass

__all__ = [
    "TeamOptimizer",
    "get_latest_team_recommendations",
    "valid_team_formation",
    "calculate_team_diversity", 
    "calculate_fixture_difficulty_score",
    "generate_key_player_explanations",
    "render_team_recommendations_tab"
]