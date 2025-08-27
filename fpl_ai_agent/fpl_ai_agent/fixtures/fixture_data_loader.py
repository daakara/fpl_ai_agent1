class FixtureDataLoader:
    """Handles loading fixture data from JSON files."""
    
    def __init__(self, fixtures_path: str, team_strength_path: str):
        self.fixtures_path = fixtures_path
        self.team_strength_path = team_strength_path

    def load_fixture_schedules(self):
        """Load fixture schedules from JSON file."""
        with open(self.fixtures_path, 'r') as file:
            fixture_data = json.load(file)
        return fixture_data

    def load_team_strength(self):
        """Load team strength data from JSON file."""
        with open(self.team_strength_path, 'r') as file:
            team_strength_data = json.load(file)
        return team_strength_data

    def get_fixtures_and_strength(self):
        """Load both fixture schedules and team strength data."""
        fixtures = self.load_fixture_schedules()
        team_strength = self.load_team_strength()
        return fixtures, team_strength