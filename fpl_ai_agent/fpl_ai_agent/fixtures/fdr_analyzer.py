class FDRAnalyzer:
    """Class to analyze Fixture Difficulty Ratings (FDR) for upcoming matches."""
    
    def __init__(self, fixture_data_loader):
        self.fixture_data_loader = fixture_data_loader
        self.fixture_data = self.fixture_data_loader.load_fixture_data()
        self.team_strength = self.fixture_data_loader.load_team_strength()

    def calculate_fdr(self, team_name):
        """Calculate fixture difficulty ratings for a given team based on the next 5 fixtures."""
        fixtures = self.get_next_fixtures(team_name)
        attack_ratings = []
        defense_ratings = []

        for fixture in fixtures:
            opponent = fixture['opponent']
            attack_ratings.append(self.team_strength[opponent]['attack'])
            defense_ratings.append(self.team_strength[team_name]['defense'])

        fdr_attack = sum(attack_ratings) / len(attack_ratings) if attack_ratings else 0
        fdr_defense = sum(defense_ratings) / len(defense_ratings) if defense_ratings else 0

        return {
            'attack_fdr': fdr_attack,
            'defense_fdr': fdr_defense
        }

    def get_next_fixtures(self, team_name):
        """Retrieve the next 5 fixtures for the specified team."""
        fixtures = self.fixture_data.get(team_name, [])
        return fixtures[:5]  # Return only the next 5 fixtures

    def load_fdr_data(self):
        """Load and return FDR data for all teams."""
        fdr_data = {}
        for team in self.fixture_data.keys():
            fdr_data[team] = self.calculate_fdr(team)
        return fdr_data