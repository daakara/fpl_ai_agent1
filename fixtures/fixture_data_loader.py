"""
Fixture Data Loader - Handles loading and processing fixture data from FPL API
"""
import streamlit as st
import pandas as pd
import requests
import logging
from typing import List, Dict


class FixtureDataLoader:
    """Loads and processes fixture data from FPL API"""
    
    def __init__(self):
        self.base_url = "https://fantasy.premierleague.com/api"
        self.logger = logging.getLogger(__name__)
    
    def load_fixtures(self) -> List[Dict]:
        """Load fixtures from FPL API"""
        try:
            url = f"{self.base_url}/fixtures/"
            response = requests.get(url, timeout=10, verify=False)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            self.logger.error(f"Error loading fixtures: {e}")
            return []
    
    def load_teams(self) -> List[Dict]:
        """Load teams data from FPL API"""
        try:
            url = f"{self.base_url}/bootstrap-static/"
            response = requests.get(url, timeout=10, verify=False)
            response.raise_for_status()
            data = response.json()
            return data.get('teams', [])
        except requests.RequestException as e:
            self.logger.error(f"Error loading teams: {e}")
            return []
    
    def get_next_5_fixtures(self, team_id: int, fixtures: List[Dict]) -> List[Dict]:
        """Get next 5 fixtures for a specific team"""
        team_fixtures = []
        
        for fixture in fixtures:
            # Check if this fixture involves the team
            if fixture['team_h'] == team_id or fixture['team_a'] == team_id:
                # Only include unfinished fixtures
                if not fixture.get('finished', False):
                    team_fixtures.append(fixture)
        
        # Sort by event (gameweek) first, then by kickoff_time if available
        def sort_key(fixture):
            event = fixture.get('event', 999)
            kickoff = fixture.get('kickoff_time', 'Z')
            return (event, kickoff)
        
        team_fixtures.sort(key=sort_key)
        return team_fixtures[:5]
    
    def process_fixtures_data(self) -> pd.DataFrame:
        """Process fixtures data into a structured DataFrame"""
        fixtures = self.load_fixtures()
        teams = self.load_teams()
        
        if not fixtures or not teams:
            st.warning("No fixtures or teams data available")
            return pd.DataFrame()
        
        team_lookup = {team['id']: team for team in teams}
        fixture_data = []
        
        # Get current gameweek
        current_gw = self._get_current_gameweek(fixtures)
        
        for team in teams:
            team_id = team['id']
            team_name = team['name']
            team_short_name = team['short_name']
            
            next_fixtures = self.get_next_5_fixtures(team_id, fixtures)
            
            if not next_fixtures:
                # Create placeholder data if no fixtures available
                opponents = self._get_placeholder_opponents(team, teams)
                for i, opponent_data in enumerate(opponents, 1):
                    fixture_data.append({
                        'team_id': team_id,
                        'team_name': team_name,
                        'team_short_name': team_short_name,
                        'fixture_number': i,
                        'gameweek': current_gw + i - 1,
                        'opponent_id': opponent_data['opponent']['id'],
                        'opponent_name': opponent_data['opponent']['name'],
                        'opponent_short_name': opponent_data['opponent']['short_name'],
                        'is_home': (i % 2 == 1),  # Alternate home/away
                        'difficulty': opponent_data['difficulty'],
                        'team_strength_overall': team.get('strength', 3),
                        'team_strength_home': team.get('strength_overall_home', 1200),
                        'team_strength_away': team.get('strength_overall_away', 1200),
                        'opponent_strength_overall': opponent_data['opponent'].get('strength', 3),
                        'opponent_strength_home': opponent_data['opponent'].get('strength_overall_home', 1200),
                        'opponent_strength_away': opponent_data['opponent'].get('strength_overall_away', 1200)
                    })
            else:
                # Process actual fixtures
                for i, fixture in enumerate(next_fixtures, 1):
                    is_home = fixture['team_h'] == team_id
                    opponent_id = fixture['team_a'] if is_home else fixture['team_h']
                    opponent = team_lookup.get(opponent_id, {})
                    
                    fixture_data.append({
                        'team_id': team_id,
                        'team_name': team_name,
                        'team_short_name': team_short_name,
                        'fixture_number': i,
                        'gameweek': fixture.get('event', current_gw + i - 1),
                        'opponent_id': opponent_id,
                        'opponent_name': opponent.get('name', 'Unknown'),
                        'opponent_short_name': opponent.get('short_name', 'UNK'),
                        'is_home': is_home,
                        'difficulty': fixture.get('team_h_difficulty' if is_home else 'team_a_difficulty', 3),
                        'team_strength_overall': team.get('strength', 3),
                        'team_strength_home': team.get('strength_overall_home', 1200),
                        'team_strength_away': team.get('strength_overall_away', 1200),
                        'opponent_strength_overall': opponent.get('strength', 3),
                        'opponent_strength_home': opponent.get('strength_overall_home', 1200),
                        'opponent_strength_away': opponent.get('strength_overall_away', 1200)
                    })
        
        return pd.DataFrame(fixture_data)
    
    def _get_current_gameweek(self, fixtures):
        """Extract current gameweek from fixtures"""
        unfinished_fixtures = [f for f in fixtures if not f.get('finished', False)]
        if unfinished_fixtures:
            current_gw_values = [f.get('event') for f in unfinished_fixtures if f.get('event') is not None]
            return min(current_gw_values) if current_gw_values else 1
        return 1
    
    def _get_placeholder_opponents(self, team, teams):
        """Generate placeholder opponents for teams without fixture data"""
        other_teams = [t for t in teams if t['id'] != team['id']]
        team_strength = team.get('strength', 3)
        
        opponents = []
        for i, opponent in enumerate(other_teams[:5]):
            opponent_strength = opponent.get('strength', 3)
            
            # Calculate difficulty based on strength difference
            strength_diff = opponent_strength - team_strength
            if strength_diff <= -1.5:
                difficulty = 2  # Easy
            elif strength_diff <= -0.5:
                difficulty = 2  # Easy
            elif strength_diff <= 0.5:
                difficulty = 3  # Average
            elif strength_diff <= 1.5:
                difficulty = 4  # Hard
            else:
                difficulty = 4  # Hard
            
            opponents.append({
                'opponent': opponent,
                'difficulty': difficulty
            })
        
        return opponents