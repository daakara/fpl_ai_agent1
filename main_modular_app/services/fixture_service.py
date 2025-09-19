"""
Enhanced Fixture Service with difficulty analysis and team strength evaluation
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import streamlit as st


class FixtureService:
    """Enhanced Fixture Service with comprehensive analysis capabilities"""
    
    def __init__(self):
        """Initialize the fixture service with team strength data"""
        self.team_strength_cache = {}
        self.fixture_difficulty_cache = {}
        
        # Premier League team strength ratings (simplified - would be dynamic in production)
        self.team_strength_ratings = {
            'MCI': {'attack': 95, 'defense': 85, 'overall': 90},
            'ARS': {'attack': 88, 'defense': 82, 'overall': 85},
            'LIV': {'attack': 92, 'defense': 80, 'overall': 86},
            'CHE': {'attack': 85, 'defense': 78, 'overall': 82},
            'MUN': {'attack': 82, 'defense': 75, 'overall': 79},
            'TOT': {'attack': 85, 'defense': 72, 'overall': 79},
            'NEW': {'attack': 75, 'defense': 85, 'overall': 80},
            'BRI': {'attack': 78, 'defense': 75, 'overall': 77},
            'AVL': {'attack': 80, 'defense': 73, 'overall': 77},
            'WHU': {'attack': 75, 'defense': 70, 'overall': 73},
            'CRY': {'attack': 70, 'defense': 75, 'overall': 73},
            'FUL': {'attack': 72, 'defense': 68, 'overall': 70},
            'BOU': {'attack': 70, 'defense': 65, 'overall': 68},
            'WOL': {'attack': 68, 'defense': 70, 'overall': 69},
            'EVE': {'attack': 65, 'defense': 68, 'overall': 67},
            'BRE': {'attack': 70, 'defense': 60, 'overall': 65},
            'NFO': {'attack': 62, 'defense': 65, 'overall': 64},
            'IPS': {'attack': 60, 'defense': 58, 'overall': 59},
            'LEI': {'attack': 65, 'defense': 55, 'overall': 60},
            'SOU': {'attack': 58, 'defense': 55, 'overall': 57}
        }
    
    def get_fixture_difficulty_score(self, team_short_name: str, is_home: bool, opponent_short_name: str) -> float:
        """
        Calculate fixture difficulty score for a team
        
        Args:
            team_short_name: Team's short name (e.g., 'ARS')
            is_home: Whether the team is playing at home
            opponent_short_name: Opponent's short name
            
        Returns:
            Difficulty score (1-5, where 1 is easiest and 5 is hardest)
        """
        cache_key = f"{team_short_name}_{opponent_short_name}_{is_home}"
        
        if cache_key in self.fixture_difficulty_cache:
            return self.fixture_difficulty_cache[cache_key]
        
        team_strength = self.team_strength_ratings.get(team_short_name, {'overall': 65})['overall']
        opponent_strength = self.team_strength_ratings.get(opponent_short_name, {'overall': 65})['overall']
        
        # Base difficulty is opponent strength relative to team strength
        base_difficulty = opponent_strength / max(team_strength, 1)
        
        # Home advantage adjustment (approximately 5-10% boost)
        if is_home:
            base_difficulty *= 0.95  # Slightly easier at home
        else:
            base_difficulty *= 1.05  # Slightly harder away
        
        # Convert to 1-5 scale
        if base_difficulty <= 0.8:
            difficulty = 1  # Very easy
        elif base_difficulty <= 0.95:
            difficulty = 2  # Easy
        elif base_difficulty <= 1.05:
            difficulty = 3  # Average
        elif base_difficulty <= 1.2:
            difficulty = 4  # Hard
        else:
            difficulty = 5  # Very hard
        
        self.fixture_difficulty_cache[cache_key] = difficulty
        return difficulty
    
    def get_upcoming_fixtures_difficulty(self, team_short_name: str, gameweeks: int = 5) -> Dict:
        """
        Get fixture difficulty for upcoming gameweeks
        
        Args:
            team_short_name: Team's short name
            gameweeks: Number of upcoming gameweeks to analyze
            
        Returns:
            Dictionary with fixture difficulty analysis
        """
        # This would integrate with actual fixture data in production
        # For now, providing a simplified implementation
        
        # Sample upcoming opponents (would be fetched from API)
        sample_fixtures = [
            {'opponent': 'MCI', 'home': False, 'gameweek': 20},
            {'opponent': 'BRE', 'home': True, 'gameweek': 21},
            {'opponent': 'LIV', 'home': False, 'gameweek': 22},
            {'opponent': 'SOU', 'home': True, 'gameweek': 23},
            {'opponent': 'TOT', 'home': False, 'gameweek': 24}
        ]
        
        fixtures_analysis = []
        total_difficulty = 0
        
        for i, fixture in enumerate(sample_fixtures[:gameweeks]):
            difficulty = self.get_fixture_difficulty_score(
                team_short_name, 
                fixture['home'], 
                fixture['opponent']
            )
            
            fixtures_analysis.append({
                'gameweek': fixture['gameweek'],
                'opponent': fixture['opponent'],
                'home': fixture['home'],
                'difficulty': difficulty,
                'difficulty_text': self._get_difficulty_text(difficulty)
            })
            
            total_difficulty += difficulty
        
        avg_difficulty = total_difficulty / len(fixtures_analysis) if fixtures_analysis else 3
        
        return {
            'fixtures': fixtures_analysis,
            'average_difficulty': avg_difficulty,
            'total_difficulty': total_difficulty,
            'rating': self._get_fixture_period_rating(avg_difficulty)
        }
    
    def _get_difficulty_text(self, difficulty: float) -> str:
        """Convert difficulty score to text description"""
        if difficulty <= 1.5:
            return "Very Easy"
        elif difficulty <= 2.5:
            return "Easy"
        elif difficulty <= 3.5:
            return "Average"
        elif difficulty <= 4.5:
            return "Hard"
        else:
            return "Very Hard"
    
    def _get_fixture_period_rating(self, avg_difficulty: float) -> str:
        """Get overall rating for fixture period"""
        if avg_difficulty <= 2:
            return "Excellent"
        elif avg_difficulty <= 2.5:
            return "Good"
        elif avg_difficulty <= 3.5:
            return "Average"
        elif avg_difficulty <= 4:
            return "Difficult"
        else:
            return "Very Difficult"
    
    def get_team_attack_defense_strength(self, team_short_name: str) -> Dict:
        """Get team's attacking and defensive strength ratings"""
        return self.team_strength_ratings.get(team_short_name, {
            'attack': 65, 'defense': 65, 'overall': 65
        })
    
    def compare_fixture_run(self, team1: str, team2: str, gameweeks: int = 5) -> Dict:
        """Compare fixture difficulty between two teams"""
        team1_fixtures = self.get_upcoming_fixtures_difficulty(team1, gameweeks)
        team2_fixtures = self.get_upcoming_fixtures_difficulty(team2, gameweeks)
        
        return {
            'team1': {'name': team1, 'data': team1_fixtures},
            'team2': {'name': team2, 'data': team2_fixtures},
            'recommendation': team1 if team1_fixtures['average_difficulty'] < team2_fixtures['average_difficulty'] else team2
        }

