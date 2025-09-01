"""
Test Configuration for FPL Analytics App
Enhanced with comprehensive fixtures for modularization testing
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import sys
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_players_data():
    """Sample players data for testing"""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'web_name': ['Salah', 'Kane', 'Haaland', 'Son', 'Bruno'],
        'team': [1, 2, 3, 4, 5],
        'element_type': [3, 4, 4, 3, 3],  # position types
        'now_cost': [130, 115, 120, 95, 105],
        'total_points': [250, 200, 280, 180, 190],
        'points_per_game': [6.5, 5.2, 7.3, 4.7, 4.9],
        'selected_by_percent': [45.6, 23.4, 38.2, 18.9, 25.1],
        'form': [6.0, 4.5, 7.5, 4.0, 5.2],
        'minutes': [2850, 2340, 2920, 2180, 2650],
        'goals_scored': [22, 18, 35, 12, 8],
        'assists': [12, 8, 5, 15, 18],
        'clean_sheets': [0, 0, 0, 0, 0],
        'saves': [0, 0, 0, 0, 0],
        'bonus': [18, 12, 22, 8, 14],
        'bps': [850, 620, 980, 520, 680],
        'influence': [1250.5, 980.2, 1450.8, 820.3, 1050.7],
        'creativity': [890.3, 520.1, 380.9, 1120.5, 1380.2],
        'threat': [1580.2, 1290.8, 1820.5, 980.3, 750.1],
        'ict_index': [372.1, 279.0, 365.1, 293.0, 318.0],
        'starts': [32, 28, 35, 26, 30],
        'expected_goals': [18.5, 15.2, 28.3, 8.9, 6.1],
        'expected_assists': [8.3, 5.1, 3.2, 12.8, 15.4],
        'expected_goal_involvements': [26.8, 20.3, 31.5, 21.7, 21.5],
        'expected_goals_conceded': [35.2, 28.9, 22.1, 31.5, 29.8]
    })


@pytest.fixture
def sample_teams_data():
    """Sample teams data for testing"""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Liverpool', 'Tottenham', 'Man City', 'Arsenal', 'Man Utd'],
        'short_name': ['LIV', 'TOT', 'MCI', 'ARS', 'MUN'],
        'strength': [5, 4, 5, 4, 3],
        'strength_overall_home': [1200, 1100, 1250, 1150, 1050],
        'strength_overall_away': [1150, 1050, 1200, 1100, 1000],
        'strength_attack_home': [1300, 1200, 1350, 1250, 1100],
        'strength_attack_away': [1250, 1150, 1300, 1200, 1050],
        'strength_defence_home': [1100, 1000, 1150, 1050, 950],
        'strength_defence_away': [1050, 950, 1100, 1000, 900]
    })


@pytest.fixture
def sample_fixtures_data():
    """Sample fixtures data for testing"""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'team_h': [1, 2, 3, 4, 5],
        'team_a': [2, 3, 4, 5, 1],
        'team_h_difficulty': [3, 4, 2, 3, 5],
        'team_a_difficulty': [4, 2, 5, 3, 2],
        'event': [1, 1, 2, 2, 3],
        'kickoff_time': [
            '2024-08-17T15:00:00Z',
            '2024-08-17T17:30:00Z',
            '2024-08-24T15:00:00Z',
            '2024-08-24T17:30:00Z',
            '2024-08-31T15:00:00Z'
        ],
        'finished': [True, True, False, False, False],
        'team_h_score': [2, 1, None, None, None],
        'team_a_score': [1, 3, None, None, None]
    })


@pytest.fixture
def sample_squad_data():
    """Sample squad data for testing"""
    return {
        'picks': [
            {'element': 1, 'position': 1, 'multiplier': 1, 'is_captain': False, 'is_vice_captain': False},
            {'element': 2, 'position': 2, 'multiplier': 1, 'is_captain': True, 'is_vice_captain': False},
            {'element': 3, 'position': 3, 'multiplier': 1, 'is_captain': False, 'is_vice_captain': True},
        ],
        'chips': [
            {'name': 'wildcard', 'time': '2024-08-15T10:00:00Z', 'event': 1},
        ],
        'transfers': {
            'cost': 4,
            'status': 'cost',
            'limit': None,
            'made': 2,
            'bank': 5,
            'value': 1000
        }
    }


@pytest.fixture
def mock_fpl_api():
    """Mock FPL API responses"""
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.json.return_value = {
            'elements': [],
            'teams': [],
            'events': [],
            'element_types': [
                {'id': 1, 'plural_name': 'Goalkeepers', 'singular_name': 'Goalkeeper'},
                {'id': 2, 'plural_name': 'Defenders', 'singular_name': 'Defender'},
                {'id': 3, 'plural_name': 'Midfielders', 'singular_name': 'Midfielder'},
                {'id': 4, 'plural_name': 'Forwards', 'singular_name': 'Forward'}
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        yield mock_get


@pytest.fixture
def mock_streamlit():
    """Mock Streamlit components for testing UI components"""
    with patch.multiple(
        'streamlit',
        columns=Mock(return_value=[Mock(), Mock(), Mock()]),
        selectbox=Mock(return_value='Test Option'),
        slider=Mock(return_value=50),
        multiselect=Mock(return_value=['Option1', 'Option2']),
        checkbox=Mock(return_value=True),
        button=Mock(return_value=False),
        dataframe=Mock(),
        metric=Mock(),
        plotly_chart=Mock(),
        write=Mock(),
        header=Mock(),
        subheader=Mock(),
        info=Mock(),
        success=Mock(),
        warning=Mock(),
        error=Mock(),
        session_state=MagicMock()
    ) as mock_st:
        yield mock_st


@pytest.fixture
def data_manager_mock():
    """Mock DataManager for testing"""
    mock_dm = Mock()
    mock_dm.load_players_data.return_value = pd.DataFrame()
    mock_dm.load_teams_data.return_value = pd.DataFrame()
    mock_dm.load_fixtures_data.return_value = pd.DataFrame()
    mock_dm.is_data_loaded.return_value = True
    return mock_dm


@pytest.fixture
def app_config_mock():
    """Mock application configuration"""
    return {
        'api': {
            'base_url': 'https://fantasy.premierleague.com/api/',
            'timeout': 30,
            'retries': 3
        },
        'ui': {
            'page_title': 'FPL Analytics Test',
            'page_icon': '⚽',
            'layout': 'wide'
        },
        'cache': {
            'ttl': 3600,
            'max_entries': 1000
        }
    }
