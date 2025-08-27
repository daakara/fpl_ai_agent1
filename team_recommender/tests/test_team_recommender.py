import pandas as pd
import pytest
from team_recommender.src.team_recommender import get_latest_team_recommendations, valid_team_formation
import logging

# Configure logging for tests (optional)
logging.basicConfig(level=logging.DEBUG)

@pytest.fixture
def sample_players():
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'web_name': ['Player A', 'Player B', 'Player C', 'Player D', 'Player E',
                     'Player F', 'Player G', 'Player H', 'Player I', 'Player J'],
        'element_type': [1, 2, 3, 4, 1, 2, 3, 4, 2, 3],
        'now_cost': [50, 60, 70, 80, 50, 60, 70, 80, 60, 70],
        'expected_points_next_5': [5, 6, 7, 8, 5, 6, 7, 8, 6, 7],
        'selected_by_percent': [10, 20, 30, 40, 10, 20, 30, 40, 20, 30],
        'fixture_difficulty': [2, 3, 4, 5, 2, 3, 4, 5, 3, 4],
        'form': [3.0, 3.5, 4.0, 4.5, 3.0, 3.5, 4.0, 4.5, 3.5, 4.0],
        'injury_status': ['Fit', 'Injured', 'Fit', 'Fit', 'Doubtful', 'Fit', 'Fit', 'Injured', 'Fit', 'Fit'],
        'team_name': ['Team A', 'Team B', 'Team C', 'Team D', 'Team A', 'Team B', 'Team C', 'Team D', 'Team B', 'Team C'],
        'minutes_per_game': [90, 80, 70, 60, 90, 80, 70, 60, 80, 70],
        'clean_sheets_rate': [0.5, 0.4, 0.3, 0.2, 0.5, 0.4, 0.3, 0.2, 0.4, 0.3],
        'bps': [10, 12, 14, 16, 10, 12, 14, 16, 12, 14],
        'xG_next_5': [1.0, 1.2, 1.4, 1.6, 1.0, 1.2, 1.4, 1.6, 1.2, 1.4],
        'xA_next_5': [0.5, 0.6, 0.7, 0.8, 0.5, 0.6, 0.7, 0.8, 0.6, 0.7],
        'consistency': [0.7, 0.6, 0.8, 0.9, 0.7, 0.6, 0.8, 0.9, 0.6, 0.8],
        'xC_next_5': [0.3, 0.4, 0.5, 0.6, 0.3, 0.4, 0.5, 0.6, 0.4, 0.5]
    })

def test_get_latest_team_recommendations(sample_players):
    recommendations = get_latest_team_recommendations(
        df_players=sample_players,
        budget=100,
        formations=[(3, 4, 3)],
        max_budget_per_player=15,
        max_players_per_club=3,
        substitute_counts=(1, 3),
        injury_penalty=0.5,
        hard_fixture_penalty=0.8,
        min_ownership=0.0,
        form_weight=0.3,
        xg_weight=0.2,
        xa_weight=0.2,
        ownership_weight=0.0,
        budget_weight=0.1,
        team_diversity_weight=0.1,
        fixture_difficulty_weight=0.1,
        consistency_weight=0.05,
        bps_weight=0.05,
        minutes_weight=0.05,
        clean_sheet_weight=0.05,
        defensive_contribution_weight=0.05,
        debug=False
    )

    assert recommendations is not None
    assert 'team' in recommendations
    assert 'formation' in recommendations
    assert 'captain' in recommendations
    assert 'vice_captain' in recommendations
    assert 'total_expected_points' in recommendations
    assert 'total_cost' in recommendations
    assert 'rationale' in recommendations

    team = recommendations['team']
    assert len(team) == 15  # Check if the team has 15 players
    assert team['now_cost'].sum() <= 1000  # Check if the total cost is within budget

def test_team_diversity(sample_players):
    team = sample_players.copy()
    team['team_name'] = ['Team A'] * 5 + ['Team B'] * 5  # 5 players from Team A and 5 from Team B
    diversity_score = calculate_team_diversity(team)

    assert diversity_score >= 0  # Diversity score should be non-negative

def test_fixture_difficulty_score(sample_players):
    team = sample_players.copy()
    team['fixture_difficulty'] = [2, 3, 4, 5, 2, 3, 4, 5, 3, 4]
    fixture_score = calculate_fixture_difficulty_score(team)

    assert fixture_score >= 0  # Fixture difficulty score should be non-negative

def test_valid_team_formation():
    team = pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        'is_starting': [True] * 11 + [False] * 4,
        'element_type': [1, 2, 3, 4, 1, 2, 3, 4, 2, 3, 1, 2, 3, 4, 1]
    })

    assert valid_team_formation(team, (3, 4, 3))  # Valid formation
    assert not valid_team_formation(team, (4, 4, 2))  # Invalid formation (not enough forwards)

def test_get_latest_team_recommendations_empty_df():
    df_players = pd.DataFrame()
    with pytest.raises(RuntimeError, match="No valid team found under constraints"):
        get_latest_team_recommendations(df_players=df_players)

def test_get_latest_team_recommendations_insufficient_players(sample_players):
    df_players = sample_players[sample_players['element_type'] == 1]
    with pytest.raises(RuntimeError, match="No valid team found under constraints"):
        get_latest_team_recommendations(df_players=df_players)

def test_get_latest_team_recommendations_high_injury_penalty(sample_players):
    recommendations = get_latest_team_recommendations(
        df_players=sample_players,
        budget=100,
        formations=[(3, 4, 3)],
        max_budget_per_player=15,
        max_players_per_club=3,
        substitute_counts=(1, 3),
        injury_penalty=1.0,  # Max penalty
        hard_fixture_penalty=0.8,
        min_ownership=0.0,
        form_weight=0.3,
        xg_weight=0.2,
        xa_weight=0.2,
        ownership_weight=0.0,
        budget_weight=0.1,
        team_diversity_weight=0.1,
        fixture_difficulty_weight=0.1,
        consistency_weight=0.05,
        bps_weight=0.05,
        minutes_weight=0.05,
        clean_sheet_weight=0.05,
        defensive_contribution_weight=0.05,
        debug=False
    )

    assert recommendations is not None

def test_get_latest_team_recommendations_high_fixture_difficulty_penalty(sample_players):
    recommendations = get_latest_team_recommendations(
        df_players=sample_players,
        budget=100,
        formations=[(3, 4, 3)],
        max_budget_per_player=15,
        max_players_per_club=3,
        substitute_counts=(1, 3),
        injury_penalty=0.5,
        hard_fixture_penalty=1.0,  # Max penalty
        min_ownership=0.0,
        form_weight=0.3,
        xg_weight=0.2,
        xa_weight=0.2,
        ownership_weight=0.0,
        budget_weight=0.1,
        team_diversity_weight=0.1,
        fixture_difficulty_weight=0.1,
        consistency_weight=0.05,
        bps_weight=0.05,
        minutes_weight=0.05,
        clean_sheet_weight=0.05,
        defensive_contribution_weight=0.05,
        debug=False
    )

    assert recommendations is not None

def test_get_latest_team_recommendations_min_ownership(sample_players):
    recommendations = get_latest_team_recommendations(
        df_players=sample_players,
        budget=100,
        formations=[(3, 4, 3)],
        max_budget_per_player=15,
        max_players_per_club=3,
        substitute_counts=(1, 3),
        injury_penalty=0.5,
        hard_fixture_penalty=0.8,
        min_ownership=35.0,  # Higher min ownership
        form_weight=0.3,
        xg_weight=0.2,
        xa_weight=0.2,
        ownership_weight=0.0,
        budget_weight=0.1,
        team_diversity_weight=0.1,
        fixture_difficulty_weight=0.1,
        consistency_weight=0.05,
        bps_weight=0.05,
        minutes_weight=0.05,
        clean_sheet_weight=0.05,
        defensive_contribution_weight=0.05,
        debug=False
    )

    assert recommendations is not None

def test_valid_team_formation_invalid_team_size():
    team = pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        'is_starting': [True] * 11 + [False] * 3,
        'element_type': [1, 2, 3, 4, 1, 2, 3, 4, 2, 3, 1, 2, 3, 4]
    })
    assert not valid_team_formation(team, (3, 4, 3))

def test_get_latest_team_recommendations_invalid_budget(sample_players):
    with pytest.raises(ValueError, match="Budget must be a positive number"):
        get_latest_team_recommendations(df_players=sample_players, budget=-10)

def test_get_latest_team_recommendations_missing_columns(sample_players):
    df = sample_players.drop(columns=['expected_points_next_5'])
    with pytest.raises(KeyError, match="expected_points_next_5"):
        get_latest_team_recommendations(df_players=df)

def test_get_latest_team_recommendations_non_numeric_form(sample_players):
    df = sample_players.copy()
    df.loc[0, 'form'] = 'invalid'
    recommendations = get_latest_team_recommendations(df_players=df)
    assert recommendations is not None  # Or assert that the player with invalid form is not selected

def test_get_latest_team_recommendations_no_valid_players_after_filtering(sample_players):
    recommendations = get_latest_team_recommendations(
        df_players=sample_players,
        min_ownership=100  # Set min_ownership so high that no players qualify
    )
    # Assert that the function returns None or raises an exception, depending on the desired behavior
    assert recommendations is None