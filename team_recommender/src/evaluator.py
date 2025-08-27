def evaluate_team_performance(team_df):
    """
    Evaluates the performance of the selected team based on expected points and team diversity.

    Parameters:
    - team_df (pd.DataFrame): DataFrame containing the team information including player stats.

    Returns:
    - performance_score (float): The overall performance score of the team.
    - expected_points (float): The total expected points from the starting players.
    - diversity_score (float): The team diversity score.
    """
    expected_points = team_df[team_df['is_starting']]['expected_points_next_5'].sum()
    diversity_score = calculate_team_diversity(team_df)

    performance_score = expected_points + diversity_score
    return performance_score, expected_points, diversity_score


def calculate_team_diversity(team_df):
    """
    Calculates the diversity score of the team based on the number of different clubs represented.

    Parameters:
    - team_df (pd.DataFrame): DataFrame containing the team information.

    Returns:
    - diversity_score (float): The diversity score of the team.
    """
    unique_teams = team_df['team_name'].nunique()
    total_players = len(team_df)
    diversity_score = unique_teams / total_players if total_players > 0 else 0
    return diversity_score


def evaluate_bench_performance(bench_df):
    """
    Evaluates the performance of the bench players based on their expected points.

    Parameters:
    - bench_df (pd.DataFrame): DataFrame containing the bench players' information.

    Returns:
    - bench_expected_points (float): The total expected points from the bench players.
    """
    bench_expected_points = bench_df['expected_points_next_5'].sum()
    return bench_expected_points


def evaluate_captaincy_options(team_df):
    """
    Evaluates captaincy options based on player form, fixture difficulty, and consistency.

    Parameters:
    - team_df (pd.DataFrame): DataFrame containing the team information.

    Returns:
    - captaincy_options (pd.DataFrame): DataFrame of players ranked by captaincy potential.
    """
    team_df['captaincy_score'] = (
        team_df['form'] * 0.5 +
        (1 / team_df['fixture_difficulty']) * 0.3 +
        team_df['consistency'] * 0.2
    )
    captaincy_options = team_df[['id', 'web_name', 'captaincy_score']].sort_values(by='captaincy_score', ascending=False)
    return captaincy_options


def calculate_team_chemistry(team_df):
    """
    Calculates a chemistry bonus for players from the same club who have a proven track record of playing well together.

    Parameters:
    - team_df (pd.DataFrame): DataFrame containing the team information.

    Returns:
    - chemistry_bonus (float): The total chemistry bonus for the team.
    """
    chemistry_bonus = 0
    club_counts = team_df['team_name'].value_counts()

    for club, count in club_counts.items():
        if count > 1:  # Only consider clubs with more than one player
            chemistry_bonus += count * 0.1  # Example bonus calculation

    return chemistry_bonus