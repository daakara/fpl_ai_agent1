def calculate_team_chemistry(team_df):
    """
    Calculate a chemistry bonus for players from the same club.
    
    Parameters:
    - team_df: DataFrame containing player information including team names.
    
    Returns:
    - chemistry_bonus: A float representing the chemistry bonus.
    """
    chemistry_bonus = 0
    team_counts = team_df['team_name'].value_counts()
    
    for count in team_counts:
        if count > 1:
            chemistry_bonus += (count - 1) * 0.1  # Example bonus for each additional player from the same team
    
    return chemistry_bonus


def transform_player_data(player_df):
    """
    Transform player data for analysis.
    
    Parameters:
    - player_df: DataFrame containing raw player data.
    
    Returns:
    - transformed_df: DataFrame with transformed player data.
    """
    transformed_df = player_df.copy()
    transformed_df['now_cost_m'] = transformed_df['now_cost'] / 10  # Convert cost to millions
    transformed_df['injury_status'] = transformed_df['injury_status'].fillna('Fit')  # Handle missing injury status
    return transformed_df


def calculate_expected_points(player_df, form_weight=0.3, xg_weight=0.2, xa_weight=0.2):
    """
    Calculate expected points for players based on form, expected goals (xG), and expected assists (xA).
    
    Parameters:
    - player_df: DataFrame containing player statistics.
    - form_weight: Weight for the player's form in the calculation.
    - xg_weight: Weight for expected goals in the calculation.
    - xa_weight: Weight for expected assists in the calculation.
    
    Returns:
    - expected_points: Series containing expected points for each player.
    """
    expected_points = (
        form_weight * player_df['form'] +
        xg_weight * player_df['xG_next_5'] +
        xa_weight * player_df['xA_next_5']
    )
    return expected_points


def optimize_bench_selection(bench_candidates, budget):
    """
    Optimize bench selection to maximize value within the budget.
    
    Parameters:
    - bench_candidates: DataFrame containing potential bench players.
    - budget: Available budget for bench players.
    
    Returns:
    - selected_bench: DataFrame of selected bench players.
    """
    selected_bench = []
    remaining_budget = budget
    
    for _, player in bench_candidates.iterrows():
        if player['now_cost'] <= remaining_budget:
            selected_bench.append(player)
            remaining_budget -= player['now_cost']
    
    return pd.DataFrame(selected_bench)