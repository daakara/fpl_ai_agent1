def recommend_captain(team_players):
    """Pick captain based on expected points, xG/xA, opposition."""
    starters = team_players[team_players["is_starting"]]
    # Weight: expected points + 0.5*(xG_next_5 + xA_next_5)
    starters["c_score"] = (starters["expected_points_next_5"] +
                           0.5 * (starters["xG_next_5"] + starters["xA_next_5"]))
    starter_sorted = starters.sort_values("c_score", ascending=False)
    cap = starter_sorted.iloc[0]
    vc = starter_sorted.iloc[1] if len(starter_sorted) > 1 else None
    message = (
        f"Captain: {cap['web_name']} (exp pts: {cap['expected_points_next_5']:.1f}, "
        f"xG: {cap['xG_next_5']:.2f}, fixture: {cap['fixture_difficulty']})"
    )
    if vc is not None:
        message += f"\nVice: {vc['web_name']} (back-up option)."
    return message
