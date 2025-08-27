def find_differentials(pool_df, threshold_pct=10.0, n=5):
    """Suggest low-ownership, high-value differentials"""
    diff = pool_df[(pool_df["selected_by_percent"] < threshold_pct) &
                   (pool_df["expected_points_next_5"] > 0)]
    diff["value_per_m"] = diff["expected_points_next_5"] / (diff["now_cost"] / 10)
    top_diff = diff.sort_values("value_per_m", ascending=False).head(n)
    return top_diff[["web_name", "team_name", "expected_points_next_5", "selected_by_percent"]]
