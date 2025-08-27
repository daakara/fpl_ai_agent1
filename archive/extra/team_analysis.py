import pandas as pd

def weakest_link_detector(team_df, pool_df, n_out=2):
    """Highlight weakest value players and suggest alternatives."""
    team_df = team_df.copy()
    team_df["value_per_m"] = team_df["expected_points_next_5"] / (team_df["now_cost"] / 10)
    weakest = team_df.nsmallest(n_out, "value_per_m")
    suggestions = []
    for _, row in weakest.iterrows():
        # Find best value replacement with same position, under budget
        budget = row["now_cost"]
        pos = row["element_type"]
        possible = pool_df[(pool_df["element_type"] == pos) & 
                           (pool_df["now_cost"] <= budget) &
                           (pool_df["id"] != row["id"])]
        possible["value_per_m"] = possible["expected_points_next_5"] / (possible["now_cost"] / 10)
        top_repl = possible.sort_values("value_per_m", ascending=False).head(1)
        if not top_repl.empty:
            suggestions.append((row["web_name"], top_repl.iloc[0]["web_name"]))
    return weakest, suggestions
