def analyze_leagues(my_team, rivals_teams):
    """Find overlaps (template players) and differentials between you and rivals."""
    my_ids = set(my_team["id"])
    rival_ids_list = [set(team["id"]) for team in rivals_teams]
    shared = set.intersection(*rival_ids_list, my_ids)
    my_diffs = my_ids - set.union(*rival_ids_list)
    tips = []
    for p in my_team.itertuples():
        if p.id in my_diffs:
            tips.append(f"Consider keeping {p.web_name}: unique to you in this mini-league.")
    for p in my_team.itertuples():
        if p.id not in shared and p.value_per_m < 3:
            tips.append(f"Watch out: {p.web_name} is low value and not shared by rivals.")
    return tips
