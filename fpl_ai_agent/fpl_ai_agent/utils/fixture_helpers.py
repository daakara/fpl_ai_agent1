def calculate_average_difficulty(fixture_data):
    total_difficulty = sum(fixture['difficulty'] for fixture in fixture_data)
    return total_difficulty / len(fixture_data) if fixture_data else 0

def format_fixture_data(fixture_data):
    formatted_data = []
    for fixture in fixture_data:
        formatted_data.append({
            'team': fixture['team'],
            'opponent': fixture['opponent'],
            'difficulty': fixture['difficulty'],
            'date': fixture['date']
        })
    return formatted_data

def get_next_five_fixtures(team, fixture_schedule):
    team_fixtures = [fixture for fixture in fixture_schedule if fixture['team'] == team]
    return team_fixtures[:5]  # Return the next 5 fixtures

def get_fdr_for_team(team, fixture_schedule, team_strength):
    next_fixtures = get_next_five_fixtures(team, fixture_schedule)
    fdr = {
        'team': team,
        'average_attack_difficulty': calculate_average_difficulty([{
            'difficulty': team_strength[fixture['opponent']]['attack']
        } for fixture in next_fixtures]),
        'average_defense_difficulty': calculate_average_difficulty([{
            'difficulty': team_strength[fixture['opponent']]['defense']
        } for fixture in next_fixtures])
    }
    return fdr