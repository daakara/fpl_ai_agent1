import asyncio
import pandas as pd
from understat import Understat
import nest_asyncio

nest_asyncio.apply()

async def fetch_xg_xa():
    understat = Understat()
    players = await understat.get_players_stats('EPL', 2023)
    data = []
    for p in players:
        data.append({
            'web_name': p['player_name'],
            'team_name': p['team_title'],
            'xG': float(p.get('xG', 0)),
            'xA': float(p.get('xA', 0)),
            'games': int(p.get('games', 0))
        })
    df = pd.DataFrame(data)
    df['xG_next_5'] = df.apply(lambda r: (r['xG'] / r['games']) * 5 if r['games'] > 0 else 0, axis=1)
    df['xA_next_5'] = df.apply(lambda r: (r['xA'] / r['games']) * 5 if r['games'] > 0 else 0, axis=1)
    df.to_csv("xg_xa_data.csv", index=False)  # save for later use
    print("xG/xA data saved to xg_xa_data.csv")

if __name__ == "__main__":
    asyncio.run(fetch_xg_xa())
