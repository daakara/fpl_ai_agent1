### Project Structure

1. **Project Directory**
   ```
   fpl_advanced_filters/
   ├── app.py
   ├── filters.py
   ├── data_loader.py
   ├── ui_components.py
   ├── utils.py
   ├── requirements.txt
   └── README.md
   ```

2. **Requirements File (`requirements.txt`)**
   ```plaintext
   streamlit
   pandas
   numpy
   requests
   ```

3. **Main Application File (`app.py`)**
   ```python
   import streamlit as st
   from filters import FilterManager
   from data_loader import load_player_data

   def main():
       st.title("Fantasy Football Team Selection")

       # Load player data
       players_df = load_player_data()

       # Initialize filter manager
       filter_manager = FilterManager(players_df)

       # Display filters
       st.sidebar.header("Filters")
       price_range = st.sidebar.slider("Price Range", 0, 15, (5, 10))
       selected_teams = st.sidebar.multiselect("Select Teams", options=players_df['team_name'].unique())
       selected_positions = st.sidebar.multiselect("Select Positions", options=players_df['element_type'].unique())
       fixture_difficulty = st.sidebar.selectbox("Fixture Difficulty", options=["All", "Easy", "Medium", "Hard"])

       # Apply filters
       filtered_players = filter_manager.apply_filters(price_range, selected_teams, selected_positions, fixture_difficulty)

       # Display filtered players
       st.subheader("Filtered Players")
       st.dataframe(filtered_players)

       # Save filter settings
       if st.button("Save Filter Settings"):
           filter_manager.save_filter_settings(price_range, selected_teams, selected_positions, fixture_difficulty)
           st.success("Filter settings saved!")

   if __name__ == "__main__":
       main()
   ```

4. **Filter Manager (`filters.py`)**
   ```python
   import pandas as pd
   import json
   import os

   class FilterManager:
       def __init__(self, players_df: pd.DataFrame):
           self.players_df = players_df
           self.settings_file = "filter_settings.json"
           self.load_filter_settings()

       def apply_filters(self, price_range, teams, positions, fixture_difficulty):
           filtered_df = self.players_df.copy()

           # Filter by price
           filtered_df = filtered_df[(filtered_df['now_cost'] / 10 >= price_range[0]) & 
                                     (filtered_df['now_cost'] / 10 <= price_range[1])]

           # Filter by team
           if teams:
               filtered_df = filtered_df[filtered_df['team_name'].isin(teams)]

           # Filter by position
           if positions:
               filtered_df = filtered_df[filtered_df['element_type'].isin(positions)]

           # Filter by fixture difficulty
           if fixture_difficulty != "All":
               filtered_df = filtered_df[filtered_df['fixture_difficulty'] == fixture_difficulty]

           return filtered_df

       def save_filter_settings(self, price_range, teams, positions, fixture_difficulty):
           settings = {
               "price_range": price_range,
               "teams": teams,
               "positions": positions,
               "fixture_difficulty": fixture_difficulty
           }
           with open(self.settings_file, 'w') as f:
               json.dump(settings, f)

       def load_filter_settings(self):
           if os.path.exists(self.settings_file):
               with open(self.settings_file, 'r') as f:
                   settings = json.load(f)
                   return settings
           return None
   ```

5. **Data Loader (`data_loader.py`)**
   ```python
   import pandas as pd

   def load_player_data() -> pd.DataFrame:
       # Simulated player data loading
       data = {
           "id": [1, 2, 3, 4],
           "web_name": ["Player A", "Player B", "Player C", "Player D"],
           "team_name": ["Team 1", "Team 2", "Team 1", "Team 3"],
           "element_type": [1, 2, 1, 3],  # 1: GK, 2: DEF, 3: MID, 4: FWD
           "now_cost": [50, 70, 60, 80],  # Cost in tenths of million
           "fixture_difficulty": ["Easy", "Medium", "Hard", "Easy"]
       }
       return pd.DataFrame(data)
   ```

6. **UI Components (`ui_components.py`)**
   - This file can contain reusable UI components for displaying player cards, team formations, etc. For simplicity, we will not implement this in detail here.

7. **Utility Functions (`utils.py`)**
   - This file can contain utility functions for logging, error handling, etc. For simplicity, we will not implement this in detail here.

8. **README File (`README.md`)**
   ```markdown
   # Fantasy Football Advanced Filters

   This project implements advanced filters for team selection in a fantasy football application.

   ## Features
   - Filter players by price, team, position, and fixture difficulty.
   - Save preferred filter settings for future use.

   ## Requirements
   - Python 3.x
   - Streamlit
   - Pandas
   - Requests

   ## How to Run
   1. Install the required packages:
      ```bash
      pip install -r requirements.txt
      ```
   2. Run the application:
      ```bash
      streamlit run app.py
      ```
   ```

### Running the Project

1. **Install Dependencies**: Run `pip install -r requirements.txt` to install the required packages.
2. **Run the Application**: Use the command `streamlit run app.py` to start the Streamlit application.
3. **Interact with Filters**: Use the sidebar to filter players based on your criteria and save your settings.

### Conclusion

This project provides a basic structure for implementing advanced filters in a fantasy football application. You can expand upon this by adding more features, improving the UI, and integrating with real data sources.