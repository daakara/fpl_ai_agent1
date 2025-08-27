import streamlit as st
from filters import FilterManager
from data_loader import load_player_data
from fpl_myteam import load_my_fpl_team  # Import the function to load the user's FPL team

def main():
    st.title("Fantasy Football Team Selection")

    # Create tabs
    tab1, tab2 = st.tabs(["Player Filters", "Import My Team"])

    # Tab 1: Player Filters
    with tab1:
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

    # Tab 2: Import My Team
    with tab2:
        st.header("Import Your FPL Team")
        team_id = st.text_input("Enter your FPL Team ID:")
        
        if st.button("Fetch My Team"):
            if team_id:
                try:
                    # Fetch the user's FPL team data
                    team_data = load_my_fpl_team(int(team_id))
                    if team_data:
                        st.success("Team data fetched successfully!")
                        st.json(team_data)  # Display the team data in JSON format
                    else:
                        st.error("Failed to fetch team data. Please check your Team ID.")
                except ValueError:
                    st.error("Please enter a valid Team ID.")
            else:
                st.error("Please enter your FPL Team ID.")

if __name__ == "__main__":
    main()