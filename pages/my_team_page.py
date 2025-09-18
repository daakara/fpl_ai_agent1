"""
My Team Page - Handles FPL team import and analysis functionality
"""
import streamlit as st
import pandas as pd
from services.fpl_data_service import FPLDataService


class MyTeamPage:
    """Handles My FPL Team functionality"""
    
    def __init__(self):
        self.data_service = FPLDataService()
    
    def render(self):
        """Main render method for My Team page"""
        st.header("👤 My FPL Team")
        
        # Check if team is loaded
        if not st.session_state.get('my_team_loaded', False):
            self._render_team_import_section()
            return
        
        # Display loaded team
        team_data = st.session_state.my_team_data
        
        # Team overview
        st.subheader(f"🏆 {team_data.get('entry_name', 'Your Team')} (ID: {st.session_state.my_team_id})")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Overall Rank", f"{team_data.get('summary_overall_rank', 'N/A'):,}")
        
        with col2:
            st.metric("Total Points", f"{team_data.get('summary_overall_points', 'N/A'):,}")
        
        with col3:
            st.metric("Gameweek Rank", f"{team_data.get('summary_event_rank', 'N/A'):,}")
        
        with col4:
            st.metric("Team Value", f"£{team_data.get('value', 1000)/10:.1f}m")
        
        # Team analysis tabs
        tab1, tab2, tab3 = st.tabs([
            "👥 Current Squad", 
            "📊 Performance Analysis", 
            "💡 Recommendations"
        ])
        
        with tab1:
            self._display_current_squad(team_data)
        
        with tab2:
            self._display_performance_analysis(team_data)
        
        with tab3:
            self._display_recommendations(team_data)
        
        # Reset team button
        if st.button("🔄 Load Different Team"):
            st.session_state.my_team_loaded = False
            st.rerun()
    
    def _render_team_import_section(self):
        """Render team import interface"""
        st.subheader("📥 Import Your FPL Team")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            team_id = st.text_input(
                "Enter your FPL Team ID:",
                placeholder="e.g., 1234567",
                help="Find your team ID in your FPL team URL"
            )
            
            # Get current gameweek for selection
            current_gw = self.data_service.get_current_gameweek()
            gameweeks = list(range(1, min(current_gw + 1, 39)))
            
            selected_gw = st.selectbox(
                "Select Gameweek:",
                gameweeks,
                index=min(current_gw - 1, 37) if current_gw else 0,
                help="Choose which gameweek's team to analyze"
            )
        
        with col2:
            if st.button("🔄 Load My Team", type="primary"):
                if team_id:
                    self._load_team_data(team_id, selected_gw)
                else:
                    st.warning("Please enter a team ID")
        
        # Instructions
        with st.expander("💡 How to find your Team ID", expanded=False):
            st.markdown("""
            **Step 1:** Go to the official FPL website and log in
            
            **Step 2:** Navigate to your team page
            
            **Step 3:** Look at the URL - it will look like:
            `https://fantasy.premierleague.com/entry/1234567/event/15`
            
            **Step 4:** Your Team ID is the number after `/entry/` (in this example: 1234567)
            
            **Note:** Your team must be public for this to work. You can change this in your FPL account settings.
            """)
    
    def _load_team_data(self, team_id, gameweek):
        """Load team data from FPL API"""
        with st.spinner("Loading your team..."):
            team_data = self.data_service.load_team_data(team_id, gameweek)
            
            if team_data:
                st.session_state.my_team_data = team_data
                st.session_state.my_team_id = team_id
                st.session_state.my_team_gameweek = gameweek
                st.session_state.my_team_loaded = True
                st.success("✅ Team loaded successfully!")
                st.rerun()
            else:
                st.error("❌ Could not load team. Please check your team ID.")
    
    def _display_current_squad(self, team_data):
        """Display current squad with player details"""
        st.subheader("👥 Current Squad")
        
        if not st.session_state.get('data_loaded', False):
            st.warning("Load player data to see detailed squad analysis")
            return
        
        picks = team_data.get('picks', [])
        if not picks:
            st.warning("No squad data available")
            return
        
        players_df = st.session_state.players_df
        
        # Match picks with player data
        squad_data = []
        formation_counts = {'Goalkeeper': 0, 'Defender': 0, 'Midfielder': 0, 'Forward': 0}
        
        for pick in picks:
            player_info = players_df[players_df['id'] == pick['element']]
            if not player_info.empty:
                player = player_info.iloc[0]
                position = player.get('position_name', 'Unknown')
                
                # Count for formation
                if position in formation_counts:
                    formation_counts[position] += 1
                
                squad_data.append({
                    'Player': player.get('web_name', 'Unknown'),
                    'Position': position,
                    'Team': player.get('team_short_name', 'UNK'),
                    'Price': f"£{player.get('cost_millions', 0):.1f}m",
                    'Points': player.get('total_points', 0),
                    'Form': f"{player.get('form', 0):.1f}",
                    'Status': '(C)' if pick.get('is_captain') else '(VC)' if pick.get('is_vice_captain') else '',
                    'Playing': '✅' if pick.get('position', 12) <= 11 else '🪑'
                })
        
        if squad_data:
            # Formation display
            formation_str = f"{formation_counts['Goalkeeper']}-{formation_counts['Defender']}-{formation_counts['Midfielder']}-{formation_counts['Forward']}"
            st.info(f"**Formation:** {formation_str}")
            
            # Squad table
            squad_df = pd.DataFrame(squad_data)
            st.dataframe(
                squad_df, 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "Price": st.column_config.TextColumn("Price"),
                    "Points": st.column_config.NumberColumn("Points"),
                    "Form": st.column_config.TextColumn("Form"),
                    "Playing": st.column_config.TextColumn("Starting?")
                }
            )
            
            # Squad statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_value = sum([float(row['Price'].replace('£', '').replace('m', '')) for row in squad_data])
                st.metric("Squad Value", f"£{total_value:.1f}m")
            
            with col2:
                total_points = sum([row['Points'] for row in squad_data])
                st.metric("Total Points", f"{total_points:,}")
            
            with col3:
                avg_form = sum([float(row['Form']) for row in squad_data]) / len(squad_data)
                st.metric("Average Form", f"{avg_form:.1f}")
            
            with col4:
                starting_players = len([row for row in squad_data if row['Playing'] == '✅'])
                st.metric("Starting XI", f"{starting_players}/11")
    
    def _display_performance_analysis(self, team_data):
        """Display performance analysis"""
        st.subheader("📊 Performance Analysis")
        
        # Basic performance metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**📈 Season Performance**")
            total_points = team_data.get('summary_overall_points', 0)
            overall_rank = team_data.get('summary_overall_rank', 0)
            
            current_gw = team_data.get('gameweek', 1)
            avg_points_per_gw = total_points / max(current_gw, 1)
            
            st.metric("Total Points", f"{total_points:,}")
            st.metric("Points/GW", f"{avg_points_per_gw:.1f}")
            st.metric("Overall Rank", f"{overall_rank:,}" if overall_rank else "N/A")
        
        with col2:
            st.write("**🎯 Recent Performance**")
            gw_points = team_data.get('summary_event_points', 0)
            gw_rank = team_data.get('summary_event_rank', 0)
            
            # Performance indicators
            avg_gw_points = 50  # League average
            performance = "🔥 Excellent" if gw_points >= 70 else "👍 Good" if gw_points >= avg_gw_points else "📈 Below Average"
            
            st.metric("GW Points", f"{gw_points}")
            st.metric("GW Rank", f"{gw_rank:,}" if gw_rank else "N/A")
            st.metric("Performance", performance)
        
        with col3:
            st.write("**💰 Team Value**")
            team_value = team_data.get('value', 1000) / 10
            bank = team_data.get('bank', 0) / 10
            total_budget = team_value + bank
            
            st.metric("Team Value", f"£{team_value:.1f}m")
            st.metric("In Bank", f"£{bank:.1f}m")
            st.metric("Total Budget", f"£{total_budget:.1f}m")
        
        # Performance insights
        st.write("**💡 Performance Insights**")
        
        if overall_rank:
            total_players = 8000000  # Approximate
            percentile = (1 - (overall_rank / total_players)) * 100
            
            if percentile >= 90:
                st.success("🏆 Elite performance - Top 10% of all managers!")
            elif percentile >= 75:
                st.info("🥇 Excellent performance - Top 25% of all managers!")
            elif percentile >= 50:
                st.info("👍 Above average performance - Top 50% of all managers")
            else:
                st.warning("📈 Room for improvement - Focus on consistency and transfers")
    
    def _display_recommendations(self, team_data):
        """Display recommendations for the team"""
        st.subheader("💡 Team Recommendations")
        
        # Basic recommendations based on available data
        recommendations = []
        
        # Bank analysis
        bank = team_data.get('bank', 0) / 10
        if bank > 2:
            recommendations.append({
                'type': 'info',
                'title': 'Unused Funds',
                'message': f"You have £{bank:.1f}m in the bank. Consider upgrading a player to improve your team."
            })
        
        # Recent performance
        gw_points = team_data.get('summary_event_points', 0)
        if gw_points < 40:
            recommendations.append({
                'type': 'warning',
                'title': 'Low Gameweek Score',
                'message': "Your recent gameweek score was below average. Consider reviewing your captain choice and active players."
            })
        
        # Overall rank
        overall_rank = team_data.get('summary_overall_rank', 0)
        if overall_rank and overall_rank > 1000000:
            recommendations.append({
                'type': 'info',
                'title': 'Rank Improvement',
                'message': "Focus on consistent captain choices and popular template players to improve your rank."
            })
        
        # Display recommendations
        if recommendations:
            for rec in recommendations:
                if rec['type'] == 'success':
                    st.success(f"✅ **{rec['title']}**: {rec['message']}")
                elif rec['type'] == 'warning':
                    st.warning(f"⚠️ **{rec['title']}**: {rec['message']}")
                else:
                    st.info(f"💡 **{rec['title']}**: {rec['message']}")
        else:
            st.success("🎉 Your team looks good! Keep monitoring form and fixtures for optimal transfers.")
        
        # General advice
        st.write("**📚 General FPL Tips:**")
        st.write("• Monitor player form and upcoming fixtures")
        st.write("• Use your free transfer each gameweek")
        st.write("• Plan your chip usage strategically")
        st.write("• Consider differential picks to climb ranks")
        st.write("• Stay active in the transfer market")

