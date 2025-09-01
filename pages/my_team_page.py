"""
My Team Page - Comprehensive team analysis and recommendations
"""
import streamlit as st
import pandas as pd
from typing import Dict, List, Optional


class MyTeamPage:
    """My team page with comprehensive analysis and recommendations"""
    
    def __init__(self):
        """Initialize the my team page"""
        pass
    
    def render(self):
        """Render the my team page"""
        st.header("🏠 My FPL Team")
        
        # Team import section
        with st.expander("📥 Import Your FPL Team", expanded=False):
            team_id = st.text_input(
                "Enter your FPL Team ID:",
                help="Find your team ID in the URL when viewing your team on the FPL website"
            )
            
            if st.button("Load My Team"):
                if team_id:
                    self._load_user_team(team_id)
                else:
                    st.warning("Please enter your team ID")
        
        # Check if team data is available
        if 'my_team_df' not in st.session_state or st.session_state.my_team_df.empty:
            st.info("Import your team using the section above to see detailed analysis!")
            return
        
        my_team_df = st.session_state.my_team_df
        
        # Create tabs for different analysis sections
        tab1, tab2, tab3, tab4 = st.tabs([
            "👥 Current Squad",
            "📊 Performance Analysis",
            "🔄 Transfer Suggestions", 
            "📈 Benchmarking"
        ])
        
        with tab1:
            self._render_current_squad(my_team_df)
        
        with tab2:
            self._render_performance_analysis(my_team_df)
        
        with tab3:
            self._render_transfer_suggestions(my_team_df)
        
        with tab4:
            self._render_benchmarking_analysis(my_team_df)
    
    def _load_user_team(self, team_id):
        """Load user team data from FPL API"""
        try:
            import requests
            
            # Load team picks
            url = f"https://fantasy.premierleague.com/api/entry/{team_id}/event/1/picks/"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            picks_data = response.json()
            
            # Get player IDs from picks
            player_ids = [pick['element'] for pick in picks_data['picks']]
            
            # Match with loaded players data
            if 'players_df' in st.session_state and not st.session_state.players_df.empty:
                players_df = st.session_state.players_df
                my_team_df = players_df[players_df.index.isin([pid-1 for pid in player_ids])].copy()
                
                # Add squad information
                for pick in picks_data['picks']:
                    player_idx = pick['element'] - 1
                    if player_idx in my_team_df.index:
                        my_team_df.loc[player_idx, 'is_captain'] = pick['is_captain']
                        my_team_df.loc[player_idx, 'is_vice_captain'] = pick['is_vice_captain']
                        my_team_df.loc[player_idx, 'multiplier'] = pick['multiplier']
                
                st.session_state.my_team_df = my_team_df
                st.session_state.team_id = team_id
                st.success(f"Successfully loaded team with {len(my_team_df)} players!")
            else:
                st.error("Please load FPL data first using the sidebar.")
                
        except Exception as e:
            st.error(f"Error loading team: {str(e)}")
            st.info("Make sure your team ID is correct and your team is public.")
    
    def _render_current_squad(self, my_team_df):
        """Render current squad analysis"""
        st.subheader("👥 Current Squad Analysis")
        
        if my_team_df.empty:
            st.warning("No team data available.")
            return
        
        # Squad overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_cost = my_team_df['now_cost'].sum()
            st.metric("Squad Value", f"£{total_cost:.1f}m")
        
        with col2:
            total_points = my_team_df['total_points'].sum()
            st.metric("Total Points", f"{total_points:,}")
        
        with col3:
            avg_points = my_team_df['total_points'].mean()
            st.metric("Avg Points/Player", f"{avg_points:.1f}")
        
        with col4:
            if 'points_per_million' in my_team_df.columns:
                avg_ppm = my_team_df['points_per_million'].mean()
                st.metric("Avg Points/Million", f"{avg_ppm:.1f}")
        
        # Squad formation
        if 'position' in my_team_df.columns:
            formation = my_team_df['position'].value_counts()
            formation_str = f"{formation.get('GK', 0)}-{formation.get('DEF', 0)}-{formation.get('MID', 0)}-{formation.get('FWD', 0)}"
            st.write(f"**Formation**: {formation_str}")
        
        # Enhanced squad table
        st.write("**🔍 Squad Details:**")
        
        # Prepare display columns
        display_cols = ['web_name', 'team_name', 'position', 'now_cost', 'total_points']
        if 'form' in my_team_df.columns:
            display_cols.append('form')
        if 'selected_by_percent' in my_team_df.columns:
            display_cols.append('selected_by_percent')
        if 'points_per_million' in my_team_df.columns:
            display_cols.append('points_per_million')
        
        # Add captain indicators
        display_df = my_team_df[display_cols].copy()
        
        # Add status column
        status_list = []
        for idx, row in my_team_df.iterrows():
            if row.get('is_captain', False):
                status_list.append('👑 Captain')
            elif row.get('is_vice_captain', False):
                status_list.append('🅒 Vice Captain')
            else:
                status_list.append('✅ Playing')
        
        display_df['Status'] = status_list
        
        st.dataframe(display_df, use_container_width=True)
        
        # Squad insights
        self._render_squad_insights(my_team_df)
    
    def _render_squad_insights(self, my_team_df):
        """Render intelligent squad insights"""
        st.write("**💡 Squad Insights:**")
        
        insights = []
        
        # Value analysis
        if 'points_per_million' in my_team_df.columns:
            low_value_players = my_team_df[my_team_df['points_per_million'] < 5.0]
            if not low_value_players.empty:
                insights.append(f"⚠️ {len(low_value_players)} players with poor value (PPM < 5.0)")
        
        # Form analysis
        if 'form' in my_team_df.columns:
            poor_form = my_team_df[pd.to_numeric(my_team_df['form'], errors='coerce').fillna(0) < 3.0]
            if not poor_form.empty:
                insights.append(f"📉 {len(poor_form)} players in poor form (< 3.0)")
        
        # Ownership analysis
        if 'selected_by_percent' in my_team_df.columns:
            high_ownership = my_team_df[my_team_df['selected_by_percent'] > 50.0]
            if not high_ownership.empty:
                insights.append(f"📈 {len(high_ownership)} highly owned players (> 50%)")
        
        # Display insights
        if insights:
            for insight in insights:
                st.write(insight)
        else:
            st.write("✅ Squad looks healthy!")
    
    def _render_performance_analysis(self, my_team_df):
        """Render performance analysis"""
        st.subheader("📊 Performance Analysis")
        
        if my_team_df.empty:
            st.warning("No team data available for performance analysis.")
            return
        
        # Performance metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🎯 Top Performers:**")
            if 'total_points' in my_team_df.columns:
                top_performers = my_team_df.nlargest(5, 'total_points')[['web_name', 'total_points', 'position']]
                st.dataframe(top_performers, use_container_width=True)
        
        with col2:
            st.write("**📉 Underperformers:**")
            if 'total_points' in my_team_df.columns:
                underperformers = my_team_df.nsmallest(5, 'total_points')[['web_name', 'total_points', 'position']]
                st.dataframe(underperformers, use_container_width=True)
        
        # Form analysis
        if 'form' in my_team_df.columns:
            st.write("**📈 Form Analysis:**")
            form_analysis = my_team_df[['web_name', 'form', 'position']].copy()
            form_analysis['form'] = pd.to_numeric(form_analysis['form'], errors='coerce').fillna(0)
            form_analysis = form_analysis.sort_values('form', ascending=False)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**🔥 Best Form:**")
                st.dataframe(form_analysis.head(5), use_container_width=True)
            
            with col2:
                st.write("**❄️ Worst Form:**")
                st.dataframe(form_analysis.tail(5), use_container_width=True)
    
    def _render_transfer_suggestions(self, my_team_df):
        """Render transfer suggestions"""
        st.subheader("🔄 Transfer Suggestions")
        
        if my_team_df.empty:
            st.warning("No team data available for transfer suggestions.")
            return
        
        # Transfer out candidates
        st.write("**📤 Consider Transferring Out:**")
        
        transfer_out_reasons = []
        
        # Poor value players
        if 'points_per_million' in my_team_df.columns:
            poor_value = my_team_df[my_team_df['points_per_million'] < 4.0]
            for _, player in poor_value.iterrows():
                transfer_out_reasons.append({
                    'Player': player['web_name'],
                    'Reason': f"Poor value (PPM: {player['points_per_million']:.1f})",
                    'Priority': 'High'
                })
        
        # Poor form players
        if 'form' in my_team_df.columns:
            poor_form = my_team_df[pd.to_numeric(my_team_df['form'], errors='coerce').fillna(0) < 2.5]
            for _, player in poor_form.iterrows():
                form_val = pd.to_numeric(player['form'], errors='coerce')
                transfer_out_reasons.append({
                    'Player': player['web_name'],
                    'Reason': f"Poor form ({form_val:.1f})",
                    'Priority': 'Medium'
                })
        
        if transfer_out_reasons:
            transfer_out_df = pd.DataFrame(transfer_out_reasons)
            st.dataframe(transfer_out_df, use_container_width=True)
        else:
            st.write("✅ No obvious transfer out candidates")
        
        # Transfer in suggestions (if players data available)
        if 'players_df' in st.session_state:
            st.write("**📥 Consider Transferring In:**")
            players_df = st.session_state.players_df
            
            # Find players not in current team
            current_player_ids = my_team_df.index.tolist()
            available_players = players_df[~players_df.index.isin(current_player_ids)]
            
            if not available_players.empty and 'points_per_million' in available_players.columns:
                # Top value players by position
                for position in ['GK', 'DEF', 'MID', 'FWD']:
                    pos_players = available_players[available_players.get('position', '') == position]
                    if not pos_players.empty:
                        top_value = pos_players.nlargest(3, 'points_per_million')
                        st.write(f"**{position} Targets:**")
                        st.dataframe(
                            top_value[['web_name', 'team_name', 'now_cost', 'points_per_million']],
                            use_container_width=True
                        )
    
    def _render_benchmarking_analysis(self, my_team_df):
        """Render benchmarking analysis"""
        st.subheader("📈 Benchmarking & Goals")
        
        if my_team_df.empty:
            st.warning("No team data available for benchmarking.")
            return
        
        # Performance comparison
        st.write("**🏆 Performance vs. Averages:**")
        
        # Compare with overall averages if available
        if 'players_df' in st.session_state:
            players_df = st.session_state.players_df
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                my_avg_points = my_team_df['total_points'].mean()
                overall_avg_points = players_df['total_points'].mean()
                diff = my_avg_points - overall_avg_points
                st.metric("Avg Points vs Overall", f"{my_avg_points:.1f}", f"{diff:+.1f}")
            
            with col2:
                if 'points_per_million' in my_team_df.columns:
                    my_avg_ppm = my_team_df['points_per_million'].mean()
                    overall_avg_ppm = players_df['points_per_million'].mean()
                    diff_ppm = my_avg_ppm - overall_avg_ppm
                    st.metric("Avg PPM vs Overall", f"{my_avg_ppm:.1f}", f"{diff_ppm:+.1f}")
            
            with col3:
                my_total_value = my_team_df['now_cost'].sum()
                st.metric("Squad Value", f"£{my_total_value:.1f}m")
        
        # Season goals and progress
        st.write("**🎯 Season Goals & Progress:**")
        
        with st.expander("Set Season Targets"):
            target_rank = st.number_input("Target Overall Rank:", min_value=1, value=100000, step=1000)
            target_points = st.number_input("Target Total Points:", min_value=1000, value=2000, step=50)
            
            if st.button("Save Targets"):
                st.session_state.target_rank = target_rank
                st.session_state.target_points = target_points
                st.success("Targets saved!")
        
        # Display current targets if set
        if 'target_rank' in st.session_state and 'target_points' in st.session_state:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Target Rank", f"{st.session_state.target_rank:,}")
                
            with col2:
                current_points = my_team_df['total_points'].sum()
                points_to_target = st.session_state.target_points - current_points
                st.metric("Points to Target", f"{st.session_state.target_points:,}", f"{points_to_target:+,}")
        
        # Improvement suggestions
        st.write("**💡 Improvement Suggestions:**")
        
        suggestions = [
            "🔄 Monitor player form and make timely transfers",
            "📅 Plan transfers around fixture difficulty",
            "👑 Optimize captaincy choices for maximum points",
            "💎 Look for differential players to gain rank",
            "⏰ Time your chips (Wildcard, Bench Boost, etc.) strategically"
        ]
        
        for suggestion in suggestions:
            st.write(suggestion)

