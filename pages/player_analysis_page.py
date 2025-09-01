"""
Player Analysis Page - Comprehensive player statistics and analysis
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class PlayerAnalysisPage:
    """Player analysis page with comprehensive filtering and insights"""
    
    def __init__(self):
        """Initialize the player analysis page"""
        pass
    
    def render(self):
        """Render the player analysis page"""
        st.header("👥 Player Analysis")
        
        # Check if data is loaded
        if 'players_df' not in st.session_state or st.session_state.players_df.empty:
            st.warning("Please load FPL data first using the sidebar.")
            return
        
        players_df = st.session_state.players_df
        
        # Create tabs for different analysis sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔍 Smart Filters & Overview",
            "📊 Performance Metrics", 
            "⚖️ Player Comparison",
            "🎯 Position Analysis",
            "💡 AI Insights"
        ])
        
        with tab1:
            self._render_enhanced_player_filters(players_df)
        
        with tab2:
            self._render_performance_metrics_dashboard(players_df)
        
        with tab3:
            self._render_player_comparison_tool(players_df)
        
        with tab4:
            self._render_position_specific_analysis(players_df)
        
        with tab5:
            self._render_ai_player_insights(players_df)
    
    def _render_enhanced_player_filters(self, df):
        """Enhanced filtering system with smart results"""
        st.subheader("🔍 Smart Player Filters")
        
        if df.empty:
            st.warning("No player data available.")
            return
        
        # Filter controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            positions = ['All'] + sorted(df['position'].unique().tolist()) if 'position' in df.columns else ['All']
            selected_position = st.selectbox("Position:", positions, key="player_position_filter")
        
        with col2:
            teams = ['All'] + sorted(df['team_name'].unique().tolist()) if 'team_name' in df.columns else ['All']
            selected_team = st.selectbox("Team:", teams, key="player_team_filter")
        
        with col3:
            min_price, max_price = float(df['now_cost'].min()), float(df['now_cost'].max())
            price_range = st.slider("Price Range (£m):", min_price, max_price, (min_price, max_price), 0.1)
        
        # Advanced filters
        with st.expander("🎯 Advanced Filters"):
            col4, col5, col6 = st.columns(3)
            
            with col4:
                min_points = st.number_input("Min Total Points:", min_value=0, value=0)
            
            with col5:
                min_form = st.number_input("Min Form:", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
            
            with col6:
                min_minutes = st.number_input("Min Minutes:", min_value=0, value=0)
        
        # Apply filters
        filtered_df = df.copy()
        
        if selected_position != 'All':
            filtered_df = filtered_df[filtered_df['position'] == selected_position]
        
        if selected_team != 'All':
            filtered_df = filtered_df[filtered_df['team_name'] == selected_team]
        
        filtered_df = filtered_df[
            (filtered_df['now_cost'] >= price_range[0]) & 
            (filtered_df['now_cost'] <= price_range[1])
        ]
        
        if min_points > 0:
            filtered_df = filtered_df[filtered_df['total_points'] >= min_points]
        
        if min_form > 0:
            filtered_df = filtered_df[pd.to_numeric(filtered_df['form'], errors='coerce').fillna(0) >= min_form]
        
        if min_minutes > 0:
            filtered_df = filtered_df[filtered_df['minutes'] >= min_minutes]
        
        # Display results
        if not filtered_df.empty:
            st.success(f"Found {len(filtered_df)} players matching your criteria")
            
            # Enhanced display columns
            display_columns = self._get_enhanced_display_columns(filtered_df)
            formatted_df = self._format_player_dataframe(filtered_df[display_columns])
            
            st.dataframe(
                formatted_df,
                use_container_width=True,
                column_config=self._get_column_config()
            )
            
            # Summary statistics
            self._display_filtered_stats_summary(filtered_df)
        else:
            st.warning("No players match your filter criteria.")
    
    def _render_performance_metrics_dashboard(self, df):
        """Performance metrics dashboard with sub-tabs"""
        st.subheader("📊 Performance Metrics Dashboard")
        
        if df.empty:
            st.warning("No data available for performance analysis.")
            return
        
        # Create sub-tabs for different metric categories
        subtab1, subtab2, subtab3 = st.tabs([
            "⚽ Attacking Metrics",
            "🛡️ Defensive Metrics", 
            "📊 General Performance"
        ])
        
        with subtab1:
            self._render_attacking_metrics(df)
        
        with subtab2:
            self._render_defensive_metrics(df)
        
        with subtab3:
            self._render_general_performance_metrics(df)
    
    def _render_attacking_metrics(self, df):
        """Render attacking performance metrics"""
        st.write("**⚽ Attacking Performance Analysis**")
        
        # Expected Goals analysis
        xg_col = None
        for col in ['expected_goals', 'xG', 'expected_goals_per_90']:
            if col in df.columns:
                xg_col = col
                break
        
        if xg_col:
            df_copy = df.copy()
            df_copy[xg_col] = pd.to_numeric(df_copy[xg_col], errors='coerce').fillna(0)
            xg_players = df_copy[df_copy[xg_col] > 0]
            
            if not xg_players.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**🎯 Top xG Leaders**")
                    top_xg = xg_players.nlargest(10, xg_col)[['web_name', 'team_name', xg_col, 'goals_scored']]
                    st.dataframe(top_xg, use_container_width=True)
                
                with col2:
                    st.write("**⚽ xG vs Actual Goals**")
                    for _, player in top_xg.head(5).iterrows():
                        xg_val = player[xg_col]
                        goals = player.get('goals_scored', 0)
                        efficiency = "🔥" if goals > xg_val else "📉" if goals < xg_val * 0.7 else "➡️"
                        st.write(f"{efficiency} {player['web_name']}: {goals}G vs {xg_val:.1f}xG")
        
        # Goals and assists
        col1, col2 = st.columns(2)
        
        with col1:
            if 'goals_scored' in df.columns:
                st.write("**🥅 Top Goal Scorers**")
                top_scorers = df[df['goals_scored'] > 0].nlargest(10, 'goals_scored')
                if not top_scorers.empty:
                    st.dataframe(top_scorers[['web_name', 'team_name', 'goals_scored', 'now_cost']], use_container_width=True)
        
        with col2:
            if 'assists' in df.columns:
                st.write("**🎯 Top Assist Providers**")
                top_assists = df[df['assists'] > 0].nlargest(10, 'assists')
                if not top_assists.empty:
                    st.dataframe(top_assists[['web_name', 'team_name', 'assists', 'now_cost']], use_container_width=True)
    
    def _render_defensive_metrics(self, df):
        """Render defensive performance metrics"""
        st.write("**🛡️ Defensive Performance Analysis**")
        
        # Filter for defensive players (GK and DEF)
        defensive_df = df[df['position'].isin(['GK', 'DEF'])] if 'position' in df.columns else df
        
        if defensive_df.empty:
            st.info("No defensive players found.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'clean_sheets' in defensive_df.columns:
                st.write("**🏆 Clean Sheet Leaders**")
                clean_sheet_leaders = defensive_df[defensive_df['clean_sheets'] > 0].nlargest(10, 'clean_sheets')
                if not clean_sheet_leaders.empty:
                    st.dataframe(clean_sheet_leaders[['web_name', 'team_name', 'clean_sheets', 'position']], use_container_width=True)
        
        with col2:
            if 'saves' in defensive_df.columns:
                st.write("**🥅 Save Masters (GK)**")
                gk_df = defensive_df[defensive_df['position'] == 'GK'] if 'position' in defensive_df.columns else defensive_df
                save_leaders = gk_df[gk_df['saves'] > 0].nlargest(10, 'saves')
                if not save_leaders.empty:
                    st.dataframe(save_leaders[['web_name', 'team_name', 'saves', 'total_points']], use_container_width=True)
    
    def _render_general_performance_metrics(self, df):
        """Render general performance metrics"""
        st.write("**📊 General Performance Analysis**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**💰 Best Value (Points per Million)**")
            if 'points_per_million' in df.columns:
                best_value = df[df['total_points'] > 0].nlargest(10, 'points_per_million')
                if not best_value.empty:
                    st.dataframe(best_value[['web_name', 'team_name', 'total_points', 'now_cost', 'points_per_million']], use_container_width=True)
        
        with col2:
            st.write("**⭐ Bonus Point Masters**")
            if 'bonus' in df.columns:
                bonus_leaders = df[df['bonus'] > 0].nlargest(10, 'bonus')
                if not bonus_leaders.empty:
                    st.dataframe(bonus_leaders[['web_name', 'team_name', 'bonus', 'total_points']], use_container_width=True)
    
    def _render_player_comparison_tool(self, df):
        """Player comparison tool"""
        st.subheader("⚖️ Player Comparison Tool")
        
        if df.empty:
            st.warning("No data available for comparison.")
            return
        
        # Player selection
        st.write("**Select players to compare (up to 4):**")
        player_names = df['web_name'].tolist() if 'web_name' in df.columns else []
        
        selected_players = st.multiselect(
            "Choose players:",
            player_names,
            max_selections=4,
            key="player_comparison"
        )
        
        if len(selected_players) >= 2:
            comparison_df = df[df['web_name'].isin(selected_players)]
            
            # Key comparison metrics
            comparison_cols = [
                'web_name', 'team_name', 'position', 'now_cost', 'total_points', 
                'form', 'selected_by_percent', 'points_per_million'
            ]
            
            # Add available advanced metrics
            for col in ['goals_scored', 'assists', 'clean_sheets', 'saves', 'bonus', 'minutes']:
                if col in df.columns:
                    comparison_cols.append(col)
            
            # Display comparison
            available_cols = [col for col in comparison_cols if col in comparison_df.columns]
            comparison_display = comparison_df[available_cols].set_index('web_name').T
            
            st.dataframe(comparison_display, use_container_width=True)
            
            # Generate insights
            self._generate_comparison_insights(comparison_df)
        else:
            st.info("Please select at least 2 players to compare.")
    
    def _render_position_specific_analysis(self, df):
        """Position-specific analysis"""
        st.subheader("🎯 Position-Specific Analysis")
        
        if 'position' not in df.columns:
            st.warning("Position data not available.")
            return
        
        position = st.selectbox(
            "Select position to analyze:",
            ['GK', 'DEF', 'MID', 'FWD'],
            key="position_analysis"
        )
        
        position_df = df[df['position'] == position]
        
        if position_df.empty:
            st.warning(f"No {position} players found.")
            return
        
        if position == 'GK':
            self._render_goalkeeper_analysis(position_df)
        elif position == 'DEF':
            self._render_defender_analysis(position_df)
        elif position == 'MID':
            self._render_midfielder_analysis(position_df)
        elif position == 'FWD':
            self._render_forward_analysis(position_df)
    
    def _render_ai_player_insights(self, df):
        """AI-powered player insights and recommendations"""
        st.subheader("💡 AI Player Insights & Recommendations")
        
        if df.empty:
            st.warning("No data available for AI analysis.")
            return
        
        # Create sub-tabs for different types of recommendations
        subtab1, subtab2, subtab3 = st.tabs([
            "🎯 Smart Picks",
            "💎 Hidden Gems", 
            "⚠️ Players to Avoid"
        ])
        
        with subtab1:
            self._render_smart_picks(df)
        
        with subtab2:
            self._render_hidden_gems(df)
        
        with subtab3:
            self._render_players_to_avoid(df)
    
    # Helper methods
    def _get_enhanced_display_columns(self, df):
        """Get enhanced display columns based on available data"""
        base_columns = ['web_name', 'team_name', 'position', 'now_cost', 'total_points']
        
        optional_columns = [
            'form', 'selected_by_percent', 'points_per_million', 'goals_scored', 
            'assists', 'clean_sheets', 'saves', 'bonus', 'minutes'
        ]
        
        available_columns = base_columns + [col for col in optional_columns if col in df.columns]
        return available_columns
    
    def _format_player_dataframe(self, df):
        """Format dataframe for better display"""
        df_copy = df.copy()
        
        # Format price columns
        if 'now_cost' in df_copy.columns:
            df_copy['now_cost'] = df_copy['now_cost'].apply(lambda x: f"£{x:.1f}m")
        
        # Format percentage columns
        if 'selected_by_percent' in df_copy.columns:
            df_copy['selected_by_percent'] = df_copy['selected_by_percent'].apply(lambda x: f"{x:.1f}%")
        
        # Format decimal columns
        if 'points_per_million' in df_copy.columns:
            df_copy['points_per_million'] = df_copy['points_per_million'].apply(lambda x: f"{x:.1f}")
        
        return df_copy
    
    def _get_column_config(self):
        """Get column configuration for dataframe display"""
        return {
            "web_name": "Player",
            "team_name": "Team", 
            "position": "Pos",
            "now_cost": "Price",
            "total_points": "Points",
            "form": "Form",
            "selected_by_percent": "Owned %",
            "points_per_million": "PPM"
        }
    
    def _display_filtered_stats_summary(self, df):
        """Display summary statistics for filtered results"""
        if df.empty:
            return
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_cost = df['now_cost'].mean()
            st.metric("Average Price", f"£{avg_cost:.1f}m")
        
        with col2:
            avg_points = df['total_points'].mean()
            st.metric("Average Points", f"{avg_points:.1f}")
        
        with col3:
            if 'points_per_million' in df.columns:
                avg_ppm = df['points_per_million'].mean()
                st.metric("Average PPM", f"{avg_ppm:.1f}")
        
        with col4:
            if 'selected_by_percent' in df.columns:
                avg_ownership = df['selected_by_percent'].mean()
                st.metric("Average Owned %", f"{avg_ownership:.1f}%")
    
    # Placeholder methods for position-specific analysis
    def _render_goalkeeper_analysis(self, df):
        """Goalkeeper-specific analysis"""
        st.write("**🥅 Goalkeeper Analysis**")
        # Implementation would go here
        st.info("Goalkeeper analysis coming soon!")
    
    def _render_defender_analysis(self, df):
        """Defender-specific analysis"""
        st.write("**🛡️ Defender Analysis**")
        # Implementation would go here
        st.info("Defender analysis coming soon!")
    
    def _render_midfielder_analysis(self, df):
        """Midfielder-specific analysis"""
        st.write("**⚽ Midfielder Analysis**")
        # Implementation would go here
        st.info("Midfielder analysis coming soon!")
    
    def _render_forward_analysis(self, df):
        """Forward-specific analysis"""
        st.write("**🎯 Forward Analysis**")
        # Implementation would go here
        st.info("Forward analysis coming soon!")
    
    def _render_smart_picks(self, df):
        """Smart pick recommendations"""
        st.write("**🎯 AI Smart Picks**")
        # Implementation would go here
        st.info("AI smart picks coming soon!")
    
    def _render_hidden_gems(self, df):
        """Hidden gem recommendations"""
        st.write("**💎 Hidden Gems**")
        # Implementation would go here
        st.info("Hidden gems analysis coming soon!")
    
    def _render_players_to_avoid(self, df):
        """Players to avoid recommendations"""
        st.write("**⚠️ Players to Avoid**")
        # Implementation would go here
        st.info("Players to avoid analysis coming soon!")
    
    def _generate_comparison_insights(self, df):
        """Generate AI insights for player comparison"""
        st.write("**🧠 Comparison Insights**")
        # Implementation would go here
        st.info("AI comparison insights coming soon!")

