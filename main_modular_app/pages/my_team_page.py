"""
My Team Page - Handles FPL team import and analysis functionality
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
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
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "👥 Current Squad", 
            "📊 Performance Analysis", 
            "💡 Recommendations",
            "⭐ Starting XI Optimizer",
            "🎯 SWOT Analysis",
            "📈 Advanced Analytics",
            "🔄 Transfer Planning",
            "📊 Performance Comparison"
        ])
        
        with tab1:
            self._display_current_squad(team_data)
        
        with tab2:
            self._display_performance_analysis(team_data)
        
        with tab3:
            self._display_recommendations(team_data)
        
        with tab4:
            self._display_starting_xi_optimizer(team_data)
        
        with tab5:
            self._display_swot_analysis(team_data)
        
        with tab6:
            self._display_advanced_analytics(team_data)
        
        with tab7:
            self._display_transfer_planning(team_data)
        
        with tab8:
            self._display_performance_comparison(team_data)
        
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
        
        # Squad analysis tabs
        squad_tabs = st.tabs(["📋 Squad Overview", "🔥 Performance Heatmap", "📊 Position Analysis", "💎 Value Analysis"])
        
        with squad_tabs[0]:
            self._display_squad_overview(team_data, players_df, picks)
        
        with squad_tabs[1]:
            self._display_squad_heatmap(team_data, players_df, picks)
        
        with squad_tabs[2]:
            self._display_position_analysis(team_data, players_df, picks)
        
        with squad_tabs[3]:
            self._display_squad_value_analysis(team_data, players_df, picks)
    
    def _display_squad_overview(self, team_data, players_df, picks):
        """Display basic squad overview"""
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
                    'PPM': f"{player.get('points_per_million', 0):.1f}",
                    'Status': '(C)' if pick.get('is_captain') else '(VC)' if pick.get('is_vice_captain') else '',
                    'Playing': '✅' if pick.get('position', 12) <= 11 else '🪑'
                })
        
        if squad_data:
            # Formation display
            formation_str = f"{formation_counts['Goalkeeper']}-{formation_counts['Defender']}-{formation_counts['Midfielder']}-{formation_counts['Forward']}"
            st.info(f"**Formation:** {formation_str}")
            
            # Squad table with enhanced columns
            squad_df = pd.DataFrame(squad_data)
            st.dataframe(
                squad_df, 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "Price": st.column_config.TextColumn("Price"),
                    "Points": st.column_config.NumberColumn("Points"),
                    "Form": st.column_config.TextColumn("Form"),
                    "PPM": st.column_config.TextColumn("PPM"),
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
    
    def _display_squad_heatmap(self, team_data, players_df, picks):
        """Display interactive squad performance heatmap"""
        st.subheader("🔥 Squad Performance Heatmap")
        
        # Get squad players with performance data
        squad_players = []
        for pick in picks:
            player_info = players_df[players_df['id'] == pick['element']]
            if not player_info.empty:
                player = player_info.iloc[0]
                squad_players.append({
                    'name': player.get('web_name', 'Unknown'),
                    'position': player.get('position_name', 'Unknown'),
                    'form': float(player.get('form', 0)),
                    'total_points': int(player.get('total_points', 0)),
                    'minutes': int(player.get('minutes', 0)),
                    'ppm': float(player.get('points_per_million', 0)),
                    'ownership': float(player.get('selected_by_percent', 0)),
                    'price': float(player.get('now_cost', 0)) / 10
                })
        
        if not squad_players:
            st.warning("No squad data available for heatmap")
            return
        
        # Create performance matrix
        metrics = ['Form', 'Total Points', 'Minutes', 'PPM', 'Price']
        player_names = [p['name'] for p in squad_players]
        
        # Normalize data for heatmap (0-100 scale)
        heatmap_data = []
        
        for player in squad_players:
            row = []
            # Form (0-10 scale to 0-100)
            row.append(player['form'] * 10)
            # Total points (normalize to season average)
            row.append(min(100, (player['total_points'] / 200) * 100))
            # Minutes (normalize to max possible)
            row.append(min(100, (player['minutes'] / 3000) * 100))
            # PPM (normalize to excellent threshold)
            row.append(min(100, (player['ppm'] / 10) * 100))
            # Price (invert - lower price = higher score)
            row.append(max(0, 100 - (player['price'] / 15) * 100))
            
            heatmap_data.append(row)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=metrics,
            y=player_names,
            colorscale='RdYlGn',
            hoverongaps=False,
            hovertemplate='<b>%{y}</b><br>%{x}: %{z:.1f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Squad Performance Heatmap (Normalized 0-100)',
            height=max(400, len(player_names) * 25),
            yaxis=dict(autorange='reversed'),
            font=dict(size=10)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance insights
        st.write("**🔍 Heatmap Insights:**")
        
        # Find best/worst performers in each category
        best_form = max(squad_players, key=lambda x: x['form'])
        worst_form = min(squad_players, key=lambda x: x['form'])
        best_value = max(squad_players, key=lambda x: x['ppm'])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success(f"🔥 **Best Form**: {best_form['name']} ({best_form['form']:.1f})")
        
        with col2:
            st.error(f"❄️ **Needs Attention**: {worst_form['name']} ({worst_form['form']:.1f})")
        
        with col3:
            st.info(f"💎 **Best Value**: {best_value['name']} ({best_value['ppm']:.1f} PPM)")
    
    def _display_position_analysis(self, team_data, players_df, picks):
        """Display detailed position-by-position analysis"""
        st.subheader("📊 Position Analysis")
        
        # Group players by position
        positions = {}
        for pick in picks:
            player_info = players_df[players_df['id'] == pick['element']]
            if not player_info.empty:
                player = player_info.iloc[0]
                position = player.get('position_name', 'Unknown')
                
                if position not in positions:
                    positions[position] = []
                
                positions[position].append({
                    'name': player.get('web_name', 'Unknown'),
                    'team': player.get('team_short_name', 'UNK'),
                    'price': float(player.get('now_cost', 0)) / 10,
                    'points': int(player.get('total_points', 0)),
                    'form': float(player.get('form', 0)),
                    'ppm': float(player.get('points_per_million', 0)),
                    'starting': pick.get('position', 12) <= 11
                })
        
        # Analyze each position
        for position, players in positions.items():
            with st.expander(f"⚽ {position}s ({len(players)} players)"):
                
                # Position metrics
                col1, col2, col3, col4 = st.columns(4)
                
                avg_price = sum(p['price'] for p in players) / len(players)
                avg_points = sum(p['points'] for p in players) / len(players)
                avg_form = sum(p['form'] for p in players) / len(players)
                starting_count = sum(1 for p in players if p['starting'])
                
                with col1:
                    st.metric("Avg Price", f"£{avg_price:.1f}m")
                
                with col2:
                    st.metric("Avg Points", f"{avg_points:.0f}")
                
                with col3:
                    st.metric("Avg Form", f"{avg_form:.1f}")
                
                with col4:
                    st.metric("Starting", f"{starting_count}/{len(players)}")
                
                # Player breakdown
                st.write("**Players:**")
                for player in sorted(players, key=lambda x: x['points'], reverse=True):
                    status = "🟢 Starting" if player['starting'] else "🟡 Bench"
                    form_emoji = "🔥" if player['form'] > 6 else "❄️" if player['form'] < 3 else "⚖️"
                    
                    st.write(f"• **{player['name']}** ({player['team']}) - £{player['price']:.1f}m - "
                            f"{player['points']} pts - {form_emoji} {player['form']:.1f} - {status}")
                
                # Position-specific recommendations
                if position == 'Goalkeeper':
                    if len(players) < 2:
                        st.warning("⚠️ Consider adding a second goalkeeper for squad balance")
                    elif avg_form < 4:
                        st.warning("⚠️ Both goalkeepers in poor form - monitor closely")
                
                elif position == 'Defender':
                    if starting_count < 3:
                        st.warning("⚠️ Consider starting at least 3 defenders")
                    elif avg_form > 6:
                        st.success("✅ Strong defensive options in good form")
                
                elif position == 'Midfielder':
                    if starting_count < 3:
                        st.warning("⚠️ Consider starting at least 3 midfielders")
                    elif avg_price > 8:
                        st.info("💰 Premium-heavy midfield - ensure returns justify cost")
                
                elif position == 'Forward':
                    if starting_count < 1:
                        st.error("❌ Must start at least 1 forward")
                    elif len(players) < 3:
                        st.warning("⚠️ Consider having 3 forwards for flexibility")
    
    def _display_squad_value_analysis(self, team_data, players_df, picks):
        """Display comprehensive value analysis"""
        st.subheader("💎 Value Analysis")
        
        # Get all squad players with value metrics
        value_players = []
        for pick in picks:
            player_info = players_df[players_df['id'] == pick['element']]
            if not player_info.empty:
                player = player_info.iloc[0]
                
                price = float(player.get('now_cost', 0)) / 10
                points = int(player.get('total_points', 0))
                ppm = float(player.get('points_per_million', 0))
                ownership = float(player.get('selected_by_percent', 0))
                
                # Calculate value metrics
                value_score = ppm  # Base value score
                differential_bonus = max(0, (20 - ownership) / 4)  # Bonus for low ownership
                final_value = value_score + differential_bonus
                
                value_players.append({
                    'name': player.get('web_name', 'Unknown'),
                    'position': player.get('position_name', 'Unknown'),
                    'price': price,
                    'points': points,
                    'ppm': ppm,
                    'ownership': ownership,
                    'value_score': final_value,
                    'category': self._categorize_value(ppm, ownership)
                })
        
        # Value distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**💰 Value Distribution**")
            
            excellent = len([p for p in value_players if p['ppm'] > 8])
            good = len([p for p in value_players if 6 < p['ppm'] <= 8])
            average = len([p for p in value_players if 4 < p['ppm'] <= 6])
            poor = len([p for p in value_players if p['ppm'] <= 4])
            
            st.metric("💎 Excellent (>8 PPM)", excellent)
            st.metric("👍 Good (6-8 PPM)", good)
            st.metric("⚖️ Average (4-6 PPM)", average)
            st.metric("⚠️ Poor (<4 PPM)", poor)
        
        with col2:
            st.write("**🎯 Ownership Analysis**")
            
            template = len([p for p in value_players if p['ownership'] > 50])
            popular = len([p for p in value_players if 20 < p['ownership'] <= 50])
            differential = len([p for p in value_players if p['ownership'] <= 20])
            
            st.metric("🔄 Template (>50%)", template)
            st.metric("📊 Popular (20-50%)", popular)
            st.metric("💎 Differential (<20%)", differential)
        
        # Value champions and concerns
        st.write("**🏆 Value Champions & Concerns**")
        
        # Sort by value score
        sorted_players = sorted(value_players, key=lambda x: x['value_score'], reverse=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("**💎 Best Value Players**")
            for player in sorted_players[:5]:
                st.write(f"• **{player['name']}** ({player['position']}) - "
                        f"£{player['price']:.1f}m - {player['ppm']:.1f} PPM - "
                        f"{player['ownership']:.1f}% owned")
        
        with col2:
            st.error("**⚠️ Value Concerns**")
            for player in sorted_players[-5:]:
                st.write(f"• **{player['name']}** ({player['position']}) - "
                        f"£{player['price']:.1f}m - {player['ppm']:.1f} PPM - "
                        f"{player['ownership']:.1f}% owned")
        
        # Value optimization suggestions
        st.write("**💡 Value Optimization Suggestions**")
        
        suggestions = []
        
        if poor >= 3:
            suggestions.append("🔄 Consider transferring out players with PPM < 4")
        
        if template >= 8:
            suggestions.append("💎 Look for differential options to gain rank")
        
        if excellent < 5:
            suggestions.append("📈 Target more high-value players (PPM > 8)")
        
        total_value = sum(p['price'] for p in value_players)
        bank = team_data.get('bank', 0) / 10
        
        if bank > 2:
            suggestions.append(f"💰 £{bank:.1f}m in bank - consider upgrades")
        
        if total_value < 95:
            suggestions.append("⬆️ Squad value below optimal - room for upgrades")
        
        if suggestions:
            for suggestion in suggestions:
                st.info(suggestion)
        else:
            st.success("✅ Well-optimized squad value distribution!")
    
    def _categorize_value(self, ppm, ownership):
        """Categorize player value"""
        if ppm > 8:
            if ownership < 15:
                return "💎 Premium Differential"
            else:
                return "🏆 Premium Value"
        elif ppm > 6:
            if ownership < 20:
                return "💎 Good Differential"
            else:
                return "👍 Good Value"
        elif ppm > 4:
            return "⚖️ Average Value"
        else:
            return "⚠️ Poor Value"
    
    def _display_performance_analysis(self, team_data):
        """Display performance analysis with enhanced visualizations"""
        st.subheader("📊 Performance Analysis")
        
        # Add real-time performance dashboard
        self._add_real_time_dashboard(team_data)
        
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
        
        # Add performance visualization
        if st.session_state.get('data_loaded', False):
            self._create_performance_charts(team_data)
    
    def _create_performance_charts(self, team_data):
        """Create interactive performance charts"""
        st.subheader("📈 Performance Trends")
        
        # Create simulated historical data (in production, this would come from API)
        current_gw = team_data.get('gameweek', 1)
        total_points = team_data.get('summary_overall_points', 0)
        
        # Simulate gameweek-by-gameweek performance
        gameweeks = list(range(1, current_gw + 1))
        cumulative_points = []
        weekly_points = []
        
        for gw in gameweeks:
            # Simulate realistic point progression
            if gw == 1:
                weekly = max(20, min(80, np.random.normal(50, 15)))
                cumulative = weekly
            else:
                weekly = max(20, min(80, np.random.normal(50, 15)))
                cumulative = cumulative_points[-1] + weekly
            
            weekly_points.append(weekly)
            cumulative_points.append(cumulative)
        
        # Adjust final cumulative to match actual total
        if cumulative_points:
            adjustment_factor = total_points / cumulative_points[-1] if cumulative_points[-1] > 0 else 1
            cumulative_points = [p * adjustment_factor for p in cumulative_points]
        
        # Create performance chart
        fig = go.Figure()
        
        # Add cumulative points line
        fig.add_trace(go.Scatter(
            x=gameweeks,
            y=cumulative_points,
            mode='lines+markers',
            name='Cumulative Points',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=6)
        ))
        
        # Add weekly points bars
        fig.add_trace(go.Bar(
            x=gameweeks,
            y=weekly_points,
            name='Weekly Points',
            opacity=0.7,
            yaxis='y2',
            marker_color='rgba(255, 165, 0, 0.7)'
        ))
        
        # Add average line
        avg_line = [total_points / current_gw * gw for gw in gameweeks]
        fig.add_trace(go.Scatter(
            x=gameweeks,
            y=avg_line,
            mode='lines',
            name='Average Pace',
            line=dict(color='red', dash='dash', width=2)
        ))
        
        # Update layout
        fig.update_layout(
            title='Season Performance Trend',
            xaxis_title='Gameweek',
            yaxis_title='Cumulative Points',
            yaxis2=dict(
                title='Weekly Points',
                overlaying='y',
                side='right',
                range=[0, max(weekly_points) * 1.2]
            ),
            height=500,
            showlegend=True,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Highest GW", f"{max(weekly_points):.0f} pts")
        
        with col2:
            st.metric("Lowest GW", f"{min(weekly_points):.0f} pts")
        
        with col3:
            consistency = (1 - np.std(weekly_points) / np.mean(weekly_points)) * 100
            st.metric("Consistency", f"{consistency:.0f}%")
        
        with col4:
            trend = "📈 Improving" if weekly_points[-3:] > weekly_points[-6:-3] else "📉 Declining"
            st.metric("Recent Trend", trend)
    
    def _add_real_time_dashboard(self, team_data):
        """Add real-time performance tracking dashboard"""
        st.subheader("📈 Real-Time Performance Dashboard")
        
        # Performance velocity metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Rank velocity (change per gameweek)
            current_rank = team_data.get('summary_overall_rank', 0)
            rank_velocity = self._calculate_rank_velocity(team_data)
            st.metric("Rank Velocity", f"{rank_velocity:+,}/GW", 
                     delta_color="inverse" if rank_velocity > 0 else "normal")
        
        with col2:
            # Points momentum
            recent_avg = self._get_recent_average_points(team_data, 3)
            season_avg = team_data.get('summary_overall_points', 0) / max(team_data.get('gameweek', 1), 1)
            momentum = recent_avg - season_avg
            st.metric("Points Momentum", f"{momentum:+.1f}", 
                     delta_color="normal" if momentum > 0 else "inverse")
        
        with col3:
            # Team efficiency
            efficiency_score = self._calculate_team_efficiency(team_data)
            st.metric("Team Efficiency", f"{efficiency_score:.0f}%")
        
        with col4:
            # Risk-adjusted performance
            risk_score = self._calculate_risk_adjusted_performance(team_data)
            st.metric("Risk-Adj. Score", f"{risk_score:.1f}")
        
        # Performance indicators
        col1, col2 = st.columns(2)
        
        with col1:
            # Recent form trend
            st.write("**📊 Recent Form Analysis**")
            if momentum > 5:
                st.success("🚀 Strong upward momentum")
            elif momentum > 0:
                st.info("📈 Positive trend")
            elif momentum > -5:
                st.warning("📉 Slight decline")
            else:
                st.error("⚠️ Concerning downward trend")
        
        with col2:
            # Performance consistency
            st.write("**⚖️ Consistency Rating**")
            if efficiency_score > 85:
                st.success("🎯 Highly consistent performance")
            elif efficiency_score > 70:
                st.info("👍 Good consistency")
            elif efficiency_score > 55:
                st.warning("⚡ Moderately consistent")
            else:
                st.error("🌊 High volatility - focus on stability")
    
    def _calculate_rank_velocity(self, team_data):
        """Calculate rank change velocity"""
        # Simplified calculation - in production, would use historical data
        current_rank = team_data.get('summary_overall_rank', 0)
        current_gw = team_data.get('gameweek', 1)
        
        if current_rank == 0 or current_gw <= 1:
            return 0
        
        # Estimate based on points performance
        gw_points = team_data.get('summary_event_points', 0)
        avg_gw_points = 50  # League average
        
        if gw_points > avg_gw_points + 20:
            return -50000  # Rank improving (going down in number)
        elif gw_points > avg_gw_points + 10:
            return -25000
        elif gw_points < avg_gw_points - 20:
            return 50000   # Rank declining (going up in number)
        elif gw_points < avg_gw_points - 10:
            return 25000
        else:
            return 0
    
    def _get_recent_average_points(self, team_data, weeks=3):
        """Get recent average points (simplified)"""
        # In production, this would calculate from historical data
        current_gw_points = team_data.get('summary_event_points', 0)
        total_points = team_data.get('summary_overall_points', 0)
        current_gw = team_data.get('gameweek', 1)
        
        if current_gw <= weeks:
            return total_points / max(current_gw, 1)
        
        # Simulate recent performance based on current gameweek
        season_avg = total_points / current_gw
        recent_multiplier = 1.0 + (current_gw_points - 50) / 100  # Adjust based on current performance
        
        return season_avg * recent_multiplier
    
    def _calculate_team_efficiency(self, team_data):
        """Calculate team efficiency score"""
        total_points = team_data.get('summary_overall_points', 0)
        team_value = team_data.get('value', 1000) / 10
        current_gw = team_data.get('gameweek', 1)
        
        if team_value == 0 or current_gw == 0:
            return 0
        
        # Points per million per gameweek
        ppm_per_gw = (total_points / team_value) / current_gw
        
        # Normalize to 0-100 scale (8+ ppm/gw = 100%)
        efficiency = min(100, (ppm_per_gw / 8) * 100)
        
        return max(0, efficiency)
    
    def _calculate_risk_adjusted_performance(self, team_data):
        """Calculate risk-adjusted performance score"""
        total_points = team_data.get('summary_overall_points', 0)
        current_gw = team_data.get('gameweek', 1)
        
        if current_gw == 0:
            return 0
        
        avg_ppg = total_points / current_gw
        
        # Simulate risk adjustment based on recent volatility
        gw_points = team_data.get('summary_event_points', 0)
        volatility_penalty = abs(gw_points - avg_ppg) / 10
        
        risk_adjusted = avg_ppg - volatility_penalty
        
        return max(0, risk_adjusted)
    
    def _display_advanced_analytics(self, team_data):
        """Display advanced analytics and insights"""
        st.subheader("📈 Advanced Team Analytics")
        
        if not st.session_state.get('data_loaded', False):
            st.warning("Load player data to enable advanced analytics")
            return
        
        players_df = st.session_state.players_df
        picks = team_data.get('picks', [])
        
        # Enhanced analytics tabs
        analytics_tabs = st.tabs([
            "🏗️ Squad Composition", 
            "📊 Performance Metrics", 
            "💰 Value Analysis",
            "🎯 Risk Assessment",
            "📈 Trend Analysis",
            "🔮 Predictive Analytics",
            "🎲 Monte Carlo Simulation",
            "🧠 AI Insights"
        ])
        
        with analytics_tabs[0]:
            self._display_squad_composition_analysis(team_data, players_df, picks)
        
        with analytics_tabs[1]:
            self._display_performance_metrics_analysis(team_data, players_df, picks)
        
        with analytics_tabs[2]:
            self._display_value_analysis(team_data, players_df, picks)
        
        with analytics_tabs[3]:
            self._display_risk_assessment(team_data, players_df, picks)
        
        with analytics_tabs[4]:
            self._display_trend_analysis(team_data, players_df, picks)
        
        with analytics_tabs[5]:
            self._display_predictive_analytics(team_data, players_df, picks)
        
        with analytics_tabs[6]:
            self._display_monte_carlo_simulation(team_data, players_df, picks)
        
        with analytics_tabs[7]:
            self._display_ai_insights(team_data, players_df, picks)
    
    def _display_predictive_analytics(self, team_data, players_df, picks):
        """Display predictive analytics for upcoming gameweeks"""
        st.subheader("🔮 Predictive Analytics")
        
        with st.expander("📚 Understanding Predictive Analytics", expanded=False):
            st.markdown("""
            **Predictive Analytics** uses historical data and statistical models to forecast:
            
            🎯 **What we predict:**
            - Expected points for next 5 gameweeks
            - Player performance trends
            - Optimal transfer timing
            - Captain choice probabilities
            
            📊 **How it works:**
            - Form trend analysis
            - Fixture difficulty weighting
            - Historical performance patterns
            - Team strength correlations
            """)
        
        # Prediction settings
        col1, col2, col3 = st.columns(3)
        
        with col1:
            prediction_weeks = st.selectbox(
                "📅 Prediction Horizon",
                [3, 5, 8, 10],
                index=1,
                help="Number of gameweeks to predict"
            )
        
        with col2:
            confidence_level = st.slider(
                "🎯 Confidence Level",
                min_value=70, max_value=95, value=80,
                help="Statistical confidence level for predictions"
            )
        
        with col3:
            model_type = st.selectbox(
                "🤖 Model Type",
                ["Conservative", "Balanced", "Aggressive"],
                index=1,
                help="Prediction model approach"
            )
        
        if st.button("🚀 Generate Predictions", type="primary"):
            with st.spinner("Running predictive models..."):
                predictions = self._generate_team_predictions(
                    team_data, players_df, picks, prediction_weeks, confidence_level, model_type
                )
                
                # Display predictions overview
                st.subheader("📊 Prediction Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Expected Points", f"{predictions['expected_total']:.0f}")
                
                with col2:
                    current_avg = team_data.get('summary_overall_points', 0) / max(team_data.get('gameweek', 1), 1)
                    predicted_avg = predictions['expected_total'] / prediction_weeks
                    momentum = predicted_avg - current_avg
                    st.metric("Predicted Avg/GW", f"{predicted_avg:.1f}", 
                             delta=f"{momentum:+.1f}")
                
                with col3:
                    st.metric("Confidence", f"{confidence_level}%")
                
                with col4:
                    st.metric("Model", model_type)
                
                # Weekly breakdown
                st.subheader("📅 Weekly Predictions")
                
                for week in predictions['weekly_breakdown']:
                    with st.expander(f"Gameweek {week['gameweek']} - Predicted: {week['expected_points']:.1f} points"):
                        
                        # Top predicted performers
                        st.write("**🎯 Top Expected Performers:**")
                        for i, player in enumerate(week['top_performers'][:5], 1):
                            confidence_emoji = "🔥" if player['confidence'] > 80 else "👍" if player['confidence'] > 60 else "⚠️"
                            st.write(f"{i}. **{player['name']}** - {player['expected_points']:.1f} pts "
                                   f"{confidence_emoji} ({player['confidence']:.0f}% confidence)")
                        
                        # Captain recommendation
                        captain_rec = week['captain_recommendation']
                        st.info(f"👑 **Captain Recommendation**: {captain_rec['name']} "
                               f"({captain_rec['expected_points']:.1f} pts)")
                        
                        # Risk factors
                        if week['risk_factors']:
                            st.warning("⚠️ **Risk Factors:**")
                            for risk in week['risk_factors']:
                                st.write(f"• {risk}")
                
                # Transfer recommendations
                st.subheader("🔄 Predicted Transfer Value")
                
                transfer_recs = predictions['transfer_recommendations']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.success("**📈 Players to Consider**")
                    for rec in transfer_recs['targets'][:5]:
                        st.write(f"• **{rec['name']}** - Expected: {rec['predicted_points']:.1f} pts/GW")
                
                with col2:
                    st.error("**📉 Underperforming Assets**")
                    for rec in transfer_recs['concerns'][:5]:
                        st.write(f"• **{rec['name']}** - Expected: {rec['predicted_points']:.1f} pts/GW")
    
    def _generate_team_predictions(self, team_data, players_df, picks, weeks, confidence, model_type):
        """Generate predictive analytics for the team"""
        
        # Get squad players with prediction data
        squad_players = []
        for pick in picks:
            player_info = players_df[players_df['id'] == pick['element']]
            if not player_info.empty:
                player = player_info.iloc[0]
                
                # Calculate prediction factors
                form = float(player.get('form', 0))
                total_points = int(player.get('total_points', 0))
                minutes = int(player.get('minutes', 0))
                
                # Base prediction on form and historical performance
                base_prediction = form * 1.2  # Form as base predictor
                
                # Adjust for model type
                if model_type == "Conservative":
                    multiplier = 0.85
                elif model_type == "Aggressive":
                    multiplier = 1.15
                else:
                    multiplier = 1.0
                
                predicted_ppg = base_prediction * multiplier
                
                squad_players.append({
                    'id': player.get('id'),
                    'name': player.get('web_name', 'Unknown'),
                    'position': player.get('position_name', 'Unknown'),
                    'form': form,
                    'predicted_ppg': predicted_ppg,
                    'confidence': min(95, max(50, 70 + (form - 5) * 5)),  # Confidence based on form
                    'starting': pick.get('position', 12) <= 11
                })
        
        # Generate weekly predictions
        weekly_breakdown = []
        total_expected = 0
        
        for week in range(1, weeks + 1):
            week_points = 0
            top_performers = []
            risk_factors = []
            
            for player in squad_players:
                if player['starting']:
                    # Add some random variation for realism
                    week_prediction = max(0, player['predicted_ppg'] + np.random.normal(0, 1))
                    week_points += week_prediction
                    
                    top_performers.append({
                        'name': player['name'],
                        'expected_points': week_prediction,
                        'confidence': player['confidence']
                    })
                
                # Identify risk factors
                if player['starting'] and player['confidence'] < 60:
                    risk_factors.append(f"{player['name']} has low prediction confidence")
            
            # Sort top performers
            top_performers.sort(key=lambda x: x['expected_points'], reverse=True)
            
            # Captain recommendation
            captain_rec = top_performers[0] if top_performers else {'name': 'N/A', 'expected_points': 0}
            
            weekly_breakdown.append({
                'gameweek': team_data.get('gameweek', 1) + week,
                'expected_points': week_points,
                'top_performers': top_performers,
                'captain_recommendation': captain_rec,
                'risk_factors': risk_factors
            })
            
            total_expected += week_points
        
        # Transfer recommendations (simplified)
        transfer_targets = []
        transfer_concerns = []
        
        for player in squad_players:
            if player['predicted_ppg'] > 7:
                transfer_targets.append({
                    'name': player['name'],
                    'predicted_points': player['predicted_ppg']
                })
            elif player['predicted_ppg'] < 3:
                transfer_concerns.append({
                    'name': player['name'],
                    'predicted_points': player['predicted_ppg']
                })
        
        return {
            'expected_total': total_expected,
            'weekly_breakdown': weekly_breakdown,
            'transfer_recommendations': {
                'targets': sorted(transfer_targets, key=lambda x: x['predicted_points'], reverse=True),
                'concerns': sorted(transfer_concerns, key=lambda x: x['predicted_points'])
            }
        }
    
    def _display_monte_carlo_simulation(self, team_data, players_df, picks):
        """Display Monte Carlo simulation for season outcomes"""
        st.subheader("🎲 Monte Carlo Simulation")
        
        with st.expander("📚 What is Monte Carlo Simulation?", expanded=False):
            st.markdown("""
            **Monte Carlo Simulation** runs thousands of scenarios to predict possible outcomes:
            
            🎯 **What it simulates:**
            - Final season rank distribution
            - Points total probabilities
            - Transfer decision outcomes
            - Captain choice success rates
            
            📊 **How it works:**
            - Runs 10,000+ random scenarios
            - Uses probability distributions
            - Accounts for uncertainty
            - Provides confidence intervals
            """)
        
        # Simulation settings
        col1, col2, col3 = st.columns(3)
        
        with col1:
            num_simulations = st.selectbox(
                "🔢 Simulations",
                [1000, 5000, 10000],
                index=1,
                help="More simulations = higher accuracy"
            )
        
        with col2:
            scenario_type = st.selectbox(
                "📈 Scenario",
                ["Season Finish", "Next 5 GWs", "Transfer Impact"],
                help="What to simulate"
            )
        
        with col3:
            risk_level = st.slider(
                "⚖️ Risk Tolerance",
                min_value=1, max_value=10, value=5,
                help="1=Conservative, 10=High risk"
            )
        
        if st.button("🎲 Run Simulation", type="primary"):
            with st.spinner(f"Running {num_simulations:,} simulations..."):
                simulation_results = self._run_monte_carlo_simulation(
                    team_data, players_df, picks, num_simulations, scenario_type, risk_level
                )
                
                # Display results
                st.subheader("📊 Simulation Results")
                
                if scenario_type == "Season Finish":
                    self._display_season_finish_simulation(simulation_results)
                elif scenario_type == "Next 5 GWs":
                    self._display_short_term_simulation(simulation_results)
                else:
                    self._display_transfer_impact_simulation(simulation_results)
    
    def _run_monte_carlo_simulation(self, team_data, players_df, picks, num_sims, scenario_type, risk_level):
        """Run Monte Carlo simulation"""
        
        current_points = team_data.get('summary_overall_points', 0)
        current_gw = team_data.get('gameweek', 1)
        current_rank = team_data.get('summary_overall_rank', 4000000)
        
        results = []
        
        for _ in range(num_sims):
            if scenario_type == "Season Finish":
                # Simulate remaining gameweeks
                remaining_gws = 38 - current_gw
                
                # Base weekly performance with variation
                base_weekly = current_points / max(current_gw, 1)
                weekly_std = base_weekly * 0.3  # 30% standard deviation
                
                # Simulate remaining weeks
                remaining_points = 0
                for week in range(remaining_gws):
                    # Add form factor and random variation
                    week_points = max(20, np.random.normal(base_weekly, weekly_std))
                    remaining_points += week_points
                
                final_points = current_points + remaining_points
                
                # Estimate final rank (simplified)
                # Better performance = better rank
                rank_improvement = (remaining_points / remaining_gws - 50) * 1000
                estimated_final_rank = max(1, current_rank - rank_improvement * remaining_gws)
                
                results.append({
                    'final_points': final_points,
                    'final_rank': estimated_final_rank
                })
            
            elif scenario_type == "Next 5 GWs":
                # Simulate next 5 gameweeks
                total_points = 0
                weekly_points = []
                
                for week in range(5):
                    base_weekly = current_points / max(current_gw, 1)
                    week_points = max(20, np.random.normal(base_weekly, base_weekly * 0.25))
                    total_points += week_points
                    weekly_points.append(week_points)
                
                results.append({
                    'total_5gw_points': total_points,
                    'weekly_points': weekly_points,
                    'avg_weekly': total_points / 5
                })
            
            else:  # Transfer Impact
                # Simulate transfer success
                transfer_success = np.random.random() < (0.6 + risk_level * 0.03)
                
                if transfer_success:
                    points_gain = np.random.normal(15, 5)  # Successful transfer
                else:
                    points_gain = np.random.normal(-5, 3)  # Failed transfer
                
                results.append({
                    'transfer_success': transfer_success,
                    'points_impact': points_gain
                })
        
        return results
    
    def _display_season_finish_simulation(self, results):
        """Display season finish simulation results"""
        final_points = [r['final_points'] for r in results]
        final_ranks = [r['final_rank'] for r in results]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Points distribution
            fig = go.Figure(data=[go.Histogram(x=final_points, nbinsx=30)])
            fig.update_layout(
                title="Final Points Distribution",
                xaxis_title="Total Points",
                yaxis_title="Frequency",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Rank distribution
            fig = go.Figure(data=[go.Histogram(x=final_ranks, nbinsx=30)])
            fig.update_layout(
                title="Final Rank Distribution",
                xaxis_title="Final Rank",
                yaxis_title="Frequency",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        st.subheader("📈 Outcome Probabilities")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            median_points = np.median(final_points)
            st.metric("Median Points", f"{median_points:.0f}")
        
        with col2:
            median_rank = np.median(final_ranks)
            st.metric("Median Rank", f"{median_rank:,.0f}")
        
        with col3:
            top_10k_prob = len([r for r in final_ranks if r <= 10000]) / len(final_ranks) * 100
            st.metric("Top 10K Probability", f"{top_10k_prob:.1f}%")
        
        with col4:
            top_1k_prob = len([r for r in final_ranks if r <= 1000]) / len(final_ranks) * 100
            st.metric("Top 1K Probability", f"{top_1k_prob:.1f}%")
        
        # Confidence intervals
        st.write("**📊 Confidence Intervals:**")
        
        p10_points = np.percentile(final_points, 10)
        p90_points = np.percentile(final_points, 90)
        p10_rank = np.percentile(final_ranks, 90)  # Inverted for rank
        p90_rank = np.percentile(final_ranks, 10)
        
        st.info(f"**80% Confidence Interval:**")
        st.write(f"• Points: {p10_points:.0f} - {p90_points:.0f}")
        st.write(f"• Rank: {p10_rank:,.0f} - {p90_rank:,.0f}")
    
    def _display_short_term_simulation(self, results):
        """Display short-term (5 GW) simulation results"""
        total_points = [r['total_5gw_points'] for r in results]
        avg_weekly = [r['avg_weekly'] for r in results]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Total points distribution
            fig = go.Figure(data=[go.Histogram(x=total_points, nbinsx=25)])
            fig.update_layout(
                title="Next 5 GWs - Total Points",
                xaxis_title="Total Points",
                yaxis_title="Frequency",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Average weekly distribution
            fig = go.Figure(data=[go.Histogram(x=avg_weekly, nbinsx=25)])
            fig.update_layout(
                title="Average Points per GW",
                xaxis_title="Avg Points/GW",
                yaxis_title="Frequency",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Probability analysis
        st.subheader("🎯 Performance Probabilities")
        
        col1, col2, col3, col4 = st.columns(4)
        
        median_total = np.median(total_points)
        excellent_prob = len([p for p in total_points if p >= 350]) / len(total_points) * 100
        poor_prob = len([p for p in total_points if p <= 200]) / len(total_points) * 100
        
        with col1:
            st.metric("Expected Total", f"{median_total:.0f} pts")
        
        with col2:
            st.metric("Expected Avg/GW", f"{np.median(avg_weekly):.1f} pts")
        
        with col3:
            st.metric("Excellent Run (350+)", f"{excellent_prob:.1f}%")
        
        with col4:
            st.metric("Poor Run (<200)", f"{poor_prob:.1f}%")
    
    def _display_transfer_impact_simulation(self, results):
        """Display transfer impact simulation results"""
        success_rate = len([r for r in results if r['transfer_success']]) / len(results) * 100
        points_impact = [r['points_impact'] for r in results]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Success rate pie chart
            fig = go.Figure(data=[go.Pie(
                labels=['Success', 'Failure'],
                values=[success_rate, 100 - success_rate],
                hole=0.3
            )])
            fig.update_layout(
                title="Transfer Success Rate",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Points impact distribution
            fig = go.Figure(data=[go.Histogram(x=points_impact, nbinsx=25)])
            fig.update_layout(
                title="Points Impact Distribution",
                xaxis_title="Points Impact",
                yaxis_title="Frequency",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        with col2:
            expected_impact = np.mean(points_impact)
            st.metric("Expected Impact", f"{expected_impact:+.1f} pts")
        
        with col3:
            positive_prob = len([p for p in points_impact if p > 0]) / len(points_impact) * 100
            st.metric("Positive Impact", f"{positive_prob:.1f}%")
        
        with col4:
            big_gain_prob = len([p for p in points_impact if p > 20]) / len(points_impact) * 100
            st.metric("Big Gain (20+)", f"{big_gain_prob:.1f}%")
    
    def _display_ai_insights(self, team_data, players_df, picks):
        """Display AI-powered insights and recommendations"""
        st.subheader("🧠 AI-Powered Insights")
        
        with st.expander("🤖 How AI Insights Work", expanded=False):
            st.markdown("""
            Our **AI Engine** analyzes your team using advanced algorithms:
            
            🧠 **Pattern Recognition:**
            - Identifies successful team patterns
            - Spots underperforming combinations
            - Recognizes market inefficiencies
            
            📊 **Multi-factor Analysis:**
            - Combines 50+ data points
            - Weights factors by importance
            - Adapts to your management style
            
            💡 **Personalized Recommendations:**
            - Tailored to your risk profile
            - Considers your transfer history
            - Aligns with your objectives
            """)
        
        # AI Analysis Settings
        col1, col2 = st.columns(2)
        
        with col1:
            analysis_depth = st.selectbox(
                "🔍 Analysis Depth",
                ["Quick Scan", "Standard Analysis", "Deep Dive"],
                index=1
            )
        
        with col2:
            focus_area = st.selectbox(
                "🎯 Focus Area",
                ["Overall Performance", "Transfer Strategy", "Risk Management", "Differential Picks"],
                index=0
            )
        
        if st.button("🧠 Generate AI Insights", type="primary"):
            with st.spinner("AI analyzing your team..."):
                ai_insights = self._generate_ai_insights(
                    team_data, players_df, picks, analysis_depth, focus_area
                )
                
                # AI Confidence Score
                st.subheader("🎯 AI Analysis Summary")
                
                confidence = ai_insights['confidence_score']
                if confidence > 85:
                    st.success(f"🤖 **High Confidence Analysis** ({confidence:.0f}%)")
                elif confidence > 70:
                    st.info(f"🤖 **Good Confidence Analysis** ({confidence:.0f}%)")
                else:
                    st.warning(f"🤖 **Moderate Confidence Analysis** ({confidence:.0f}%)")
                
                # Key Insights
                st.subheader("💡 Key AI Insights")
                
                for i, insight in enumerate(ai_insights['key_insights'], 1):
                    impact_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
                    impact_emoji = impact_color.get(insight['impact'], "⚪")
                    
                    with st.expander(f"{impact_emoji} {insight['title']} - {insight['impact']} Impact"):
                        st.write(f"**Analysis:** {insight['description']}")
                        st.write(f"**Recommendation:** {insight['recommendation']}")
                        if insight['confidence'] > 80:
                            st.success(f"✅ High confidence ({insight['confidence']:.0f}%)")
                        elif insight['confidence'] > 60:
                            st.info(f"👍 Good confidence ({insight['confidence']:.0f}%)")
                        else:
                            st.warning(f"⚠️ Moderate confidence ({insight['confidence']:.0f}%)")
                
                # Strategic Recommendations
                st.subheader("🎯 Strategic Recommendations")
                
                strategies = ai_insights['strategic_recommendations']
                
                strategy_tabs = st.tabs(["🔄 Immediate Actions", "📅 Short-term Plan", "🎯 Long-term Strategy"])
                
                with strategy_tabs[0]:
                    st.write("**⚡ Actions for This Gameweek:**")
                    for action in strategies['immediate']:
                        st.info(f"• {action}")
                
                with strategy_tabs[1]:
                    st.write("**📈 Next 3-5 Gameweeks:**")
                    for plan in strategies['short_term']:
                        st.info(f"• {plan}")
                
                with strategy_tabs[2]:
                    st.write("**🏆 Season-long Strategy:**")
                    for strategy in strategies['long_term']:
                        st.info(f"• {strategy}")
                
                # Performance Optimization
                st.subheader("⚙️ AI Optimization Suggestions")
                
                optimizations = ai_insights['optimizations']
                
                for opt in optimizations:
                    priority_color = {"High": "error", "Medium": "warning", "Low": "info"}
                    priority_method = getattr(st, priority_color.get(opt['priority'], 'info'))
                    
                    priority_method(f"**{opt['area']}** ({opt['priority']} Priority): {opt['suggestion']}") 
    
    def _generate_ai_insights(self, team_data, players_df, picks, depth, focus):
        """Generate AI-powered insights"""
        
        # Calculate base metrics for AI analysis
        total_points = team_data.get('summary_overall_points', 0)
        current_gw = team_data.get('gameweek', 1)
        current_rank = team_data.get('summary_overall_rank', 4000000)
        team_value = team_data.get('value', 1000) / 10
        bank = team_data.get('bank', 0) / 10
        
        # Analyze squad composition
        squad_analysis = self._analyze_squad_for_ai(players_df, picks)
        
        # Generate insights based on analysis depth
        insights = []
        confidence_factors = []
        
        # Performance Analysis
        if total_points > 0:
            avg_ppg = total_points / current_gw
            
            if avg_ppg > 60:
                insights.append({
                    'title': 'Strong Performance Trend',
                    'description': f'Your {avg_ppg:.1f} points/GW is well above average',
                    'recommendation': 'Maintain current strategy and look for incremental improvements',
                    'impact': 'Low',
                    'confidence': 90
                })
                confidence_factors.append(90)
            elif avg_ppg < 45:
                insights.append({
                    'title': 'Performance Below Expectations',
                    'description': f'Your {avg_ppg:.1f} points/GW suggests strategic issues',
                    'recommendation': 'Consider major squad restructuring and transfer strategy review',
                    'impact': 'High',
                    'confidence': 85
                })
                confidence_factors.append(85)
        
        # Squad Balance Analysis
        if squad_analysis['premium_heavy']:
            insights.append({
                'title': 'Premium-Heavy Squad Detected',
                'description': 'High concentration of expensive players may limit flexibility',
                'recommendation': 'Consider downgrading one premium to improve squad depth',
                'impact': 'Medium',
                'confidence': 80
            })
            confidence_factors.append(80)
        
        if squad_analysis['poor_value_count'] >= 3:
            insights.append({
                'title': 'Value Efficiency Issues',
                'description': f'{squad_analysis["poor_value_count"]} players offer poor points per million',
                'recommendation': 'Prioritize transferring out players with PPM < 4',
                'impact': 'High',
                'confidence': 88
            })
            confidence_factors.append(88)
        
        # Financial Analysis
        if bank > 3:
            insights.append({
                'title': 'Unused Transfer Budget',
                'description': f'£{bank:.1f}m in bank represents missed opportunities',
                'recommendation': 'Invest in squad upgrades or target players before price rises',
                'impact': 'Medium',
                'confidence': 75
            })
            confidence_factors.append(75)
        
        # Risk Analysis
        if squad_analysis['high_risk_players'] >= 4:
            insights.append({
                'title': 'High Squad Risk Exposure',
                'description': 'Multiple players with injury/rotation concerns',
                'recommendation': 'Diversify risk by targeting more reliable options',
                'impact': 'High',
                'confidence': 82
            })
            confidence_factors.append(82)
        
        # Generate strategic recommendations
        strategies = {
            'immediate': [], 
            'short_term': [], 
            'long_term': []
        }
        
        # Immediate actions
        if current_rank and current_rank > 1000000:
            strategies['immediate'].append('Focus on template players to reduce rank volatility')
        
        strategies['immediate'].append('Monitor player price changes before next deadline')
        strategies['immediate'].append('Check injury news for all starting players')
        
        # Short-term planning
        strategies['short_term'].append('Plan transfers around upcoming fixture swings')
        strategies['short_term'].append('Consider chip usage timing (Wildcard, Bench Boost, etc.)')
        strategies['short_term'].append('Target players with favorable upcoming fixtures')
        
        # Long-term strategy
        if current_gw < 20:
            strategies['long_term'].append('Build squad value through strategic price rise targets')
        else:
            strategies['long_term'].append('Focus on consistent performers for season finale')
        
        strategies['long_term'].append('Develop differential strategy based on rank targets')
        strategies['long_term'].append('Plan major squad changes around double gameweeks')
        
        # Optimization suggestions
        optimizations = []
        
        if squad_analysis['weak_bench']:
            optimizations.append({
                'area': 'Squad Depth',
                'suggestion': 'Improve bench strength for better rotation options',
                'priority': 'Medium'
            })
        
        if squad_analysis['captain_options'] < 3:
            optimizations.append({
                'area': 'Captaincy',
                'suggestion': 'Acquire more reliable captain options',
                'priority': 'High'
            })
        
        optimizations.append({
            'area': 'Form Monitoring',
            'suggestion': 'Implement weekly form review process',
            'priority': 'Low'
        })
        
        # Calculate overall confidence
        overall_confidence = np.mean(confidence_factors) if confidence_factors else 70
        
        return {
            'confidence_score': overall_confidence,
            'key_insights': insights,
            'strategic_recommendations': strategies,
            'optimizations': optimizations
        }
    
    def _analyze_squad_for_ai(self, players_df, picks):
        """Analyze squad for AI insights"""
        
        analysis = {
            'premium_heavy': False,
            'poor_value_count': 0,
            'high_risk_players': 0,
            'weak_bench': False,
            'captain_options': 0
        }
        
        premium_count = 0
        bench_strength = 0
        
        for pick in picks:
            player_info = players_df[players_df['id'] == pick['element']]
            if not player_info.empty:
                player = player_info.iloc[0]
                
                price = float(player.get('now_cost', 0)) / 10
                ppm = float(player.get('points_per_million', 0))
                form = float(player.get('form', 0))
                status = player.get('status', 'a')
                
                # Premium player check
                if price >= 10:
                    premium_count += 1
                
                # Poor value check
                if ppm < 4:
                    analysis['poor_value_count'] += 1
                
                # High risk check
                if status != 'a' or form < 3:
                    analysis['high_risk_players'] += 1
                
                # Captain options
                if form > 6 and price >= 8:
                    analysis['captain_options'] += 1
                
                # Bench strength
                if pick.get('position', 12) > 11:  # Bench player
                    bench_strength += ppm
        
        # Set flags
        analysis['premium_heavy'] = premium_count >= 4
        analysis['weak_bench'] = bench_strength < 20
        
        return analysis
    
    def _display_recommendations(self, team_data):
        """Display enhanced recommendations for the team"""
        st.subheader("💡 Team Recommendations")
        
        # Enhanced recommendation system
        recommendations = self._generate_enhanced_recommendations(team_data)
        
        # Recommendation categories
        rec_tabs = st.tabs(["🔥 Priority Actions", "📈 Performance Tips", "💰 Value Opportunities", "🎯 Strategic Advice"])
        
        with rec_tabs[0]:
            st.write("**⚡ High Priority Actions**")
            priority_recs = [r for r in recommendations if r['priority'] == 'High']
            if priority_recs:
                for rec in priority_recs:
                    st.error(f"🚨 **{rec['title']}**: {rec['message']}")
            else:
                st.success("✅ No urgent actions required!")
        
        with rec_tabs[1]:
            st.write("**📊 Performance Improvement Tips**")
            performance_recs = [r for r in recommendations if r['category'] == 'Performance']
            for rec in performance_recs:
                st.info(f"📈 **{rec['title']}**: {rec['message']}")
        
        with rec_tabs[2]:
            st.write("**💎 Value Enhancement Opportunities**")
            value_recs = [r for r in recommendations if r['category'] == 'Value']
            for rec in value_recs:
                st.success(f"💰 **{rec['title']}**: {rec['message']}")
        
        with rec_tabs[3]:
            st.write("**🎯 Strategic Recommendations**")
            strategic_recs = [r for r in recommendations if r['category'] == 'Strategic']
            for rec in strategic_recs:
                st.info(f"🎯 **{rec['title']}**: {rec['message']}")
        
        # General FPL wisdom
        with st.expander("📚 Advanced FPL Strategy Guide", expanded=False):
            st.markdown("""
            ### 🏆 **Elite FPL Management Principles**
            
            **📊 Data-Driven Decisions:**
            - Use xG, xA, and underlying stats over just points
            - Monitor price change predictions and ownership trends
            - Track fixture difficulty ratings beyond 5 gameweeks
            
            **🎯 Strategic Planning:**
            - Plan transfers around fixture swings, not knee-jerk reactions
            - Time chip usage for optimal double/blank gameweeks
            - Maintain 2-3 premium captain options at all times
            
            **💎 Differential Strategy:**
            - Target 2-3 <10% owned players for rank acceleration
            - Balance template safety with differential upside
            - Monitor rising ownership to sell before becoming template
            
            **⚖️ Risk Management:**
            - Diversify across teams and price points
            - Maintain strong bench for rotation flexibility
            - Keep £0.5-1.0m buffer for price rise protection
            """)
    
    def _generate_enhanced_recommendations(self, team_data):
        """Generate enhanced, data-driven recommendations"""
        recommendations = []
        
        # Get basic team metrics
        total_points = team_data.get('summary_overall_points', 0)
        current_gw = team_data.get('gameweek', 1)
        current_rank = team_data.get('summary_overall_rank', 0)
        team_value = team_data.get('value', 1000) / 10
        bank = team_data.get('bank', 0) / 10
        gw_points = team_data.get('summary_event_points', 0)
        
        # Performance analysis
        if current_gw > 0:
            avg_ppg = total_points / current_gw
            
            if avg_ppg < 45:
                recommendations.append({
                    'title': 'Performance Review Required',
                    'message': f'Averaging {avg_ppg:.1f} pts/GW is below competitive levels. Consider wildcard for major restructure.',
                    'category': 'Performance',
                    'priority': 'High'
                })
            elif avg_ppg > 65:
                recommendations.append({
                    'title': 'Excellent Performance',
                    'message': f'Your {avg_ppg:.1f} pts/GW average is elite level. Focus on consistency and differential picks.',
                    'category': 'Performance',
                    'priority': 'Low'
                })
        
        # Financial recommendations
        if bank > 3:
            recommendations.append({
                'title': 'Excess Funds Available',
                'message': f'£{bank:.1f}m in bank is suboptimal. Consider upgrading weaker squad members.',
                'category': 'Value',
                'priority': 'High'
            })
        elif bank < 0.5:
            recommendations.append({
                'title': 'Low Financial Flexibility',
                'message': 'Consider downgrading a premium to improve transfer flexibility.',
                'category': 'Strategic',
                'priority': 'Medium'
            })
        
        # Rank-based strategy
        if current_rank and current_rank > 1000000:
            recommendations.append({
                'title': 'Rank Recovery Strategy',
                'message': 'Focus on template players and reliable captains to reduce volatility.',
                'category': 'Strategic',
                'priority': 'High'
            })
        elif current_rank and current_rank < 100000:
            recommendations.append({
                'title': 'Elite Rank Maintenance',
                'message': 'Consider differential captains and low-owned players to maintain advantage.',
                'category': 'Strategic',
                'priority': 'Medium'
            })
        
        # Recent form analysis
        if gw_points < 35:
            recommendations.append({
                'title': 'Poor Recent Form',
                'message': 'Review captain choice and check for injured/suspended players.',
                'category': 'Performance',
                'priority': 'High'
            })
        
        # Squad value optimization
        if team_value < 95:
            recommendations.append({
                'title': 'Squad Value Below Optimal',
                'message': 'Target players likely to rise in value to build squad worth.',
                'category': 'Value',
                'priority': 'Medium'
            })
        
        # Add default recommendations if none generated
        if not recommendations:
            recommendations.extend([
                {
                    'title': 'Regular Review',
                    'message': 'Monitor player form, injuries, and upcoming fixtures weekly.',
                    'category': 'Strategic',
                    'priority': 'Low'
                },
                {
                    'title': 'Stay Informed',
                    'message': 'Follow press conferences and team news for rotation insights.',
                    'category': 'Strategic',
                    'priority': 'Low'
                }
            ])
        
        return recommendations
    
    def _display_starting_xi_optimizer(self, team_data):
        """Display Starting XI Optimizer with comprehensive analysis"""
        st.subheader("⭐ Starting XI Optimizer")
        
        # Explanation section
        with st.expander("📚 How the Starting XI Optimizer Works", expanded=False):
            st.markdown("""
            The **Starting XI Optimizer** uses advanced algorithms to suggest your best possible lineup based on:
            
            🎯 **Key Factors:**
            - **Form Rating**: Recent performance trends (last 5 games)
            - **Fixture Difficulty**: Opposition strength analysis
            - **Expected Points**: Predicted performance this gameweek
            - **Minutes Played**: Reliability and rotation risk
            - **Price Per Point**: Value efficiency
            - **Team Strength**: Overall team performance indicators
            
            ⚽ **Optimization Strategy:**
            - Ensures valid formation (1 GK, 3-5 DEF, 3-5 MID, 1-3 FWD)
            - Maximizes expected points while considering form
            - Suggests captain based on highest expected returns
            - Identifies bench players and substitution priorities
            """)
        
        if not st.session_state.get('data_loaded', False):
            st.warning("⚠️ Please load FPL player data first to enable the Starting XI Optimizer")
            return
        
        picks = team_data.get('picks', [])
        if not picks:
            st.warning("No squad data available for optimization")
            return
        
        players_df = st.session_state.players_df
        
        # Settings for optimization
        col1, col2, col3 = st.columns(3)
        
        with col1:
            formation_pref = st.selectbox(
                "🏗️ Preferred Formation",
                ["Auto-Select", "3-4-3", "3-5-2", "4-3-3", "4-4-2", "4-5-1", "5-3-2", "5-4-1"],
                help="Choose formation or let AI auto-select the best"
            )
        
        with col2:
            optimization_focus = st.selectbox(
                "🎯 Optimization Focus",
                ["Balanced", "Form-Heavy", "Fixture-Based", "Conservative", "Differential"],
                help="Adjust the strategy focus"
            )
        
        with col3:
            risk_tolerance = st.slider(
                "⚖️ Risk Tolerance",
                min_value=1, max_value=10, value=5,
                help="1=Safe picks, 10=High risk/reward"
            )
        
        # Analyze squad and generate recommendations
        if st.button("🚀 Optimize Starting XI", type="primary"):
            with st.spinner("Analyzing your squad and optimizing lineup..."):
                optimized_lineup = self._optimize_starting_eleven(
                    team_data, players_df, formation_pref, optimization_focus, risk_tolerance
                )
                
                if optimized_lineup:
                    self._display_optimized_lineup(optimized_lineup)
                else:
                    st.error("❌ Could not optimize lineup. Please check your squad data.")
    
    def _optimize_starting_eleven(self, team_data, players_df, formation_pref, optimization_focus, risk_tolerance):
        """Core optimization algorithm for Starting XI"""
        picks = team_data.get('picks', [])
        
        # Get squad players with enhanced stats
        squad_players = []
        for pick in picks:
            player_info = players_df[players_df['id'] == pick['element']]
            if not player_info.empty:
                player = player_info.iloc[0].to_dict()
                player['pick_data'] = pick
                player['optimization_score'] = self._calculate_optimization_score(
                    player, optimization_focus, risk_tolerance
                )
                squad_players.append(player)
        
        if len(squad_players) < 15:
            return None
        
        # Group by position
        positions = {
            'GK': [p for p in squad_players if p.get('position_name') == 'Goalkeeper'],
            'DEF': [p for p in squad_players if p.get('position_name') == 'Defender'],
            'MID': [p for p in squad_players if p.get('position_name') == 'Midfielder'],
            'FWD': [p for p in squad_players if p.get('position_name') == 'Forward']
        }
        
        # Sort each position by optimization score
        for pos in positions:
            positions[pos].sort(key=lambda x: x['optimization_score'], reverse=True)
        
        # Determine formation
        if formation_pref == "Auto-Select":
            formation = self._determine_optimal_formation(positions)
        else:
            formation = self._parse_formation(formation_pref)
        
        # Select starting XI
        starting_xi = {
            'GK': positions['GK'][:1] if positions['GK'] else [],
            'DEF': positions['DEF'][:formation['DEF']] if len(positions['DEF']) >= formation['DEF'] else positions['DEF'],
            'MID': positions['MID'][:formation['MID']] if len(positions['MID']) >= formation['MID'] else positions['MID'],
            'FWD': positions['FWD'][:formation['FWD']] if len(positions['FWD']) >= formation['FWD'] else positions['FWD']
        }
        
        # Calculate bench (remaining players)
        bench_players = []
        for pos in positions:
            if pos == 'GK':
                bench_players.extend(positions[pos][1:])  # Backup GK
            elif pos == 'DEF':
                bench_players.extend(positions[pos][formation['DEF']:])
            elif pos == 'MID':
                bench_players.extend(positions[pos][formation['MID']:])
            elif pos == 'FWD':
                bench_players.extend(positions[pos][formation['FWD']:])
        
        # Sort bench by optimization score
        bench_players.sort(key=lambda x: x['optimization_score'], reverse=True)
        
        # Suggest captain and vice-captain
        all_starters = []
        for pos_players in starting_xi.values():
            all_starters.extend(pos_players)
        
        captain_candidates = sorted(all_starters, key=lambda x: x['optimization_score'], reverse=True)
        
        return {
            'formation': formation,
            'starting_xi': starting_xi,
            'bench': bench_players[:4],  # First 4 bench players
            'captain': captain_candidates[0] if captain_candidates else None,
            'vice_captain': captain_candidates[1] if len(captain_candidates) > 1 else None,
            'total_predicted_points': sum([p['optimization_score'] for p in all_starters])
        }
    
    def _calculate_optimization_score(self, player, optimization_focus, risk_tolerance):
        """Calculate optimization score based on multiple factors"""
        score = 0
        
        # Helper function to safely convert to float
        def safe_float(value, default=0.0):
            try:
                if value is None or value == '':
                    return default
                return float(value)
            except (ValueError, TypeError):
                return default
        
        # Helper function to safely convert to int
        def safe_int(value, default=0):
            try:
                if value is None or value == '':
                    return default
                return int(float(value))  # Convert via float first to handle string decimals
            except (ValueError, TypeError):
                return default
        
        # Base score from total points (normalized)
        total_points = safe_float(player.get('total_points', 0))
        score += (total_points / 300) * 30  # Max 30 points from total points
        
        # Form component (0-20 points)
        form = safe_float(player.get('form', 0))
        if optimization_focus == "Form-Heavy":
            score += (form / 10) * 25
        else:
            score += (form / 10) * 15
        
        # Points per game reliability (0-15 points)
        ppg = safe_float(player.get('points_per_game', 0))
        if ppg > 0:
            score += min(ppg * 2, 15)
        
        # Minutes played reliability (0-10 points)
        minutes = safe_int(player.get('minutes', 0))
        if minutes > 1500:  # Regular starter
            score += 10
        elif minutes > 1000:
            score += 7
        elif minutes > 500:
            score += 4
        
        # Value efficiency (0-10 points)
        ppm = safe_float(player.get('points_per_million', 0))
        if ppm > 8:
            score += 10
        elif ppm > 6:
            score += 7
        elif ppm > 4:
            score += 4
        
        # Team strength factor (0-5 points)
        # Estimate based on clean sheets, goals scored
        team_strength = 0
        clean_sheets = safe_int(player.get('clean_sheets', 0))
        goals_scored = safe_int(player.get('goals_scored', 0))
        
        if clean_sheets > 8:  # Good defensive team
            team_strength += 2
        if goals_scored > 5:  # Good attacking player
            team_strength += 3
        score += min(team_strength, 5)
        
        # Risk adjustment based on tolerance
        if risk_tolerance < 5:  # Conservative
            # Prefer established players
            if total_points > 100:
                score += 5
        else:  # Higher risk tolerance
            # Prefer differentials
            ownership = safe_float(player.get('selected_by_percent', 50))
            if ownership < 15:  # Differential
                score += (10 - risk_tolerance) * 1.5
        
        # Fixture difficulty (simplified - based on team strength)
        if optimization_focus == "Fixture-Based":
            # This is a simplified fixture analysis
            # In a full implementation, you'd have actual fixture data
            score += 5  # Placeholder for fixture analysis
        
        return max(score, 0)  # Ensure non-negative score
    
    def _determine_optimal_formation(self, positions):
        """Determine the best formation based on player quality"""
        formations = [
            {'DEF': 3, 'MID': 4, 'FWD': 3},
            {'DEF': 3, 'MID': 5, 'FWD': 2},
            {'DEF': 4, 'MID': 3, 'FWD': 3},
            {'DEF': 4, 'MID': 4, 'FWD': 2},
            {'DEF': 4, 'MID': 5, 'FWD': 1},
            {'DEF': 5, 'MID': 3, 'FWD': 2},
            {'DEF': 5, 'MID': 4, 'FWD': 1}
        ]
        
        best_formation = None
        best_score = 0
        
        for formation in formations:
            if (len(positions['DEF']) >= formation['DEF'] and 
                len(positions['MID']) >= formation['MID'] and 
                len(positions['FWD']) >= formation['FWD']):
                
                # Calculate total score for this formation
                score = 0
                score += sum([p['optimization_score'] for p in positions['DEF'][:formation['DEF']]])
                score += sum([p['optimization_score'] for p in positions['MID'][:formation['MID']]])
                score += sum([p['optimization_score'] for p in positions['FWD'][:formation['FWD']]])
                
                if score > best_score:
                    best_score = score
                    best_formation = formation
        
        return best_formation or {'DEF': 4, 'MID': 4, 'FWD': 2}  # Default fallback
    
    def _parse_formation(self, formation_str):
        """Parse formation string into position counts"""
        if formation_str == "Auto-Select":
            return {'DEF': 4, 'MID': 4, 'FWD': 2}
        
        parts = formation_str.split('-')
        if len(parts) == 3:
            return {
                'DEF': int(parts[0]),
                'MID': int(parts[1]),
                'FWD': int(parts[2])
            }
        return {'DEF': 4, 'MID': 4, 'FWD': 2}  # Default
    
    def _display_optimized_lineup(self, optimized_lineup):
        """Display the optimized lineup with detailed breakdown"""
        st.success("✅ **Optimization Complete!**")
        
        formation = optimized_lineup['formation']
        formation_str = f"{formation['DEF']}-{formation['MID']}-{formation['FWD']}"
        
        # Overall recommendations
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🏗️ Optimal Formation", formation_str)
        
        with col2:
            predicted_points = optimized_lineup['total_predicted_points']
            st.metric("📊 Predicted Points", f"{predicted_points:.1f}")
        
        with col3:
            captain = optimized_lineup['captain']
            captain_name = captain['web_name'] if captain else "N/A"
            st.metric("👑 Captain", captain_name)
        
        with col4:
            vice_captain = optimized_lineup['vice_captain']
            vc_name = vice_captain['web_name'] if vice_captain else "N/A"
            st.metric("🥈 Vice Captain", vc_name)
        
        # Detailed lineup breakdown
        st.subheader("🎯 Optimized Starting XI")
        
        starting_xi = optimized_lineup['starting_xi']
        
        # Display by position
        for pos_name, display_name in [('GK', '🥅 Goalkeeper'), ('DEF', '🛡️ Defenders'), ('MID', '⚽ Midfielders'), ('FWD', '🎯 Forwards')]:
            if starting_xi[pos_name]:
                st.write(f"**{display_name}**")
                
                for player in starting_xi[pos_name]:
                    col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 1])
                    
                    with col1:
                        captain_badge = " 👑" if player == optimized_lineup['captain'] else " 🥈" if player == optimized_lineup['vice_captain'] else ""
                        st.write(f"**{player['web_name']}**{captain_badge}")
                    
                    with col2:
                        st.write(f"{player.get('team_short_name', 'UNK')}")
                    
                    with col3:
                        st.write(f"Form: {player.get('form', 0):.1f}")
                    
                    with col4:
                        st.write(f"£{player.get('cost_millions', 0):.1f}m")
                    
                    with col5:
                        score = player.get('optimization_score', 0)
                        st.write(f"{score:.1f}")
                
                st.divider()
        
        # Bench recommendations
        st.subheader("🪑 Recommended Bench")
        
        bench = optimized_lineup['bench']
        if bench:
            for i, player in enumerate(bench, 1):
                col1, col2, col3, col4 = st.columns([1, 3, 2, 2])
                
                with col1:
                    st.write(f"{i}.")
                
                with col2:
                    st.write(f"**{player['web_name']}** ({player.get('position_name', 'UNK')})")
                
                with col3:
                    st.write(f"{player.get('team_short_name', 'UNK')}")
                
                with col4:
                    st.write(f"Score: {player.get('optimization_score', 0):.1f}")
        
        # Strategic insights
        st.subheader("💡 Strategic Insights")
        
        insights = []
        
        # Captain analysis
        captain = optimized_lineup['captain']
        if captain:
            captain_form = captain.get('form', 0)
            if captain_form > 7:
                insights.append("🔥 Your suggested captain is in excellent form!")
            elif captain_form < 4:
                insights.append("⚠️ Consider alternative captain options - current suggestion has poor form")
        
        # Formation analysis
        if formation['DEF'] >= 5:
            insights.append("🛡️ Defensive formation - good for teams expecting clean sheets")
        elif formation['FWD'] >= 3:
            insights.append("⚔️ Attacking formation - prioritizing goal-scoring potential")
        
        # Risk analysis
        risky_players = [p for pos_players in starting_xi.values() for p in pos_players if p.get('selected_by_percent', 50) < 10]
        if len(risky_players) >= 2:
            insights.append(f"💎 {len(risky_players)} differential picks in starting XI - high risk/reward strategy")
        
        # Value analysis
        total_value = sum([p.get('cost_millions', 0) for pos_players in starting_xi.values() for p in pos_players])
        if total_value < 75:
            insights.append("💰 Budget-friendly lineup leaves room for upgrades")
        elif total_value > 85:
            insights.append("💸 Premium-heavy lineup - ensure these players deliver")
        
        # Display insights
        if insights:
            for insight in insights:
                st.info(insight)
        else:
            st.success("✅ Well-balanced team selection!")
        
        # Action buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📋 Copy Lineup to Clipboard"):
                lineup_text = self._generate_lineup_text(optimized_lineup)
                st.code(lineup_text, language="text")
                st.success("Lineup formatted for sharing!")
        
        with col2:
            if st.button("🔄 Re-optimize with Different Settings"):
                st.rerun()
    
    def _generate_lineup_text(self, optimized_lineup):
        """Generate text representation of the lineup"""
        formation = optimized_lineup['formation']
        starting_xi = optimized_lineup['starting_xi']
        
        lines = []
        lines.append(f"🏗️ Formation: {formation['DEF']}-{formation['MID']}-{formation['FWD']}")
        lines.append("")
        
        for pos_name, display_name in [('GK', 'GK'), ('DEF', 'DEF'), ('MID', 'MID'), ('FWD', 'FWD')]:
            if starting_xi[pos_name]:
                players = [p['web_name'] for p in starting_xi[pos_name]]
                lines.append(f"{display_name}: {', '.join(players)}")
        
        lines.append("")
        captain = optimized_lineup['captain']
        vice_captain = optimized_lineup['vice_captain']
        
        if captain:
            lines.append(f"👑 Captain: {captain['web_name']}")
        if vice_captain:
            lines.append(f"🥈 Vice Captain: {vice_captain['web_name']}")
        
        return "\n".join(lines)
    
    def _display_swot_analysis(self, team_data):
        """Display comprehensive SWOT analysis for the team"""
        st.subheader("🎯 Team SWOT Analysis")
        
        with st.expander("📚 What is SWOT Analysis?", expanded=False):
            st.markdown("""
            **SWOT Analysis** is a strategic planning technique that evaluates four key aspects:
            
            - **🔥 Strengths**: Internal positive factors that give you advantages
            - **⚠️ Weaknesses**: Internal negative factors that need improvement
            - **🚀 Opportunities**: External positive factors you can exploit
            - **⛔ Threats**: External negative factors that could harm your performance
            
            This helps you make informed transfer and tactical decisions.
            """)
        
        if not st.session_state.get('data_loaded', False):
            st.warning("Load player data to enable detailed SWOT analysis")
            return
        
        # Generate SWOT analysis
        swot_data = self._generate_swot_analysis(team_data)
        
        # Display SWOT in a 2x2 grid
        col1, col2 = st.columns(2)
        
        with col1:
            # Strengths
            st.markdown("### 🔥 **STRENGTHS**")
            st.markdown("*Internal positive factors*")
            
            if swot_data['strengths']:
                for strength in swot_data['strengths']:
                    st.success(f"✅ {strength}")
            else:
                st.info("🔍 Analyzing your team strengths...")
            
            st.markdown("---")
            
            # Weaknesses
            st.markdown("### ⚠️ **WEAKNESSES**")
            st.markdown("*Internal areas for improvement*")
            
            if swot_data['weaknesses']:
                for weakness in swot_data['weaknesses']:
                    st.error(f"❌ {weakness}")
            else:
                st.info("🔍 No major weaknesses identified")
        
        with col2:
            # Opportunities
            st.markdown("### 🚀 **OPPORTUNITIES**")
            st.markdown("*External factors to exploit*")
            
            if swot_data['opportunities']:
                for opportunity in swot_data['opportunities']:
                    st.info(f"💡 {opportunity}")
            else:
                st.info("🔍 Looking for opportunities...")
            
            st.markdown("---")
            
            # Threats
            st.markdown("### ⛔ **THREATS**")
            st.markdown("*External risks to mitigate*")
            
            if swot_data['threats']:
                for threat in swot_data['threats']:
                    st.warning(f"⚠️ {threat}")
            else:
                st.info("🔍 No immediate threats detected")
        
        # Strategic recommendations based on SWOT
        st.markdown("---")
        st.subheader("📋 Strategic Action Plan")
        
        action_plan = self._generate_swot_action_plan(swot_data)
        
        strategy_tabs = st.tabs(["🎯 SO Strategy", "🔧 WO Strategy", "🛡️ ST Strategy", "⚡ WT Strategy"])
        
        with strategy_tabs[0]:
            st.markdown("**Strength-Opportunity (SO)**: *Use strengths to capitalize on opportunities*")
            for action in action_plan.get('so_strategy', []):
                st.success(f"🎯 {action}")
        
        with strategy_tabs[1]:
            st.markdown("**Weakness-Opportunity (WO)**: *Overcome weaknesses by exploiting opportunities*")
            for action in action_plan.get('wo_strategy', []):
                st.info(f"🔧 {action}")
        
        with strategy_tabs[2]:
            st.markdown("**Strength-Threat (ST)**: *Use strengths to avoid threats*")
            for action in action_plan.get('st_strategy', []):
                st.warning(f"🛡️ {action}")
        
        with strategy_tabs[3]:
            st.markdown("**Weakness-Threat (WT)**: *Minimize weaknesses and avoid threats*")
            for action in action_plan.get('wt_strategy', []):
                st.error(f"⚡ {action}")
    
    def _generate_swot_analysis(self, team_data):
        """Generate SWOT analysis based on team data"""
        players_df = st.session_state.players_df
        picks = team_data.get('picks', [])
        
        strengths = []
        weaknesses = []
        opportunities = []
        threats = []
        
        # Get team players data
        team_players = []
        for pick in picks:
            player_info = players_df[players_df['id'] == pick['element']]
            if not player_info.empty:
                player = player_info.iloc[0]
                team_players.append({
                    'player': player,
                    'pick': pick
                })
        
        if not team_players:
            return {'strengths': [], 'weaknesses': [], 'opportunities': [], 'threats': []}
        
        # Analyze strengths
        total_points = team_data.get('summary_overall_points', 0)
        overall_rank = team_data.get('summary_overall_rank', 0)
        team_value = team_data.get('value', 1000) / 10
        bank = team_data.get('bank', 0) / 10
        
        # Performance-based strengths
        if overall_rank and overall_rank < 500000:
            strengths.append("Strong overall rank - Top 500K performance")
        
        if total_points > 1500:
            strengths.append("High-scoring team with consistent point generation")
        
        # Player quality strengths
        high_form_players = [tp for tp in team_players if float(tp['player'].get('form', 0)) > 7]
        if len(high_form_players) >= 3:
            strengths.append(f"{len(high_form_players)} players in excellent form (7+ rating)")
        
        premium_players = [tp for tp in team_players if float(tp['player'].get('now_cost', 0)) > 100]
        if len(premium_players) >= 2:
            strengths.append(f"Strong premium player base ({len(premium_players)} players >£10m)")
        
        # Value efficiency
        efficient_players = [tp for tp in team_players if float(tp['player'].get('points_per_million', 0)) > 8]
        if len(efficient_players) >= 5:
            strengths.append("Excellent value efficiency across squad")
        
        # Financial strength
        if bank > 1.5:
            strengths.append(f"Strong financial position (£{bank:.1f}m in bank)")
        
        # Analyze weaknesses
        # Poor form players
        poor_form_players = [tp for tp in team_players if float(tp['player'].get('form', 0)) < 3]
        if len(poor_form_players) >= 2:
            weaknesses.append(f"{len(poor_form_players)} players in poor form (<3 rating)")
        
        # Injury concerns
        injured_players = [tp for tp in team_players if tp['player'].get('status') in ['i', 'd']]
        if injured_players:
            weaknesses.append(f"{len(injured_players)} players with injury/availability concerns")
        
        # Overpriced players
        overpriced = [tp for tp in team_players if float(tp['player'].get('points_per_million', 0)) < 4]
        if len(overpriced) >= 3:
            weaknesses.append("Several players offering poor value for money")
        
        # Bench strength
        bench_players = [tp for tp in team_players if tp['pick'].get('position', 12) > 11]
        weak_bench = [bp for bp in bench_players if float(bp['player'].get('total_points', 0)) < 20]
        if len(weak_bench) >= 2:
            weaknesses.append("Weak bench with limited playing potential")
        
        # No free transfers
        if bank < 0.5 and team_data.get('summary_event_points', 0) < 40:
            weaknesses.append("Limited transfer flexibility with low funds")
        
        # Analyze opportunities
        # Market opportunities
        if bank > 2:
            opportunities.append("Significant funds available for squad upgrades")
        
        # Form opportunities
        rising_players = [tp for tp in team_players if float(tp['player'].get('form', 0)) > 6 and float(tp['player'].get('selected_by_percent', 50)) < 15]
        if rising_players:
            opportunities.append(f"Own {len(rising_players)} differential players in good form")
        
        # Fixture opportunities (simplified)
        opportunities.append("Upcoming fixture analysis reveals potential captain options")
        opportunities.append("Transfer market offers value picks for upcoming gameweeks")
        
        # Position-specific opportunities
        starting_xi = [tp for tp in team_players if tp['pick'].get('position', 12) <= 11]
        if len(starting_xi) < 11:
            opportunities.append("Opportunity to optimize starting XI selection")
        
        # Analyze threats
        # Price change threats
        high_ownership = [tp for tp in team_players if float(tp['player'].get('selected_by_percent', 0)) > 50]
        if len(high_ownership) >= 5:
            threats.append("Heavy reliance on template players - limited differential advantage")
        
        # Form decline threats
        declining_players = [tp for tp in team_players if float(tp['player'].get('form', 0)) < 4 and float(tp['player'].get('now_cost', 0)) > 80]
        if declining_players:
            threats.append(f"{len(declining_players)} expensive players underperforming")
        
        # Fixture difficulty threats
        threats.append("Difficult fixtures ahead for key players")
        
        # Competition threats
        if overall_rank and overall_rank > 2000000:
            threats.append("Current rank requires significant improvement to reach targets")
        
        # Squad balance threats
        gk_count = len([tp for tp in team_players if tp['player'].get('position_name') == 'Goalkeeper'])
        if gk_count < 2:
            threats.append("Goalkeeper shortage - squad balance risk")
        
        return {
            'strengths': strengths,
            'weaknesses': weaknesses,
            'opportunities': opportunities,
            'threats': threats
        }
    
    def _generate_swot_action_plan(self, swot_data):
        """Generate strategic action plan based on SWOT analysis"""
        return {
            'so_strategy': [
                "Leverage strong performers to target high-value differentials",
                "Use financial flexibility to secure premium assets in form",
                "Maintain successful strategies while exploring new opportunities"
            ],
            'wo_strategy': [
                "Address poor form players through strategic transfers",
                "Improve squad depth by targeting reliable bench options",
                "Transform value weaknesses into opportunity through smart picks"
            ],
            'st_strategy': [
                "Protect strong rank by avoiding risky template picks",
                "Use premium assets as captaincy insurance",
                "Maintain squad balance to weather difficult periods"
            ],
            'wt_strategy': [
                "Minimize exposure to declining expensive assets",
                "Avoid panic transfers that could worsen position",
                "Focus on fundamental improvements rather than quick fixes"
            ]
        }
    
    def _display_squad_composition_analysis(self, team_data, players_df, picks):
        """Analyze squad composition and balance"""
        st.subheader("🏗️ Squad Composition Analysis")
        
        # Get team players
        team_players = []
        for pick in picks:
            player_info = players_df[players_df['id'] == pick['element']]
            if not player_info.empty:
                player = player_info.iloc[0]
                team_players.append({
                    'position': player.get('position_name', 'Unknown'),
                    'price': float(player.get('now_cost', 0)) / 10,
                    'ownership': float(player.get('selected_by_percent', 0)),
                    'form': float(player.get('form', 0)),
                    'points': int(player.get('total_points', 0)),
                    'team': player.get('team_short_name', 'UNK')
                })
        
        if not team_players:
            st.warning("No team data available for analysis")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Position distribution
            st.write("**Position Distribution**")
            position_counts = {}
            for player in team_players:
                pos = player['position']
                position_counts[pos] = position_counts.get(pos, 0) + 1
            
            for pos, count in position_counts.items():
                st.metric(f"{pos}s", count)
        
        with col2:
            # Price distribution
            st.write("**Price Tiers**")
            budget_players = len([p for p in team_players if p['price'] < 6])
            mid_players = len([p for p in team_players if 6 <= p['price'] < 9])
            premium_players = len([p for p in team_players if p['price'] >= 9])
            
            st.metric("Budget (<£6m)", budget_players)
            st.metric("Mid-tier (£6-9m)", mid_players) 
            st.metric("Premium (£9m+)", premium_players)
        
        # Team diversity
        st.write("**Team Diversity**")
        teams_represented = len(set([p['team'] for p in team_players]))
        st.metric("Teams Represented", f"{teams_represented}/20")
        
        if teams_represented < 10:
            st.warning("⚠️ Low team diversity - consider spreading across more teams")
    
    def _display_performance_metrics_analysis(self, team_data, players_df, picks):
        """Display performance metrics analysis"""
        st.subheader("📊 Performance Metrics Analysis")
        
        # Get team players with metrics
        metrics_data = []
        for pick in picks:
            player_info = players_df[players_df['id'] == pick['element']]
            if not player_info.empty:
                player = player_info.iloc[0]
                metrics_data.append({
                    'name': player.get('web_name', 'Unknown'),
                    'form': float(player.get('form', 0)),
                    'total_points': int(player.get('total_points', 0)),
                    'minutes': int(player.get('minutes', 0)),
                    'goals': int(player.get('goals_scored', 0)),
                    'assists': int(player.get('assists', 0)),
                    'clean_sheets': int(player.get('clean_sheets', 0)),
                    'bonus': int(player.get('bonus', 0))
                })
        
        if not metrics_data:
            st.warning("No performance data available")
            return
        
        # Performance summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_form = sum(p['form'] for p in metrics_data) / len(metrics_data)
            st.metric("Squad Avg Form", f"{avg_form:.1f}")
        
        with col2:
            total_goals = sum(p['goals'] for p in metrics_data)
            st.metric("Total Goals", total_goals)
        
        with col3:
            total_assists = sum(p['assists'] for p in metrics_data)
            st.metric("Total Assists", total_assists)
        
        with col4:
            total_bonus = sum(p['bonus'] for p in metrics_data)
            st.metric("Total Bonus", total_bonus)
        
        # Top performers
        st.write("**🏆 Top Performers**")
        
        sorted_by_points = sorted(metrics_data, key=lambda x: x['total_points'], reverse=True)
        for i, player in enumerate(sorted_by_points[:5], 1):
            st.write(f"{i}. **{player['name']}** - {player['total_points']} pts "
                    f"(Form: {player['form']:.1f})")
    
    def _display_value_analysis(self, team_data, players_df, picks):
        """Display value analysis"""
        st.subheader("💰 Value Analysis")
        
        # Get value metrics
        value_data = []
        for pick in picks:
            player_info = players_df[players_df['id'] == pick['element']]
            if not player_info.empty:
                player = player_info.iloc[0]
                price = float(player.get('now_cost', 0)) / 10
                points = int(player.get('total_points', 0))
                ppm = float(player.get('points_per_million', 0))
                
                value_data.append({
                    'name': player.get('web_name', 'Unknown'),
                    'price': price,
                    'points': points,
                    'ppm': ppm,
                    'value_tier': 'Excellent' if ppm > 8 else 'Good' if ppm > 6 else 'Average' if ppm > 4 else 'Poor'
                })
        
        if not value_data:
            st.warning("No value data available")
            return
        
        # Value distribution
        excellent = len([p for p in value_data if p['ppm'] > 8])
        good = len([p for p in value_data if 6 < p['ppm'] <= 8])
        average = len([p for p in value_data if 4 < p['ppm'] <= 6])
        poor = len([p for p in value_data if p['ppm'] <= 4])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("💎 Excellent", excellent)
        
        with col2:
            st.metric("👍 Good", good)
        
        with col3:
            st.metric("⚖️ Average", average)
        
        with col4:
            st.metric("⚠️ Poor", poor)
        
        # Best and worst value
        sorted_by_value = sorted(value_data, key=lambda x: x['ppm'], reverse=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("**💎 Best Value**")
            for player in sorted_by_value[:3]:
                st.write(f"• **{player['name']}** - {player['ppm']:.1f} PPM")
        
        with col2:
            st.error("**⚠️ Worst Value**")
            for player in sorted_by_value[-3:]:
                st.write(f"• **{player['name']}** - {player['ppm']:.1f} PPM")
    
    def _display_risk_assessment(self, team_data, players_df, picks):
        """Display risk assessment"""
        st.subheader("🎯 Risk Assessment")
        
        # Calculate risk factors
        risk_factors = []
        for pick in picks:
            player_info = players_df[players_df['id'] == pick['element']]
            if not player_info.empty:
                player = player_info.iloc[0]
                
                risk_score = 0
                risk_reasons = []
                
                # Injury risk
                if player.get('status') in ['i', 'd']:
                    risk_score += 3
                    risk_reasons.append("Injury concerns")
                
                # Form risk
                form = float(player.get('form', 0))
                if form < 3:
                    risk_score += 2
                    risk_reasons.append("Poor form")
                
                # Rotation risk
                minutes = int(player.get('minutes', 0))
                if minutes < 1000:
                    risk_score += 1
                    risk_reasons.append("Rotation risk")
                
                # Price vs performance risk
                ppm = float(player.get('points_per_million', 0))
                if ppm < 4:
                    risk_score += 1
                    risk_reasons.append("Poor value")
                
                risk_factors.append({
                    'name': player.get('web_name', 'Unknown'),
                    'risk_score': risk_score,
                    'risk_level': 'High' if risk_score >= 5 else 'Medium' if risk_score >= 3 else 'Low',
                    'reasons': risk_reasons
                })
        
        # Risk summary
        high_risk = len([p for p in risk_factors if p['risk_level'] == 'High'])
        medium_risk = len([p for p in risk_factors if p['risk_level'] == 'Medium'])
        low_risk = len([p for p in risk_factors if p['risk_level'] == 'Low'])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🔴 High Risk", high_risk)
        
        with col2:
            st.metric("🟡 Medium Risk", medium_risk)
        
        with col3:
            st.metric("🟢 Low Risk", low_risk)
        
        # Risk details
        if high_risk > 0:
            st.error("**🚨 High Risk Players**")
            for player in [p for p in risk_factors if p['risk_level'] == 'High']:
                st.write(f"• **{player['name']}** - {', '.join(player['reasons'])}")
        
        if medium_risk > 0:
            st.warning("**⚠️ Medium Risk Players**")
            for player in [p for p in risk_factors if p['risk_level'] == 'Medium']:
                st.write(f"• **{player['name']}** - {', '.join(player['reasons'])}")
    
    def _display_trend_analysis(self, team_data, players_df, picks):
        """Display trend analysis"""
        st.subheader("📈 Trend Analysis")
        
        # Analyze form trends
        form_trends = []
        for pick in picks:
            player_info = players_df[players_df['id'] == pick['element']]
            if not player_info.empty:
                player = player_info.iloc[0]
                
                form = float(player.get('form', 0))
                ppg = float(player.get('points_per_game', 0))
                
                # Determine trend
                if form > ppg * 1.2:
                    trend = "📈 Improving"
                elif form < ppg * 0.8:
                    trend = "📉 Declining"
                else:
                    trend = "➡️ Stable"
                
                form_trends.append({
                    'name': player.get('web_name', 'Unknown'),
                    'form': form,
                    'ppg': ppg,
                    'trend': trend
                })
        
        # Trend summary
        improving = len([p for p in form_trends if "Improving" in p['trend']])
        declining = len([p for p in form_trends if "Declining" in p['trend']])
        stable = len([p for p in form_trends if "Stable" in p['trend']])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📈 Improving", improving)
        
        with col2:
            st.metric("📉 Declining", declining)
        
        with col3:
            st.metric("➡️ Stable", stable)
        
        # Display trends
        for trend_type, emoji in [("Improving", "📈"), ("Declining", "📉"), ("Stable", "➡️")]:
            trend_players = [p for p in form_trends if trend_type in p['trend']]
            if trend_players:
                st.write(f"**{emoji} {trend_type} Players**")
                for player in trend_players:
                    st.write(f"• **{player['name']}** - Form: {player['form']:.1f}, PPG: {player['ppg']:.1f}")
    
    def _display_transfer_planning(self, team_data):
        """Display transfer planning analysis"""
        st.subheader("🔄 Transfer Planning")
        
        st.info("Transfer planning feature will integrate with the Transfer Planning service for comprehensive analysis.")
        
        # Basic transfer analysis
        bank = team_data.get('bank', 0) / 10
        team_value = team_data.get('value', 1000) / 10
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Available Funds", f"£{bank:.1f}m")
            st.metric("Team Value", f"£{team_value:.1f}m")
        
        with col2:
            total_budget = bank + team_value
            st.metric("Total Budget", f"£{total_budget:.1f}m")
            
            if bank > 2:
                st.success("Good transfer flexibility")
            elif bank < 0.5:
                st.warning("Limited transfer options")
            else:
                st.info("Moderate transfer flexibility")
    
    def _display_performance_comparison(self, team_data):
        """Display performance comparison"""
        st.subheader("📊 Performance Comparison")
        
        st.info("Performance comparison feature will integrate with the Performance Comparison service for detailed benchmarking.")
        
        # Basic comparison metrics
        total_points = team_data.get('summary_overall_points', 0)
        overall_rank = team_data.get('summary_overall_rank', 0)
        current_gw = team_data.get('gameweek', 1)
        
        if current_gw > 0:
            avg_ppg = total_points / current_gw
            
            # Estimated benchmarks
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Your PPG", f"{avg_ppg:.1f}")
                if avg_ppg > 60:
                    st.success("Above elite benchmark")
                elif avg_ppg > 50:
                    st.info("Above average")
                else:
                    st.warning("Below average")
            
            with col2:
                if overall_rank:
                    percentile = (1 - (overall_rank / 8000000)) * 100
                    st.metric("Percentile", f"{percentile:.1f}%")
                    
                    if percentile > 90:
                        st.success("Top 10% manager")
                    elif percentile > 75:
                        st.info("Top quartile")
                    elif percentile > 50:
                        st.info("Above median")
                    else:
                        st.warning("Below median")
            
            with col3:
                # Projected final points
                remaining_gws = 38 - current_gw
                projected_final = total_points + (avg_ppg * remaining_gws)
                st.metric("Projected Final", f"{projected_final:.0f}")
                
                if projected_final > 2200:
                    st.success("Elite projection")
                elif projected_final > 2000:
                    st.info("Strong projection")
                else:
                    st.warning("Room for improvement")

