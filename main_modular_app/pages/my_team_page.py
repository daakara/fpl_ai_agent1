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
        tab1, tab2, tab3, tab4 = st.tabs([
            "👥 Current Squad", 
            "📊 Performance Analysis", 
            "💡 Recommendations",
            "⭐ Starting XI Optimizer"
        ])
        
        with tab1:
            self._display_current_squad(team_data)
        
        with tab2:
            self._display_performance_analysis(team_data)
        
        with tab3:
            self._display_recommendations(team_data)
        
        with tab4:
            self._display_starting_xi_optimizer(team_data)
        
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

