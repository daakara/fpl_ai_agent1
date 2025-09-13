"""
My Team Page - Comprehensive team analysis and recommendations
"""
import streamlit as st
import pandas as pd
from typing import Dict, List, Optional
import plotly.express as px


class MyTeamPage:
    """My team page with comprehensive analysis and recommendations"""
    
    def __init__(self):
        """Initialize the my team page"""
        pass
    
    def render(self):
        """Render the my team page with comprehensive analysis tabs"""
        st.header("👤 My FPL Team")
        
        # Team import section - moved to top for better UX
        if 'my_team_df' not in st.session_state or st.session_state.my_team_df.empty:
            st.subheader("📥 Import Your FPL Team")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                team_id = st.text_input(
                    "Enter your FPL Team ID:",
                    help="Find your team ID in the URL when viewing your team on the FPL website (e.g., fantasy.premierleague.com/entry/1234567/...)",
                    placeholder="e.g., 1234567"
                )
            
            with col2:
                st.write("")  # Space for alignment
                if st.button("📊 Load My Team", type="primary"):
                    if team_id:
                        self._load_user_team(team_id)
                    else:
                        st.warning("Please enter your team ID")
            
            # Instructions
            with st.expander("💡 How to find your Team ID", expanded=False):
                st.markdown("""
                **Step-by-step guide:**
                1. Go to [fantasy.premierleague.com](https://fantasy.premierleague.com)
                2. Log into your FPL account
                3. Navigate to your team's "Points" or "Transfers" page
                4. Look at the URL: `https://fantasy.premierleague.com/entry/1234567/event/X`
                5. Your Team ID is the number after `/entry/` (e.g., `1234567`)
                """)
            
            st.info("💡 Import your team to unlock comprehensive analysis, transfer suggestions, and strategic insights!")
            return
        
        my_team_df = st.session_state.my_team_df
        
        # Quick team overview at the top
        self._render_quick_overview(my_team_df)
        
        # Main navigation tabs with all requested sections
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "👥 Current Squad Analysis",
            "📊 Performance Analysis & Benchmarking", 
            "🔄 Smart Transfer Suggestions",
            "🏆 Performance Benchmarking & Comparisons",
            "🎯 Chip Strategy & Planning",
            "⚔️ SWOT Analysis"
        ])
        
        with tab1:
            self._render_current_squad_analysis(my_team_df)
        
        with tab2:
            self._render_performance_analysis_benchmarking(my_team_df)
        
        with tab3:
            self._render_smart_transfer_suggestions(my_team_df)
            
        with tab4:
            self._render_performance_benchmarking_comparisons(my_team_df)
            
        with tab5:
            self._render_chip_strategy_planning(my_team_df)
            
        with tab6:
            self._render_swot_analysis(my_team_df)

    def _render_quick_overview(self, my_team_df):
        """Render quick team overview metrics"""
        st.subheader(f"⚽ Team Overview")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
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
                st.metric("Avg Value", f"{avg_ppm:.1f}")
        
        with col5:
            budget_remaining = 100.0 - total_cost
            st.metric("Budget Left", f"£{budget_remaining:.1f}m")
        
        st.divider()

    def _render_current_squad_analysis(self, my_team_df):
        """Render comprehensive current squad analysis"""
        squad_tab1, squad_tab2, squad_tab3, squad_tab4 = st.tabs([
            "🏃 Starting XI Analysis",
            "🪑 Bench Analysis",
            "👑 Captaincy Analysis", 
            "💡 Squad Insights"
        ])
        
        with squad_tab1:
            self._render_starting_xi_detailed(my_team_df)
            
        with squad_tab2:
            self._render_bench_detailed(my_team_df)
            
        with squad_tab3:
            self._render_captaincy_detailed(my_team_df)
            
        with squad_tab4:
            self._render_squad_insights_detailed(my_team_df)

    def _render_performance_analysis_benchmarking(self, my_team_df):
        """Render performance analysis and benchmarking"""
        perf_tab1, perf_tab2, perf_tab3, perf_tab4 = st.tabs([
            "📈 Player Performance",
            "📊 Team Analytics",
            "🎯 Performance Metrics",
            "📉 Trend Analysis"
        ])
        
        with perf_tab1:
            self._render_player_performance_detailed(my_team_df)
            
        with perf_tab2:
            self._render_team_analytics_detailed(my_team_df)
            
        with perf_tab3:
            self._render_performance_metrics(my_team_df)
            
        with perf_tab4:
            self._render_trend_analysis(my_team_df)

    def _render_smart_transfer_suggestions(self, my_team_df):
        """Render smart transfer suggestions"""
        transfer_tab1, transfer_tab2, transfer_tab3, transfer_tab4 = st.tabs([
            "📤 Transfer Out Analysis",
            "📥 Transfer In Targets",
            "💰 Budget Planning",
            "🎯 Strategic Planning"
        ])
        
        with transfer_tab1:
            self._render_transfer_out_detailed(my_team_df)
            
        with transfer_tab2:
            self._render_transfer_in_detailed(my_team_df)
            
        with transfer_tab3:
            self._render_budget_planning(my_team_df)
            
        with transfer_tab4:
            self._render_transfer_strategic_planning(my_team_df)

    def _render_performance_benchmarking_comparisons(self, my_team_df):
        """Render performance benchmarking and comparisons"""
        bench_tab1, bench_tab2, bench_tab3, bench_tab4 = st.tabs([
            "🏆 Elite Comparison",
            "📊 League Averages",
            "🎯 Goal Setting",
            "📈 Progress Tracking"
        ])
        
        with bench_tab1:
            self._render_elite_comparison(my_team_df)
            
        with bench_tab2:
            self._render_league_averages(my_team_df)
            
        with bench_tab3:
            self._render_goal_setting(my_team_df)
            
        with bench_tab4:
            self._render_progress_tracking(my_team_df)

    def _render_chip_strategy_planning(self, my_team_df):
        """Render comprehensive chip strategy and planning"""
        chip_tab1, chip_tab2, chip_tab3, chip_tab4 = st.tabs([
            "⏰ Optimal Timing",
            "📊 Chip Analysis",
            "🔮 Strategic Planning",
            "📈 Advanced Strategy"
        ])
        
        with chip_tab1:
            self._render_chip_optimal_timing(my_team_df)
            
        with chip_tab2:
            self._render_chip_analysis(my_team_df)
            
        with chip_tab3:
            self._render_chip_strategic_planning(my_team_df)
            
        with chip_tab4:
            self._render_chip_advanced_strategy(my_team_df)

    def _render_swot_analysis(self, my_team_df):
        """Render comprehensive SWOT analysis"""
        swot_tab1, swot_tab2, swot_tab3, swot_tab4 = st.tabs([
            "💪 Strengths",
            "⚠️ Weaknesses",
            "🌟 Opportunities",
            "⚡ Threats"
        ])
        
        with swot_tab1:
            self._render_team_strengths(my_team_df)
            
        with swot_tab2:
            self._render_team_weaknesses(my_team_df)
            
        with swot_tab3:
            self._render_team_opportunities(my_team_df)
            
        with swot_tab4:
            self._render_team_threats(my_team_df)

    # Detailed rendering methods for each sub-tab
    def _render_starting_xi_detailed(self, my_team_df):
        """Detailed starting XI analysis"""
        if 'pick_position' not in my_team_df.columns:
            st.warning("⚠️ Could not determine starting XI. Please reload your team.")
            return

        starting_xi = my_team_df[my_team_df['pick_position'] <= 11].sort_values('pick_position')
        
        if starting_xi.empty:
            st.warning("No starting XI data available.")
            return
        
        # Formation display
        if 'position' in starting_xi.columns:
            formation = starting_xi['position'].value_counts()
            gk = formation.get('GK', 0)
            def_players = formation.get('DEF', 0) 
            mid = formation.get('MID', 0)
            fwd = formation.get('FWD', 0)
            
            st.success(f"**Formation**: {gk}-{def_players}-{mid}-{fwd}")
        
        # Key metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            starting_points = starting_xi['total_points'].sum()
            st.metric("Starting XI Points", f"{starting_points:,}")
        
        with col2:
            starting_value = starting_xi['now_cost'].sum()
            st.metric("Starting XI Value", f"£{starting_value:.1f}m")
        
        with col3:
            if 'form' in starting_xi.columns:
                avg_form = pd.to_numeric(starting_xi['form'], errors='coerce').fillna(0).mean()
                st.metric("Avg Form", f"{avg_form:.1f}")
        
        # Starting XI table with enhanced display
        display_cols = ['web_name', 'position', 'team_name', 'now_cost', 'total_points']
        if 'form' in starting_xi.columns:
            display_cols.append('form')
        
        display_df = starting_xi[display_cols].copy()
        
        # Add status column
        status_list = []
        for idx, row in starting_xi.iterrows():
            if row.get('is_captain', False):
                status_list.append('👑 Captain')
            elif row.get('is_vice_captain', False):
                status_list.append('🅥 Vice-Captain')
            else:
                status_list.append('✅ Starting')
        
        display_df['Status'] = status_list
        
        st.dataframe(
            display_df,
            use_container_width=True,
            column_config={
                'web_name': 'Player',
                'position': 'Position',
                'team_name': 'Team',
                'now_cost': st.column_config.NumberColumn('Price', format="£%.1f"),
                'total_points': st.column_config.NumberColumn('Points'),
                'form': 'Form',
                'Status': 'Status'
            }
        )

    def _render_bench_detailed(self, my_team_df):
        """Detailed bench analysis"""
        if 'pick_position' not in my_team_df.columns:
            st.warning("⚠️ Could not determine bench. Please reload your team.")
            return

        bench = my_team_df[my_team_df['pick_position'] > 11].sort_values('pick_position')
        
        if bench.empty:
            st.warning("No bench data available.")
            return
        
        # Bench strength analysis
        bench_points = bench['total_points'].sum()
        starting_xi_points = my_team_df[my_team_df['pick_position'] <= 11]['total_points'].sum()
        total_points = bench_points + starting_xi_points
        bench_percentage = (bench_points / total_points) * 100 if total_points > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Bench Points", f"{bench_points:,}")
        
        with col2:
            st.metric("Bench Strength", f"{bench_percentage:.1f}%")
        
        with col3:
            avg_bench_cost = bench['now_cost'].mean()
            st.metric("Avg Bench Cost", f"£{avg_bench_cost:.1f}m")
        
        # Bench quality assessment
        if bench_percentage < 12:
            st.error("🔴 **Weak Bench** - Consider strengthening for Bench Boost potential")
        elif bench_percentage > 25:
            st.success("🟢 **Strong Bench** - Great for Bench Boost strategy!")
        else:
            st.info("🟡 **Average Bench** - Reasonable but could be improved")
        
        # Bench players table
        display_cols = ['web_name', 'position', 'team_name', 'now_cost', 'total_points']
        if 'form' in bench.columns:
            display_cols.append('form')
        
        st.dataframe(
            bench[display_cols],
            use_container_width=True,
            column_config={
                'web_name': 'Player',
                'position': 'Position', 
                'team_name': 'Team',
                'now_cost': st.column_config.NumberColumn('Price', format="£%.1f"),
                'total_points': st.column_config.NumberColumn('Points'),
                'form': 'Form'
            }
        )

    def _render_captaincy_detailed(self, my_team_df):
        """Detailed captaincy analysis"""
        if 'pick_position' not in my_team_df.columns:
            st.warning("⚠️ Could not determine starting XI for captaincy analysis.")
            return

        starting_xi = my_team_df[my_team_df['pick_position'] <= 11]
        captain = starting_xi[starting_xi.get('is_captain', False)]
        vice_captain = starting_xi[starting_xi.get('is_vice_captain', False)]

        col1, col2 = st.columns(2)
        
        # Current captain analysis
        with col1:
            st.write("**👑 Current Captain**")
            if not captain.empty:
                cap_info = captain.iloc[0]
                st.success(f"**{cap_info['web_name']}** ({cap_info.get('team_name', 'Unknown')})")
                
                cap_col1, cap_col2 = st.columns(2)
                with cap_col1:
                    st.metric("Total Points", f"{cap_info.get('total_points', 0):,}")
                with cap_col2:
                    st.metric("Form", f"{cap_info.get('form', 'N/A')}")
            else:
                st.warning("❌ No captain selected")
        
        # Current vice-captain
        with col2:
            st.write("**🅥 Vice Captain**")
            if not vice_captain.empty:
                vc_info = vice_captain.iloc[0]
                st.info(f"**{vc_info['web_name']}** ({vc_info.get('team_name', 'Unknown')})")
                
                vc_col1, vc_col2 = st.columns(2)
                with vc_col1:
                    st.metric("Total Points", f"{vc_info.get('total_points', 0):,}")
                with vc_col2:
                    st.metric("Form", f"{vc_info.get('form', 'N/A')}")
            else:
                st.warning("❌ No vice-captain selected")
        
        # Captain alternatives
        st.write("**🔄 Alternative Captain Options**")
        if 'form' in starting_xi.columns:
            candidates = starting_xi.copy()
            candidates['form'] = pd.to_numeric(candidates['form'], errors='coerce').fillna(0)
            candidates['captain_score'] = candidates['form'] * (candidates['total_points'] / 100)
            
            top_candidates = candidates.nlargest(5, 'captain_score')
            
            st.dataframe(
                top_candidates[['web_name', 'team_name', 'form', 'total_points', 'captain_score']],
                use_container_width=True,
                column_config={
                    'web_name': 'Player',
                    'team_name': 'Team',
                    'form': 'Form',
                    'total_points': 'Points',
                    'captain_score': 'Captain Score'
                }
            )
        else:
            st.info("Form data required for captain analysis.")

    def _render_squad_insights_detailed(self, my_team_df):
        """Detailed squad insights and analysis"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📊 Squad Distribution**")
            
            # Team distribution
            if 'team_name' in my_team_df.columns:
                team_counts = my_team_df['team_name'].value_counts()
                st.dataframe(team_counts.head(8))
                
                if any(team_counts > 3):
                    st.warning("⚠️ You have more than 3 players from the same team!")
                else:
                    st.success("✅ Good team distribution")
            
            # Position distribution
            if 'position' in my_team_df.columns:
                pos_counts = my_team_df['position'].value_counts()
                st.write("**Position Breakdown:**")
                for pos, count in pos_counts.items():
                    st.write(f"• {pos}: {count} players")
        
        with col2:
            st.write("**💡 Key Insights**")
            
            insights = []
            
            # Value insights
            if 'points_per_million' in my_team_df.columns:
                low_value = my_team_df[my_team_df['points_per_million'] < 5.0]
                if not low_value.empty:
                    insights.append(f"⚠️ {len(low_value)} players with poor value (PPM < 5.0)")
                
                excellent_value = my_team_df[my_team_df['points_per_million'] > 10.0]
                if not excellent_value.empty:
                    insights.append(f"💎 {len(excellent_value)} excellent value players (PPM > 10.0)")
            
            # Form insights
            if 'form' in my_team_df.columns:
                poor_form = my_team_df[pd.to_numeric(my_team_df['form'], errors='coerce').fillna(0) < 3.0]
                if not poor_form.empty:
                    insights.append(f"📉 {len(poor_form)} players in poor form (< 3.0)")
                
                excellent_form = my_team_df[pd.to_numeric(my_team_df['form'], errors='coerce').fillna(0) > 7.0]
                if not excellent_form.empty:
                    insights.append(f"🔥 {len(excellent_form)} players in excellent form (> 7.0)")
            
            # Ownership insights
            if 'selected_by_percent' in my_team_df.columns:
                high_ownership = my_team_df[my_team_df['selected_by_percent'] > 50.0]
                if not high_ownership.empty:
                    insights.append(f"📈 {len(high_ownership)} highly owned players (> 50%)")
                
                differentials = my_team_df[my_team_df['selected_by_percent'] < 10.0]
                if not differentials.empty:
                    insights.append(f"💎 {len(differentials)} differential picks (< 10% owned)")
            
            # Display insights
            if insights:
                for insight in insights:
                    st.write(insight)
            else:
                st.success("✅ Squad analysis looks positive overall!")

    def _render_transfer_out_detailed(self, my_team_df):
        """Detailed transfer out analysis"""
        st.write("**📤 Players to Consider Transferring Out**")
        
        transfer_candidates = []
        
        # Poor value players
        if 'points_per_million' in my_team_df.columns:
            poor_value = my_team_df[my_team_df['points_per_million'] < 4.0]
            for _, player in poor_value.iterrows():
                transfer_candidates.append({
                    'Player': player['web_name'],
                    'Reason': f"Poor value (PPM: {player['points_per_million']:.1f})",
                    'Priority': 'High',
                    'Price': f"£{player['now_cost']:.1f}m"
                })
        
        # Poor form players
        if 'form' in my_team_df.columns:
            poor_form = my_team_df[pd.to_numeric(my_team_df['form'], errors='coerce').fillna(0) < 2.5]
            for _, player in poor_form.iterrows():
                form_val = pd.to_numeric(player['form'], errors='coerce')
                transfer_candidates.append({
                    'Player': player['web_name'],
                    'Reason': f"Poor form ({form_val:.1f})",
                    'Priority': 'Medium',
                    'Price': f"£{player['now_cost']:.1f}m"
                })
        
        # Expensive underperformers
        expensive_players = my_team_df[my_team_df['now_cost'] > 8.0]
        if not expensive_players.empty and 'points_per_million' in expensive_players.columns:
            underperforming_expensive = expensive_players[expensive_players['points_per_million'] < 6.0]
            for _, player in underperforming_expensive.iterrows():
                transfer_candidates.append({
                    'Player': player['web_name'],
                    'Reason': f"Expensive but underperforming (£{player['now_cost']:.1f}m, PPM: {player['points_per_million']:.1f})",
                    'Priority': 'High',
                    'Price': f"£{player['now_cost']:.1f}m"
                })
        
        if transfer_candidates:
            # Remove duplicates
            candidates_df = pd.DataFrame(transfer_candidates).drop_duplicates(subset=['Player'])
            
            # Color code by priority
            st.dataframe(
                candidates_df,
                use_container_width=True,
                column_config={
                    'Player': 'Player Name',
                    'Reason': 'Transfer Reason', 
                    'Priority': 'Priority Level',
                    'Price': 'Current Price'
                }
            )
            
            if len(candidates_df) > 0:
                st.info(f"💡 Found {len(candidates_df)} potential transfer candidates. Focus on 'High' priority players first.")
        else:
            st.success("✅ No obvious transfer out candidates based on current analysis!")

    def _render_transfer_in_detailed(self, my_team_df):
        """Detailed transfer in suggestions"""
        st.write("**📥 Recommended Transfer Targets**")
        
        if 'players_df' not in st.session_state or st.session_state.players_df.empty:
            st.info("💡 Load player data from the sidebar to see transfer suggestions.")
            return
        
        players_df = st.session_state.players_df
        current_player_ids = my_team_df.index.tolist()
        available_players = players_df[~players_df.index.isin(current_player_ids)]
        
        if available_players.empty:
            st.warning("No available players found.")
            return
        
        # Best value picks by position
        for position in ['GK', 'DEF', 'MID', 'FWD']:
            pos_players = available_players[available_players.get('position', '') == position]
            
            if not pos_players.empty and 'points_per_million' in pos_players.columns:
                # Filter for reasonable options
                filtered_pos = pos_players[
                    (pos_players['total_points'] > 20) &  # Minimum points threshold
                    (pos_players['now_cost'] <= 15.0)      # Maximum price filter
                ]
                
                if not filtered_pos.empty:
                    top_targets = filtered_pos.nlargest(5, 'points_per_million')
                    
                    with st.expander(f"**🎯 Top {position} Targets**", expanded=position in ['MID', 'FWD']):
                        display_cols = ['web_name', 'team_name', 'now_cost', 'total_points', 'points_per_million']
                        if 'form' in top_targets.columns:
                            display_cols.append('form')
                        
                        st.dataframe(
                            top_targets[display_cols],
                            use_container_width=True,
                            column_config={
                                'web_name': 'Player',
                                'team_name': 'Team',
                                'now_cost': st.column_config.NumberColumn('Price', format="£%.1f"),
                                'total_points': 'Points',
                                'points_per_million': 'PPM',
                                'form': 'Form'
                            }
                        )

    def _render_budget_planning(self, my_team_df):
        """Detailed budget analysis and planning"""
        st.write("**💰 Budget Analysis & Planning**")
        
        current_value = my_team_df['now_cost'].sum()
        budget_remaining = 100.0 - current_value
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Squad Value", f"£{current_value:.1f}m")
        
        with col2:
            st.metric("Budget Remaining", f"£{budget_remaining:.1f}m")
        
        with col3:
            avg_player_cost = current_value / len(my_team_df)
            st.metric("Avg Player Cost", f"£{avg_player_cost:.1f}m")
        
        # Budget recommendations
        if budget_remaining < 0.5:
            st.error("🔴 **Critical Budget Issue** - Very tight budget! Consider downgrading expensive underperformers.")
        elif budget_remaining < 1.0:
            st.warning("🟡 **Limited Budget** - Be cautious with transfers. Look for sideways moves.")
        elif budget_remaining > 2.0:
            st.success("🟢 **Good Budget Flexibility** - You have room to upgrade key positions.")
        else:
            st.info("🔵 **Balanced Budget** - Standard budget allocation.")
        
        # Position value breakdown
        st.write("**📊 Budget Allocation by Position**")
        
        if 'position' in my_team_df.columns:
            pos_spending = my_team_df.groupby('position')['now_cost'].agg(['sum', 'mean', 'count']).round(1)
            pos_spending.columns = ['Total Spent', 'Avg Cost', 'Players']
            
            st.dataframe(
                pos_spending,
                use_container_width=True,
                column_config={
                    'Total Spent': st.column_config.NumberColumn('Total (£m)', format="£%.1f"),
                    'Avg Cost': st.column_config.NumberColumn('Average (£m)', format="£%.1f"),
                    'Players': 'Count'
                }
            )

    def _render_transfer_strategic_planning(self, my_team_df):
        """Render transfer strategic planning"""
        st.write("**🎯 Transfer Strategy Planning**")
        
        strategy_options = [
            "🔥 Premium Focus - Target high-cost, high-performance players",
            "💰 Value Hunt - Find budget players with excellent PPM",
            "📈 Form Focus - Target players with rising form",
            "💎 Differential Strategy - Low ownership, high potential players"
        ]
        
        selected_strategy = st.selectbox("Choose Your Transfer Strategy:", strategy_options)
        
        if "Premium Focus" in selected_strategy:
            st.info("**Strategy**: Focus on premium players (£9m+) with consistent returns")
            st.write("• Target proven performers with reliable minutes")
            st.write("• Consider captaincy potential")
            st.write("• Accept higher ownership for reliability")
            
        elif "Value Hunt" in selected_strategy:
            st.info("**Strategy**: Find budget-friendly players with excellent value")
            st.write("• Target players with PPM > 8.0")
            st.write("• Look for price rises potential")
            st.write("• Enable funds for premium upgrades")
            
        elif "Form Focus" in selected_strategy:
            st.info("**Strategy**: Ride the wave of in-form players")
            st.write("• Target players with form > 6.0")
            st.write("• Consider underlying stats (xG/xA)")
            st.write("• Be ready to move quickly when form drops")
            
        elif "Differential Strategy" in selected_strategy:
            st.info("**Strategy**: Take calculated risks on low-owned players")
            st.write("• Target ownership < 15%")
            st.write("• Look for fixture swings")
            st.write("• Higher risk, higher reward approach")

    def _render_player_performance_detailed(self, my_team_df):
        """Detailed player performance analysis"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🎯 Top Performers**")
            if 'total_points' in my_team_df.columns:
                top_performers = my_team_df.nlargest(5, 'total_points')[['web_name', 'total_points', 'position', 'now_cost']]
                st.dataframe(top_performers, use_container_width=True)
        
        with col2:
            st.write("**📉 Need Improvement**")
            if 'total_points' in my_team_df.columns:
                underperformers = my_team_df.nsmallest(5, 'total_points')[['web_name', 'total_points', 'position', 'now_cost']]
                st.dataframe(underperformers, use_container_width=True)
        
        # Value analysis
        if 'points_per_million' in my_team_df.columns:
            st.write("**💰 Value Analysis**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Best Value Players:**")
                best_value = my_team_df.nlargest(5, 'points_per_million')[['web_name', 'points_per_million', 'now_cost']]
                st.dataframe(best_value, use_container_width=True)
            
            with col2:
                st.write("**Poor Value Players:**")
                worst_value = my_team_df.nsmallest(5, 'points_per_million')[['web_name', 'points_per_million', 'now_cost']]
                st.dataframe(worst_value, use_container_width=True)

    def _render_team_analytics_detailed(self, my_team_df):
        """Detailed team analytics"""
        # Visual analytics with charts
        col1, col2 = st.columns(2)

        with col1:
            # Points contribution by player
            if 'total_points' in my_team_df.columns and 'web_name' in my_team_df.columns:
                fig = px.bar(
                    my_team_df.sort_values('total_points', ascending=True),
                    x='total_points',
                    y='web_name',
                    orientation='h',
                    title="Player Points Contribution",
                    labels={'web_name': 'Player', 'total_points': 'Total Points'}
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Points by position
            if 'position' in my_team_df.columns and 'total_points' in my_team_df.columns:
                points_by_pos = my_team_df.groupby('position')['total_points'].sum().reset_index()
                fig = px.pie(
                    points_by_pos,
                    values='total_points',
                    names='position',
                    title="Points Distribution by Position",
                    hole=0.3
                )
                st.plotly_chart(fig, use_container_width=True)

    def _render_performance_metrics(self, my_team_df):
        """Render detailed performance metrics"""
        st.write("**🎯 Key Performance Indicators**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_points = my_team_df['total_points'].sum()
            st.metric("Total Points", f"{total_points:,}")
        
        with col2:
            if 'points_per_million' in my_team_df.columns:
                avg_ppm = my_team_df['points_per_million'].mean()
                st.metric("Avg PPM", f"{avg_ppm:.1f}")
        
        with col3:
            squad_value = my_team_df['now_cost'].sum()
            st.metric("Squad Value", f"£{squad_value:.1f}m")
        
        with col4:
            if 'form' in my_team_df.columns:
                avg_form = pd.to_numeric(my_team_df['form'], errors='coerce').fillna(0).mean()
                st.metric("Avg Form", f"{avg_form:.1f}")
        
        # Performance breakdown by position
        st.write("**📊 Performance by Position**")
        if 'position' in my_team_df.columns:
            pos_analysis = my_team_df.groupby('position').agg({
                'total_points': ['sum', 'mean'],
                'now_cost': ['sum', 'mean'],
                'points_per_million': 'mean' if 'points_per_million' in my_team_df.columns else lambda x: None
            }).round(2)
            
            pos_analysis.columns = ['Total Points', 'Avg Points', 'Total Cost', 'Avg Cost', 'Avg PPM']
            st.dataframe(pos_analysis, use_container_width=True)

    def _render_trend_analysis(self, my_team_df):
        """Render trend analysis"""
        st.write("**📈 Form and Trend Analysis**")
        
        if 'form' in my_team_df.columns:
            form_data = my_team_df[['web_name', 'form', 'total_points']].copy()
            form_data['form'] = pd.to_numeric(form_data['form'], errors='coerce').fillna(0)
            
            # Form distribution
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🔥 Players in Excellent Form (>6.0)**")
                excellent_form = form_data[form_data['form'] > 6.0].sort_values('form', ascending=False)
                if not excellent_form.empty:
                    st.dataframe(excellent_form, use_container_width=True)
                else:
                    st.info("No players in excellent form")
            
            with col2:
                st.write("**📉 Players in Poor Form (<3.0)**")
                poor_form = form_data[form_data['form'] < 3.0].sort_values('form', ascending=True)
                if not poor_form.empty:
                    st.dataframe(poor_form, use_container_width=True)
                else:
                    st.success("No players in poor form!")

    def _render_benchmarking_detailed(self, my_team_df):
        """Detailed benchmarking analysis"""
        st.write("**🏆 Performance Benchmarking**")
        
        if 'players_df' in st.session_state and not st.session_state.players_df.empty:
            players_df = st.session_state.players_df
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                my_avg_points = my_team_df['total_points'].mean()
                overall_avg_points = players_df['total_points'].mean()
                diff = my_avg_points - overall_avg_points
                st.metric("Avg Points vs League", f"{my_avg_points:.1f}", f"{diff:+.1f}")
            
            with col2:
                if 'points_per_million' in my_team_df.columns:
                    my_avg_ppm = my_team_df['points_per_million'].mean()
                    overall_avg_ppm = players_df['points_per_million'].mean()
                    diff_ppm = my_avg_ppm - overall_avg_ppm
                    st.metric("Avg PPM vs League", f"{my_avg_ppm:.1f}", f"{diff_ppm:+.1f}")
            
            with col3:
                my_total_value = my_team_df['now_cost'].sum()
                st.metric("Squad Value", f"£{my_total_value:.1f}m")
        else:
            st.info("Load player data for benchmarking analysis.")

    def _render_elite_comparison(self, my_team_df):
        """Render elite manager comparison"""
        st.write("**🏆 Elite Manager Comparison**")
        
        # Simulated elite benchmarks (in a real app, these would come from API)
        elite_benchmarks = {
            "Top 1k": {"avg_points": 85.2, "avg_ppm": 9.8, "squad_value": 99.2},
            "Top 10k": {"avg_points": 78.5, "avg_ppm": 8.9, "squad_value": 98.8},
            "Top 100k": {"avg_points": 72.1, "avg_ppm": 8.2, "squad_value": 98.1},
            "Top 500k": {"avg_points": 65.8, "avg_ppm": 7.6, "squad_value": 97.5}
        }
        
        # Your team metrics
        your_avg_points = my_team_df['total_points'].mean()
        your_avg_ppm = my_team_df['points_per_million'].mean() if 'points_per_million' in my_team_df.columns else 0
        your_squad_value = my_team_df['now_cost'].sum()
        
        comparison_data = []
        for tier, metrics in elite_benchmarks.items():
            comparison_data.append({
                "Tier": tier,
                "Elite Avg Points": metrics["avg_points"],
                "Your Avg Points": your_avg_points,
                "Points Gap": your_avg_points - metrics["avg_points"],
                "Elite PPM": metrics["avg_ppm"],
                "Your PPM": your_avg_ppm,
                "PPM Gap": your_avg_ppm - metrics["avg_ppm"]
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Performance assessment
        if your_avg_points >= elite_benchmarks["Top 1k"]["avg_points"]:
            st.success("🏆 **Elite Performance** - You're performing at Top 1k level!")
        elif your_avg_points >= elite_benchmarks["Top 10k"]["avg_points"]:
            st.success("🥇 **Excellent Performance** - You're in Top 10k territory!")
        elif your_avg_points >= elite_benchmarks["Top 100k"]["avg_points"]:
            st.info("🥈 **Good Performance** - You're performing well!")
        else:
            st.warning("📈 **Room for Improvement** - Focus on value and consistency!")

    def _render_league_averages(self, my_team_df):
        """Render league average comparisons"""
        st.write("**📊 League Average Comparisons**")
        
        # Simulated league averages
        league_avg = {
            "avg_points_per_player": 45.2,
            "avg_ppm": 6.8,
            "avg_squad_value": 96.5,
            "avg_form": 4.2
        }
        
        your_metrics = {
            "avg_points_per_player": my_team_df['total_points'].mean(),
            "avg_ppm": my_team_df['points_per_million'].mean() if 'points_per_million' in my_team_df.columns else 0,
            "avg_squad_value": my_team_df['now_cost'].sum(),
            "avg_form": pd.to_numeric(my_team_df['form'], errors='coerce').fillna(0).mean() if 'form' in my_team_df.columns else 0
        }
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            diff_points = your_metrics["avg_points_per_player"] - league_avg["avg_points_per_player"]
            st.metric("Avg Points vs League", f"{your_metrics['avg_points_per_player']:.1f}", f"{diff_points:+.1f}")
        
        with col2:
            diff_ppm = your_metrics["avg_ppm"] - league_avg["avg_ppm"]
            st.metric("PPM vs League", f"{your_metrics['avg_ppm']:.1f}", f"{diff_ppm:+.1f}")
        
        with col3:
            diff_value = your_metrics["avg_squad_value"] - league_avg["avg_squad_value"]
            st.metric("Squad Value vs League", f"£{your_metrics['avg_squad_value']:.1f}m", f"£{diff_value:+.1f}m")
        
        with col4:
            diff_form = your_metrics["avg_form"] - league_avg["avg_form"]
            st.metric("Form vs League", f"{your_metrics['avg_form']:.1f}", f"{diff_form:+.1f}")

    def _render_goal_setting(self, my_team_df):
        """Render goal setting interface"""
        st.write("**🎯 Season Goals & Targets**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Set Your Targets:**")
            target_rank = st.number_input("Target Overall Rank:", min_value=1, value=100000, step=1000)
            target_points = st.number_input("Target Total Points:", min_value=1000, value=2000, step=50)
            target_ppm = st.number_input("Target Avg PPM:", min_value=5.0, value=8.0, step=0.1)
            
            if st.button("💾 Save Targets"):
                st.session_state.target_rank = target_rank
                st.session_state.target_points = target_points
                st.session_state.target_ppm = target_ppm
                st.success("✅ Targets saved!")
        
        with col2:
            st.write("**Current Progress:**")
            if all(key in st.session_state for key in ['target_rank', 'target_points', 'target_ppm']):
                current_points = my_team_df['total_points'].sum()
                current_ppm = my_team_df['points_per_million'].mean() if 'points_per_million' in my_team_df.columns else 0
                
                points_to_target = st.session_state.target_points - current_points
                ppm_to_target = st.session_state.target_ppm - current_ppm
                
                st.metric("Target Rank", f"{st.session_state.target_rank:,}")
                st.metric("Points to Target", f"{st.session_state.target_points:,}", f"{points_to_target:+,}")
                st.metric("PPM to Target", f"{st.session_state.target_ppm:.1f}", f"{ppm_to_target:+.1f}")
            else:
                st.info("Set your targets to track progress")

    def _render_progress_tracking(self, my_team_df):
        """Render progress tracking"""
        st.write("**📈 Progress Tracking & Analytics**")
        
        # Progress indicators
        total_points = my_team_df['total_points'].sum()
        avg_ppm = my_team_df['points_per_million'].mean() if 'points_per_million' in my_team_df.columns else 0
        
        # Performance tier classification
        if total_points >= 600:
            tier = "🏆 Elite Tier"
            tier_color = "success"
        elif total_points >= 500:
            tier = "🥇 Excellent Tier"
            tier_color = "success"
        elif total_points >= 400:
            tier = "🥈 Good Tier"
            tier_color = "info"
        else:
            tier = "📈 Building Tier"
            tier_color = "warning"
        
        st.success(f"**Current Performance Tier**: {tier}")
        
        # Progress metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Points", f"{total_points:,}")
        
        with col2:
            st.metric("Average PPM", f"{avg_ppm:.1f}")
        
        with col3:
            reliability_score = len(my_team_df[my_team_df['total_points'] > 20]) / len(my_team_df) * 100
            st.metric("Squad Reliability", f"{reliability_score:.0f}%")

    def _render_chip_optimal_timing(self, my_team_df):
        """Render chip optimal timing analysis"""
        st.write("**⏰ Optimal Chip Timing Strategy**")
        
        # Season phase analysis
        current_gw = 10  # This would come from API in real implementation
        
        if current_gw <= 10:
            phase = "Early Season (GW 1-10)"
            recommendations = {
                "Wildcard": "🟢 Good time - Template establishment phase",
                "Bench Boost": "🔴 Too early - Save for Double Gameweeks",
                "Triple Captain": "🟡 Possible - But better opportunities ahead",
                "Free Hit": "🔴 Too early - Save for Blank Gameweeks"
            }
        elif current_gw <= 20:
            phase = "Mid Season (GW 11-20)"
            recommendations = {
                "Wildcard": "🟡 Tactical use - Major template shifts",
                "Bench Boost": "🟢 Prime time - Double Gameweeks likely",
                "Triple Captain": "🟢 Good timing - Premium players hitting form",
                "Free Hit": "🟡 Selective use - Minor blank gameweeks"
            }
        else:
            phase = "Late Season (GW 21+)"
            recommendations = {
                "Wildcard": "🟢 Final chance - Prepare for run-in",
                "Bench Boost": "🟢 Must use - Final Double Gameweeks",
                "Triple Captain": "🟢 Final opportunities - Use or lose",
                "Free Hit": "🟢 Essential - Navigate blank gameweeks"
            }
        
        st.info(f"**Current Phase**: {phase}")
        
        for chip, rec in recommendations.items():
            st.write(f"**{chip}**: {rec}")

    def _render_chip_analysis(self, my_team_df):
        """Render detailed chip analysis"""
        st.write("**📊 Chip Readiness Analysis**")
        
        # Wildcard readiness
        underperformers = len(my_team_df[my_team_df['total_points'] < 30])
        wc_readiness = max(0, min(100, (15 - len(my_team_df)) * 6.67 + underperformers * 15))
        
        # Bench Boost readiness
        bench = my_team_df[my_team_df.get('pick_position', 0) > 11] if 'pick_position' in my_team_df.columns else my_team_df.tail(4)
        bench_points = bench['total_points'].sum() if not bench.empty else 0
        bb_readiness = min(100, (bench_points / 80) * 100)
        
        # Triple Captain readiness
        premium_players = len(my_team_df[my_team_df['now_cost'] >= 9.0])
        tc_readiness = min(100, premium_players * 25)
        
        # Free Hit readiness (always available)
        fh_readiness = 100
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Wildcard Readiness", f"{wc_readiness:.0f}%")
            if wc_readiness > 70:
                st.success("✅ Squad needs refresh")
            elif wc_readiness > 40:
                st.warning("🟡 Consider for major changes")
            else:
                st.success("🟢 Squad looks good")
        
        with col2:
            st.metric("Bench Boost Readiness", f"{bb_readiness:.0f}%")
            if bb_readiness > 60:
                st.success("✅ Bench is strong")
            else:
                st.warning("⚠️ Strengthen bench first")
        
        with col3:
            st.metric("Triple Captain Readiness", f"{tc_readiness:.0f}%")
            if tc_readiness > 50:
                st.success("✅ Have premium options")
            else:
                st.warning("⚠️ Need premium captain options")
        
        with col4:
            st.metric("Free Hit Readiness", f"{fh_readiness:.0f}%")
            st.success("✅ Always available")

    def _render_chip_strategic_planning(self, my_team_df):
        """Render chip strategic planning"""
        st.write("**🔮 Chip Strategy Scenarios**")
        
        scenarios = {
            "🔥 Aggressive Strategy": {
                "description": "Use chips early for immediate gains",
                "timeline": "Wildcard GW8 → Triple Captain GW12 → Bench Boost GW16 → Free Hit GW29",
                "risk": "High",
                "reward": "High short-term gains, rank climbing"
            },
            "⚖️ Balanced Strategy": {
                "description": "Strategic timing with flexibility",
                "timeline": "Wildcard GW15 → Bench Boost GW19 → Triple Captain GW25 → Free Hit GW33",
                "risk": "Medium",
                "reward": "Steady progression, adaptable"
            },
            "🛡️ Conservative Strategy": {
                "description": "Patient approach, optimal timing",
                "timeline": "Wildcard GW20 → Free Hit GW28 → Bench Boost GW34 → Triple Captain GW37",
                "risk": "Low",
                "reward": "Maximum efficiency, late surge"
            }
        }
        
        for strategy, details in scenarios.items():
            with st.expander(f"**{strategy}**", expanded=False):
                st.write(f"**Description**: {details['description']}")
                st.write(f"**Timeline**: {details['timeline']}")
                st.write(f"**Risk Level**: {details['risk']}")
                st.write(f"**Expected Reward**: {details['reward']}")

    def _render_chip_advanced_strategy(self, my_team_df):
        """Render advanced chip strategy"""
        st.write("**📈 Advanced Chip Coordination**")
        
        advanced_strategies = [
            "🔄 **Wildcard → Bench Boost Combo**: Use wildcard to build strong bench, then boost",
            "👑 **Triple Captain Timing**: Coordinate with Double Gameweeks and easy fixtures",
            "🎯 **Free Hit Mastery**: Use for blank gameweeks or differential team building",
            "📊 **Template vs Differential**: When to follow crowd vs take risks with chips"
        ]
        
        for strategy in advanced_strategies:
            st.write(strategy)
        
        st.write("**💡 Pro Tips:**")
        tips = [
            "Monitor template movements before using chips",
            "Consider using Free Hit for one-week punts",
            "Wildcard timing affects whole season trajectory",
            "Bench Boost works best with DGW + playing bench"
        ]
        
        for tip in tips:
            st.write(f"• {tip}")

    def _render_team_strengths(self, my_team_df):
        """Render team strengths analysis"""
        st.write("**💪 Team Strengths**")
        
        strengths = []
        
        # High value players
        if 'points_per_million' in my_team_df.columns:
            high_value_players = my_team_df[my_team_df['points_per_million'] > 8.0]
            if len(high_value_players) >= 8:
                strengths.append(f"✅ **Excellent Value**: {len(high_value_players)} players with PPM > 8.0")
        
        # Good form players
        if 'form' in my_team_df.columns:
            good_form = my_team_df[pd.to_numeric(my_team_df['form'], errors='coerce').fillna(0) > 5.0]
            if len(good_form) >= 6:
                strengths.append(f"✅ **Strong Form**: {len(good_form)} players in good form (>5.0)")
        
        # Squad value efficiency
        total_value = my_team_df['now_cost'].sum()
        if total_value < 98.0:
            budget_left = 100.0 - total_value
            strengths.append(f"✅ **Budget Flexibility**: £{budget_left:.1f}m remaining for upgrades")
        
        # Consistent performers
        consistent_players = my_team_df[my_team_df['total_points'] > 40]
        if len(consistent_players) >= 10:
            strengths.append(f"✅ **Consistency**: {len(consistent_players)} reliable performers (40+ points)")
        
        if strengths:
            for strength in strengths:
                st.write(strength)
        else:
            st.info("Focus on building team strengths through strategic transfers")

    def _render_team_weaknesses(self, my_team_df):
        """Render team weaknesses analysis"""
        st.write("**⚠️ Team Weaknesses**")
        
        weaknesses = []
        
        # Poor value players
        if 'points_per_million' in my_team_df.columns:
            poor_value = my_team_df[my_team_df['points_per_million'] < 5.0]
            if len(poor_value) > 0:
                weaknesses.append(f"⚠️ **Poor Value**: {len(poor_value)} players with PPM < 5.0")
        
        # Poor form players
        if 'form' in my_team_df.columns:
            poor_form = my_team_df[pd.to_numeric(my_team_df['form'], errors='coerce').fillna(0) < 3.0]
            if len(poor_form) > 0:
                weaknesses.append(f"⚠️ **Form Issues**: {len(poor_form)} players in poor form (<3.0)")
        
        # Underperformers
        underperformers = my_team_df[my_team_df['total_points'] < 20]
        if len(underperformers) > 2:
            weaknesses.append(f"⚠️ **Underperformers**: {len(underperformers)} players with <20 points")
        
        # Budget constraints
        total_value = my_team_df['now_cost'].sum()
        if total_value > 99.5:
            weaknesses.append("⚠️ **Budget Constraints**: Very tight budget limits transfer options")
        
        if weaknesses:
            for weakness in weaknesses:
                st.write(weakness)
        else:
            st.success("✅ No major weaknesses identified!")

    def _render_team_opportunities(self, my_team_df):
        """Render team opportunities analysis"""
        st.write("**🌟 Opportunities**")
        
        opportunities = [
            "📈 **Fixture Swings**: Monitor teams with improving fixtures",
            "💰 **Price Rises**: Target players likely to increase in value",
            "📊 **Template Gaps**: Identify highly-owned players you're missing",
            "🎯 **Form Players**: Jump on players hitting peak form",
            "💎 **Differentials**: Find low-owned players with high potential",
            "🔄 **Chip Timing**: Strategic chip usage for rank improvement",
            "⚡ **DGW Players**: Target Double Gameweek participants",
            "🏆 **Premium Moves**: Upgrade to proven premium performers"
        ]
        
        for opportunity in opportunities:
            st.write(opportunity)
        
        st.info("💡 Focus on 2-3 key opportunities to avoid over-tinkering")

    def _render_team_threats(self, my_team_df):
        """Render team threats analysis"""
        st.write("**⚡ Threats**")
        
        threats = []
        
        # Injury risks
        threats.append("🏥 **Injury Risks**: Monitor key players for injury concerns")
        
        # Rotation risks
        threats.append("🔄 **Rotation Risks**: Premium players may be rested in busy periods")
        
        # Price falls
        if 'points_per_million' in my_team_df.columns:
            poor_performers = my_team_df[my_team_df['points_per_million'] < 4.0]
            if len(poor_performers) > 0:
                threats.append(f"📉 **Price Fall Risk**: {len(poor_performers)} players at risk of price drops")
        
        # Template divergence
        threats.append("📊 **Template Divergence**: Falling behind popular picks")
        
        # Fixture difficulty
        threats.append("🔴 **Difficult Fixtures**: Players facing tough upcoming matches")
        
        # Chip timing
        threats.append("⏰ **Chip Timing**: Missing optimal chip usage windows")
        
        for threat in threats:
            st.write(threat)
        
        st.warning("⚠️ Stay vigilant and plan ahead to mitigate these risks")

    def _render_season_planning(self, my_team_df):
        """Season planning and goals"""
        st.write("**🎯 Season Goals & Planning**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Set Your Targets:**")
            target_rank = st.number_input("Target Overall Rank:", min_value=1, value=100000, step=1000)
            target_points = st.number_input("Target Total Points:", min_value=1000, value=2000, step=50)
            
            if st.button("💾 Save Targets"):
                st.session_state.target_rank = target_rank
                st.session_state.target_points = target_points
                st.success("✅ Targets saved!")
        
        with col2:
            st.write("**Current Progress:**")
            if 'target_rank' in st.session_state and 'target_points' in st.session_state:
                current_points = my_team_df['total_points'].sum()
                points_to_target = st.session_state.target_points - current_points
                
                st.metric("Target Rank", f"{st.session_state.target_rank:,}")
                st.metric("Points to Target", f"{st.session_state.target_points:,}", f"{points_to_target:+,}")

    def _render_weekly_focus(self, my_team_df):
        """Weekly focus and action items"""
        st.write("**🧠 This Week's Focus Areas**")
        
        focus_areas = [
            "👑 Optimize captain choice",
            "🏥 Monitor injury news", 
            "📈 Review player form trends",
            "🎯 Check fixture difficulty",
            "💰 Plan next transfer move",
            "📊 Analyze rival managers",
            "🔄 Consider chip usage timing"
        ]
        
        st.write("**Priority Actions:**")
        for i, area in enumerate(focus_areas, 1):
            checked = st.checkbox(area, key=f"focus_{i}")
            
        # Quick recommendations
        st.write("**💡 Quick Recommendations:**")
        recommendations = [
            "Review captain options based on fixtures",
            "Check for any injury updates before deadline",
            "Monitor player price changes",
            "Plan 2-3 gameweeks ahead for transfers"
        ]
        
        for rec in recommendations:
            st.write(f"• {rec}")

    # Helper method for loading user team (placeholder)
    def _load_user_team(self, team_id):
        """Load user team data (placeholder implementation)"""
        st.error("Team loading functionality needs to be implemented with FPL API integration")
        # This would integrate with the FPL API to load actual team data
        pass

