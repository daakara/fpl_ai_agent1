"""
Player Analysis Page - Handles all player-related analysis and filtering
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class PlayerAnalysisPage:
    """Handles player analysis functionality"""
    
    def __init__(self):
        pass
    
    def render(self):
        """Main render method for player analysis page"""
        st.header("📊 Advanced Player Analysis")
        
        # Comprehensive explanation section
        with st.expander("📚 Master Guide to Player Analysis", expanded=False):
            st.markdown("""
            **Advanced Player Analysis** is your data-driven approach to identifying the best FPL assets. This comprehensive tool evaluates players across multiple dimensions to help you make informed decisions.
            
            🎯 **Core Analysis Framework:**
            
            **1. Performance Metrics**
            - **Total Points**: Season accumulation showing overall contribution
            - **Points Per Game (PPG)**: True ability indicator regardless of games played
            - **Form**: Last 5 games momentum - crucial for current decisions
            - **Expected Points**: AI-driven predictions based on fixtures, form, and underlying stats
            
            **2. Value Analysis**
            - **Points Per Million (PPM)**: Budget efficiency - maximize returns per £spent
            - **Price Changes**: Track rising/falling assets for optimal timing
            - **Value Over Replacement**: How much better than cheapest viable option
            
            **3. Ownership & Differential Analysis**
            - **Template Players**: High ownership, essential for rank protection
            - **Differentials**: Low ownership gems for rank climbing
            - **Captaincy Data**: Popular captain choices and success rates
            """)
        
        if not st.session_state.get('data_loaded', False):
            st.info("Please load data first from the Dashboard.")
            return
        
        df = st.session_state.players_df
        
        if df.empty:
            st.warning("No player data available")
            return
        
        # Enhanced tab structure for comprehensive analysis
        st.subheader("Player Analysis Tools")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔍 Player Filtering & Overview",
            "📈 Performance Dashboards", 
            "⚖️ Player Comparison",
            "🎯 Positional Analysis",
            "💡 AI-Powered Insights"
        ])
        
        with tab1:
            self._render_enhanced_player_filters(df)
        
        with tab2:
            self._render_performance_metrics_dashboard(df)
            
        with tab3:
            self._render_player_comparison_tool(df)
            
        with tab4:
            self._render_position_specific_analysis(df)
            
        with tab5:
            self._render_ai_player_insights(df)
    
    def _render_enhanced_player_filters(self, df):
        """Enhanced filtering system with advanced metrics"""
        st.subheader("🔍 Smart Player Filters & Overview")
        
        # Enhanced filtering system
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Position filter
            if 'position_name' in df.columns:
                positions = st.multiselect("Position", df['position_name'].unique())
            else:
                positions = []
                st.warning("Position data not available")
        
        with col2:
            # Team filter
            if 'team_name' in df.columns:
                teams = st.multiselect("Team", df['team_name'].unique())
            else:
                teams = []
                st.warning("Team data not available")
        
        with col3:
            # Price range
            if 'cost_millions' in df.columns:
                min_price, max_price = st.slider(
                    "Price Range (£m)", 
                    float(df['cost_millions'].min()), 
                    float(df['cost_millions'].max()), 
                    (float(df['cost_millions'].min()), float(df['cost_millions'].max()))
                )
            else:
                min_price, max_price = 4.0, 15.0
        
        with col4:
            # Advanced metric filters
            st.write("**🎪 Advanced Filters:**")
            min_points = st.number_input("Min Total Points", 0, 300, 0, help="Minimum total points threshold")
            min_form = st.slider("Min Form", 0.0, 10.0, 0.0, 0.1, help="Minimum form rating")
        
        # Apply filters
        filtered_df = self._apply_filters(df, positions, teams, min_price, max_price, min_points, min_form)
        
        # Store filtered data in session state for other tabs
        st.session_state.filtered_players_df = filtered_df
        
        # Results overview
        st.write(f"📊 **{len(filtered_df)} players found** (filtered from {len(df)} total)")
        
        if not filtered_df.empty:
            # Display results
            self._display_filtered_results(filtered_df)
        else:
            st.warning("No players match your filters. Try adjusting the criteria.")
    
    def _apply_filters(self, df, positions, teams, min_price, max_price, min_points, min_form):
        """Apply all filters to the dataframe"""
        filtered_df = df.copy()
        
        # Standard filters
        if positions:
            filtered_df = filtered_df[filtered_df['position_name'].isin(positions)]
        if teams:
            filtered_df = filtered_df[filtered_df['team_name'].isin(teams)]
        if 'cost_millions' in df.columns:
            filtered_df = filtered_df[
                (filtered_df['cost_millions'] >= min_price) &
                (filtered_df['cost_millions'] <= max_price)
            ]
        
        # Advanced filters
        if 'total_points' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['total_points'] >= min_points]
        if 'form' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['form'] >= min_form]
        
        return filtered_df
    
    def _display_filtered_results(self, filtered_df):
        """Display the filtered results in a table"""
        # Get display columns
        display_cols = ['web_name', 'position_name', 'team_short_name', 'cost_millions', 'total_points']
        
        # Add additional columns if available
        optional_cols = ['form', 'points_per_million', 'selected_by_percent', 'minutes']
        for col in optional_cols:
            if col in filtered_df.columns:
                display_cols.append(col)
        
        # Sort options
        col1, col2 = st.columns(2)
        with col1:
            sort_by = st.selectbox("Sort by", display_cols, index=4 if 'total_points' in display_cols else 0)
        with col2:
            ascending = st.checkbox("Ascending order", value=False)
        
        # Sort and display
        sorted_df = filtered_df[display_cols].sort_values(sort_by, ascending=ascending)
        
        st.dataframe(
            sorted_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "web_name": "Player",
                "position_name": "Position", 
                "team_short_name": "Team",
                "cost_millions": st.column_config.NumberColumn("Price", format="£%.1f"),
                "total_points": "Points",
                "form": st.column_config.NumberColumn("Form", format="%.1f"),
                "points_per_million": st.column_config.NumberColumn("PPM", format="%.1f"),
                "selected_by_percent": st.column_config.NumberColumn("Own%", format="%.1f%%"),
                "minutes": "Minutes"
            }
        )
    
    def _render_performance_metrics_dashboard(self, df):
        """Enhanced performance metrics dashboard with interactive visualizations"""
        st.subheader("📈 Performance Metrics Dashboard")
        
        # Use filtered data if available
        display_df = st.session_state.get('filtered_players_df', df)
        
        if display_df.empty:
            st.warning("No players to analyze. Adjust your filters.")
            return
        
        # Performance metrics tabs
        metric_tab1, metric_tab2, metric_tab3, metric_tab4 = st.tabs([
            "⚽ Top Performers",
            "💰 Value Analysis",
            "📊 Form & Consistency",
            "📈 Interactive Charts"
        ])
        
        with metric_tab1:
            self._render_top_performers(display_df)
        
        with metric_tab2:
            self._render_value_analysis(display_df)
        
        with metric_tab3:
            self._render_form_analysis(display_df)
        
        with metric_tab4:
            self._render_interactive_charts(display_df)

    def _render_interactive_charts(self, df):
        """Render interactive Plotly visualizations with advanced colorbar configurations"""
        st.write("**📈 Interactive Performance Visualizations**")
        
        if df.empty:
            st.info("No data available for charts")
            return
        
        # Chart selection with new options
        chart_type = st.selectbox(
            "Select Chart Type",
            [
                "Performance Heatmap", 
                "Value vs Points Scatter", 
                "Form Trends", 
                "Position Comparison",
                "Advanced Team Heatmap",
                "Ownership vs Performance",
                "Player Risk Matrix"
            ]
        )
        
        if chart_type == "Performance Heatmap":
            self._render_performance_heatmap(df)
        elif chart_type == "Value vs Points Scatter":
            self._render_value_scatter(df)
        elif chart_type == "Form Trends":
            self._render_form_trends(df)
        elif chart_type == "Position Comparison":
            self._render_position_comparison_chart(df)
        elif chart_type == "Advanced Team Heatmap":
            self._render_advanced_team_heatmap(df)
        elif chart_type == "Ownership vs Performance":
            self._render_ownership_performance_chart(df)
        elif chart_type == "Player Risk Matrix":
            self._render_player_risk_matrix(df)

    def _render_advanced_team_heatmap(self, df):
        """Create advanced team-based performance heatmap with properly configured colorbar"""
        st.write("**🏟️ Team Performance Matrix**")
        
        required_cols = ['team_name', 'position_name', 'total_points']
        if not all(col in df.columns for col in required_cols):
            st.info("Required team/position data not available")
            return
        
        # Aggregate data by team and position
        team_position_stats = df.groupby(['team_name', 'position_name']).agg({
            'total_points': 'mean',
            'form': 'mean' if 'form' in df.columns else lambda x: 0,
            'points_per_million': 'mean' if 'points_per_million' in df.columns else lambda x: 0
        }).reset_index()
        
        # Pivot for heatmap
        heatmap_data = team_position_stats.pivot(
            index='team_name', 
            columns='position_name', 
            values='total_points'
        ).fillna(0)
        
        # Create heatmap with properly configured colorbar
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='Viridis',
            colorbar=dict(
                title=dict(text="Average Points"),
                tickmode="linear",
                tick0=0,
                dtick=20,
                thickness=20,
                len=0.8
            ),
            hovertemplate='<b>%{y}</b><br>Position: %{x}<br>Avg Points: %{z:.1f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Team Performance by Position - Average Points",
            xaxis_title="Position",
            yaxis_title="Team",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def _render_ownership_performance_chart(self, df):
        """Create ownership vs performance scatter with properly configured colorbar"""
        st.write("**👥 Ownership vs Performance Analysis**")
        
        required_cols = ['selected_by_percent', 'total_points', 'web_name', 'form']
        if not all(col in df.columns for col in required_cols):
            st.info("Required ownership/performance data not available")
            return
        
        # Create scatter plot with form as color dimension
        fig = go.Figure(data=go.Scatter(
            x=df['selected_by_percent'],
            y=df['total_points'],
            mode='markers',
            marker=dict(
                size=df.get('points_per_million', 8),
                color=df['form'],
                colorscale='RdYlBu_r',
                showscale=True,
                colorbar=dict(
                    title=dict(text="Form Rating"),
                    tickmode="linear",
                    tick0=0,
                    dtick=2,
                    thickness=15,
                    len=0.6
                ),
                line=dict(width=1, color='darkblue'),
                opacity=0.7
            ),
            text=df['web_name'],
            hovertemplate='<b>%{text}</b><br>' +
                         'Ownership: %{x:.1f}%<br>' +
                         'Points: %{y}<br>' +
                         'Form: %{marker.color:.1f}<extra></extra>'
        ))
        
        # Add quadrant lines
        avg_ownership = df['selected_by_percent'].mean()
        avg_points = df['total_points'].mean()
        
        fig.add_hline(y=avg_points, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=avg_ownership, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Add quadrant labels
        fig.add_annotation(x=avg_ownership/2, y=df['total_points'].max()*0.95, 
                          text="Hidden Gems", showarrow=False, font=dict(color="green", size=12))
        fig.add_annotation(x=df['selected_by_percent'].max()*0.8, y=df['total_points'].max()*0.95, 
                          text="Template Assets", showarrow=False, font=dict(color="blue", size=12))
        
        fig.update_layout(
            title="Ownership vs Performance Matrix (Size = Value, Color = Form)",
            xaxis_title="Ownership Percentage (%)",
            yaxis_title="Total Points",
            height=550,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def _render_player_risk_matrix(self, df):
        """Create player risk assessment matrix with properly configured colorbar"""
        st.write("**⚠️ Player Risk Assessment Matrix**")
        
        # Calculate risk factors
        risk_df = df.copy()
        
        # Risk score calculation
        risk_df['injury_risk'] = 0
        if 'news' in df.columns:
            risk_df['injury_risk'] = df['news'].str.contains('injured|doubt|miss', case=False, na=False).astype(int) * 5
        
        risk_df['form_risk'] = 0
        if 'form' in df.columns:
            risk_df['form_risk'] = (10 - df['form']).clip(0, 10)
        
        risk_df['minutes_risk'] = 0
        if 'minutes' in df.columns:
            risk_df['minutes_risk'] = ((2000 - df['minutes']) / 200).clip(0, 10)
        
        risk_df['price_risk'] = 0
        if 'cost_millions' in df.columns:
            risk_df['price_risk'] = (df['cost_millions'] / 2).clip(0, 10)
        
        # Combined risk score
        risk_df['total_risk'] = (
            risk_df['injury_risk'] * 0.4 +
            risk_df['form_risk'] * 0.3 +
            risk_df['minutes_risk'] * 0.2 +
            risk_df['price_risk'] * 0.1
        )
        
        # Create risk matrix with properly configured colorbar
        fig = go.Figure(data=go.Scatter(
            x=risk_df.get('cost_millions', risk_df.index),
            y=risk_df.get('total_points', 0),
            mode='markers',
            marker=dict(
                size=12,
                color=risk_df['total_risk'],
                colorscale=[[0, 'green'], [0.3, 'yellow'], [0.7, 'orange'], [1, 'red']],
                showscale=True,
                colorbar=dict(
                    title=dict(text="Risk Level"),
                    tickmode="array",
                    tickvals=[0, 2.5, 5, 7.5, 10],
                    ticktext=["Very Low", "Low", "Medium", "High", "Very High"],
                    thickness=18,
                    len=0.7
                ),
                line=dict(width=1, color='black'),
                opacity=0.8
            ),
            text=risk_df['web_name'],
            hovertemplate='<b>%{text}</b><br>' +
                         'Price: £%{x:.1f}m<br>' +
                         'Points: %{y}<br>' +
                         'Risk Score: %{marker.color:.1f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Player Risk Assessment Matrix",
            xaxis_title="Price (£m)",
            yaxis_title="Total Points",
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk insights
        high_risk_players = risk_df[risk_df['total_risk'] > 7]
        if not high_risk_players.empty:
            st.warning(f"⚠️ **High Risk Players Identified**: {len(high_risk_players)} players with risk score > 7")
            for _, player in high_risk_players.head(5).iterrows():
                st.write(f"• **{player['web_name']}** - Risk: {player['total_risk']:.1f}")
        
    def _render_performance_heatmap(self, df):
        """Create performance heatmap using Plotly"""
        st.write("**🔥 Player Performance Heatmap**")
        
        # Select relevant numeric columns for heatmap
        numeric_cols = ['total_points', 'form', 'points_per_million', 'selected_by_percent']
        available_cols = [col for col in numeric_cols if col in df.columns]
        
        if len(available_cols) < 2:
            st.info("Insufficient numeric data for heatmap")
            return
        
        # Prepare data for heatmap
        heatmap_df = df[['web_name'] + available_cols].head(20)  # Top 20 players
        
        # Normalize data for better visualization
        for col in available_cols:
            heatmap_df[col] = (heatmap_df[col] - heatmap_df[col].min()) / (heatmap_df[col].max() - heatmap_df[col].min())
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_df[available_cols].values,
            x=available_cols,
            y=heatmap_df['web_name'],
            colorscale='RdYlGn',
            colorbar=dict(
                title=dict(text="Performance Score"),
                tickmode="linear",
                tick0=0,
                dtick=0.2,
                thickness=15,
                len=0.7
            )
        ))
        
        fig.update_layout(
            title="Player Performance Heatmap (Normalized Metrics)",
            xaxis_title="Metrics",
            yaxis_title="Players",
            height=600,
            font=dict(size=10)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_value_scatter(self, df):
        """Create value vs points scatter plot"""
        st.write("**💎 Value vs Performance Scatter Plot**")
        
        required_cols = ['total_points', 'cost_millions', 'web_name', 'position_name']
        if not all(col in df.columns for col in required_cols):
            st.info("Required data not available for scatter plot")
            return
        
        # Create scatter plot
        fig = px.scatter(
            df,
            x='cost_millions',
            y='total_points',
            color='position_name',
            size='selected_by_percent' if 'selected_by_percent' in df.columns else None,
            hover_name='web_name',
            hover_data=['form', 'points_per_million'] if 'form' in df.columns else None,
            title="Player Value vs Performance Analysis",
            labels={
                'cost_millions': 'Price (£m)',
                'total_points': 'Total Points',
                'position_name': 'Position',
                'selected_by_percent': 'Ownership %'
            }
        )
        
        # Update layout with custom colorbar properties
        fig.update_layout(
            height=500,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )
        
        # Add trend line
        if len(df) > 5:
            fig.add_traces(px.scatter(df, x='cost_millions', y='total_points', trendline="ols").data[1:])
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_form_trends(self, df):
        """Create form trends visualization"""
        st.write("**📊 Form Analysis Distribution**")
        
        if 'form' not in df.columns:
            st.info("Form data not available")
            return
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Form Distribution', 'Form by Position', 'Form vs Points', 'Top Form Players'),
            specs=[[{"type": "histogram"}, {"type": "box"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Form distribution histogram
        fig.add_trace(
            go.Histogram(x=df['form'], name="Form Distribution", nbinsx=20),
            row=1, col=1
        )
        
        # Form by position box plot
        if 'position_name' in df.columns:
            for position in df['position_name'].unique():
                pos_data = df[df['position_name'] == position]['form']
                fig.add_trace(
                    go.Box(y=pos_data, name=position),
                    row=1, col=2
                )
        
        # Form vs Points scatter
        if 'total_points' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['form'], 
                    y=df['total_points'],
                    mode='markers',
                    name="Form vs Points",
                    text=df['web_name'],
                    hovertemplate='<b>%{text}</b><br>Form: %{x}<br>Points: %{y}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Top form players bar chart
        top_form = df.nlargest(10, 'form')
        fig.add_trace(
            go.Bar(
                x=top_form['web_name'], 
                y=top_form['form'],
                name="Top Form",
                marker_color='green'
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False, title_text="Comprehensive Form Analysis")
        fig.update_xaxes(title_text="Form", row=2, col=1)
        fig.update_yaxes(title_text="Points", row=2, col=1)
        fig.update_xaxes(title_text="Players", row=2, col=2)
        fig.update_yaxes(title_text="Form", row=2, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_position_comparison_chart(self, df):
        """Create position comparison radar chart"""
        st.write("**🎯 Position Performance Comparison**")
        
        if 'position_name' not in df.columns:
            st.info("Position data not available")
            return
        
        # Calculate average metrics by position
        metrics = ['total_points', 'form', 'points_per_million', 'selected_by_percent']
        available_metrics = [col for col in metrics if col in df.columns]
        
        if len(available_metrics) < 3:
            st.info("Insufficient metrics for position comparison")
            return
        
        position_stats = df.groupby('position_name')[available_metrics].mean().reset_index()
        
        # Create radar chart
        fig = go.Figure()
        
        for _, position in position_stats.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[position[metric] for metric in available_metrics],
                theta=available_metrics,
                fill='toself',
                name=position['position_name'],
                line=dict(width=2)
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max([position_stats[col].max() for col in available_metrics])]
                )),
            showlegend=True,
            title="Average Performance by Position",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_top_performers(self, df):
        """Render top performing players"""
        st.write("**🏆 Top Performers by Points**")
        
        if 'total_points' in df.columns:
            top_scorers = df.nlargest(10, 'total_points')[
                ['web_name', 'position_name', 'team_short_name', 'total_points', 'cost_millions']
            ]
            
            for i, (_, player) in enumerate(top_scorers.iterrows(), 1):
                st.write(f"{i}. **{player['web_name']}** ({player['position_name']}) - "
                        f"{player['total_points']} pts (£{player['cost_millions']:.1f}m)")
        else:
            st.info("Points data not available")
    
    def _render_value_analysis(self, df):
        """Render value analysis"""
        st.write("**💎 Best Value Players (Points per Million)**")
        
        if 'points_per_million' in df.columns:
            best_value = df.nlargest(10, 'points_per_million')[
                ['web_name', 'position_name', 'team_short_name', 'points_per_million', 'cost_millions']
            ]
            
            for i, (_, player) in enumerate(best_value.iterrows(), 1):
                st.write(f"{i}. **{player['web_name']}** ({player['position_name']}) - "
                        f"{player['points_per_million']:.1f} PPM (£{player['cost_millions']:.1f}m)")
        else:
            st.info("Value data not available")
    
    def _render_form_analysis(self, df):
        """Render form analysis"""
        st.write("**🔥 Best Form Players**")
        
        if 'form' in df.columns:
            hot_form = df.nlargest(10, 'form')[
                ['web_name', 'position_name', 'team_short_name', 'form', 'total_points']
            ]
            
            for i, (_, player) in enumerate(hot_form.iterrows(), 1):
                st.write(f"{i}. **{player['web_name']}** ({player['position_name']}) - "
                        f"Form: {player['form']:.1f} ({player['total_points']} pts)")
        else:
            st.info("Form data not available")
    
    def _render_player_comparison_tool(self, df):
        """Advanced player comparison tool"""
        st.subheader("⚖️ Advanced Player Comparison")
        
        # Use filtered data if available
        display_df = st.session_state.get('filtered_players_df', df)
        
        if display_df.empty:
            st.warning("No players to compare. Adjust your filters.")
            return
        
        st.write("**🔍 Select Players to Compare**")
        
        # Player selection
        available_players = display_df['web_name'].tolist() if 'web_name' in display_df.columns else []
        
        if len(available_players) < 2:
            st.info("Need at least 2 players in filtered results for comparison")
            return
        
        selected_players = st.multiselect(
            "Choose players to compare (max 4)",
            available_players,
            max_selections=4,
            help="Select 2-4 players for detailed comparison"
        )
        
        if len(selected_players) >= 2:
            # Filter for selected players
            comparison_df = display_df[display_df['web_name'].isin(selected_players)]
            
            # Comparison metrics
            comparison_cols = ['web_name', 'position_name', 'team_short_name', 'cost_millions']
            optional_cols = ['total_points', 'form', 'points_per_million', 'selected_by_percent', 'minutes']
            
            for col in optional_cols:
                if col in comparison_df.columns:
                    comparison_cols.append(col)
            
            st.write("**📊 Player Comparison Table**")
            st.dataframe(
                comparison_df[comparison_cols],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "web_name": "Player",
                    "position_name": "Position",
                    "team_short_name": "Team",
                    "cost_millions": st.column_config.NumberColumn("Price", format="£%.1f"),
                    "total_points": "Points",
                    "form": st.column_config.NumberColumn("Form", format="%.1f"),
                    "points_per_million": st.column_config.NumberColumn("PPM", format="%.1f"),
                    "selected_by_percent": st.column_config.NumberColumn("Own%", format="%.1f%%"),
                    "minutes": "Minutes"
                }
            )
            
            # Quick comparison insights
            if 'total_points' in comparison_df.columns:
                best_points = comparison_df.loc[comparison_df['total_points'].idxmax()]
                st.success(f"🏆 **Highest Points**: {best_points['web_name']} ({best_points['total_points']} pts)")
            
            if 'points_per_million' in comparison_df.columns:
                best_value = comparison_df.loc[comparison_df['points_per_million'].idxmax()]
                st.success(f"💎 **Best Value**: {best_value['web_name']} ({best_value['points_per_million']:.1f} PPM)")
            
            if 'form' in comparison_df.columns:
                best_form = comparison_df.loc[comparison_df['form'].idxmax()]
                st.success(f"🔥 **Best Form**: {best_form['web_name']} ({best_form['form']:.1f})")
        
        else:
            st.info("Please select at least 2 players to compare")
    
    def _render_position_specific_analysis(self, df):
        """Position-specific detailed analysis with comprehensive metrics"""
        st.subheader("🎯 Position-Specific Analysis")
        
        # Use filtered data if available
        display_df = st.session_state.get('filtered_players_df', df)
        
        if display_df.empty:
            st.warning("No players to analyze. Adjust your filters.")
            return
        
        # Position-specific tabs
        pos_tab1, pos_tab2, pos_tab3, pos_tab4 = st.tabs([
            "🥅 Goalkeepers",
            "🛡️ Defenders", 
            "⚽ Midfielders",
            "🎯 Forwards"
        ])
        
        with pos_tab1:
            self._render_goalkeeper_analysis(display_df)
        
        with pos_tab2:
            self._render_defender_analysis(display_df)
        
        with pos_tab3:
            self._render_midfielder_analysis(display_df)
        
        with pos_tab4:
            self._render_forward_analysis(display_df)
    
    def _render_goalkeeper_analysis(self, df):
        """Comprehensive goalkeeper analysis"""
        st.subheader("🥅 Goalkeeper Analysis")
        
        # Filter for goalkeepers
        if 'element_type' in df.columns:
            gk_df = df[df['element_type'] == 1].copy()
        elif 'position_name' in df.columns:
            gk_df = df[df['position_name'] == 'Goalkeeper'].copy()
        else:
            st.warning("Position data not available")
            return
        
        if gk_df.empty:
            st.info("No goalkeepers found in current filter selection")
            return
        
        # Goalkeeper-specific tabs
        gk_tab1, gk_tab2, gk_tab3, gk_tab4 = st.tabs([
            "📊 Performance",
            "🥅 GK-Specific Stats",
            "📈 Advanced Metrics", 
            "💎 Value Analysis"
        ])
        
        with gk_tab1:
            st.write("**🏆 Performance Metrics**")
            
            # Performance columns
            perf_cols = ['web_name', 'team_short_name', 'total_points', 'form']
            optional_perf = ['points_per_game', 'selected_by_percent', 'in_dreamteam', 'news']
            
            display_cols = [col for col in perf_cols + optional_perf if col in gk_df.columns]
            
            if display_cols:
                performance_df = gk_df[display_cols].sort_values('total_points', ascending=False)
                
                st.dataframe(
                    performance_df.head(10),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "web_name": "Goalkeeper",
                        "team_short_name": "Team", 
                        "total_points": "Points",
                        "form": st.column_config.NumberColumn("Form", format="%.1f"),
                        "points_per_game": st.column_config.NumberColumn("PPG", format="%.1f"),
                        "selected_by_percent": st.column_config.NumberColumn("Own%", format="%.1f%%"),
                        "in_dreamteam": "In Dream Team",
                        "news": "News"
                    }
                )
            else:
                st.info("Performance data not available")
        
        with gk_tab2:
            st.write("**🥅 Goalkeeper-Specific Statistics**")
            
            # GK-specific columns
            gk_cols = ['web_name', 'team_short_name']
            gk_stats = ['saves', 'penalties_saved', 'goals_conceded', 'clean_sheets']
            per_90_stats = ['saves_per_90', 'goals_conceded_per_90', 'clean_sheets_per_90', 'expected_goals_conceded_per_90']
            
            available_cols = [col for col in gk_cols + gk_stats + per_90_stats if col in gk_df.columns]
            
            if len(available_cols) > 2:  # More than just name and team
                gk_stats_df = gk_df[available_cols].sort_values('clean_sheets' if 'clean_sheets' in gk_df.columns else available_cols[0], ascending=False)
                
                st.dataframe(
                    gk_stats_df.head(10),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "web_name": "Goalkeeper",
                        "team_short_name": "Team",
                        "saves": "Saves",
                        "penalties_saved": "Pen Saved",
                        "goals_conceded": "Goals Conceded",
                        "clean_sheets": "Clean Sheets",
                        "saves_per_90": st.column_config.NumberColumn("Saves/90", format="%.1f"),
                        "goals_conceded_per_90": st.column_config.NumberColumn("Conceded/90", format="%.1f"),
                        "clean_sheets_per_90": st.column_config.NumberColumn("CS/90", format="%.2f"),
                        "expected_goals_conceded_per_90": st.column_config.NumberColumn("xGC/90", format="%.2f")
                    }
                )
            else:
                st.info("Goalkeeper-specific statistics not available")
        
        with gk_tab3:
            st.write("**📈 Advanced Goalkeeper Metrics**")
            
            # Advanced metrics
            adv_cols = ['web_name', 'team_short_name']
            adv_stats = ['expected_goals_conceded', 'bps', 'bonus', 'influence', 'creativity', 'threat']
            
            available_adv = [col for col in adv_cols + adv_stats if col in gk_df.columns]
            
            if len(available_adv) > 2:
                adv_df = gk_df[available_adv].sort_values('bps' if 'bps' in gk_df.columns else available_adv[0], ascending=False)
                
                st.dataframe(
                    adv_df.head(10),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "web_name": "Goalkeeper",
                        "team_short_name": "Team",
                        "expected_goals_conceded": st.column_config.NumberColumn("xGC", format="%.2f"),
                        "bps": "BPS",
                        "bonus": "Bonus",
                        "influence": st.column_config.NumberColumn("Influence", format="%.1f"),
                        "creativity": st.column_config.NumberColumn("Creativity", format="%.1f"),
                        "threat": st.column_config.NumberColumn("Threat", format="%.1f")
                    }
                )
            else:
                st.info("Advanced metrics not available")
        
        with gk_tab4:
            st.write("**💎 Goalkeeper Value Analysis**")
            
            # Value columns
            val_cols = ['web_name', 'team_short_name', 'cost_millions']
            val_stats = ['value_form', 'value_season', 'points_per_million', 'value_score', 'differential_score']
            
            available_val = [col for col in val_cols + val_stats if col in gk_df.columns]
            
            if len(available_val) > 3:
                val_df = gk_df[available_val].sort_values('points_per_million' if 'points_per_million' in gk_df.columns else available_val[0], ascending=False)
                
                st.dataframe(
                    val_df.head(10),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "web_name": "Goalkeeper",
                        "team_short_name": "Team",
                        "cost_millions": st.column_config.NumberColumn("Price", format="£%.1f"),
                        "value_form": st.column_config.NumberColumn("Value Form", format="%.2f"),
                        "value_season": st.column_config.NumberColumn("Value Season", format="%.2f"),
                        "points_per_million": st.column_config.NumberColumn("PPM", format="%.1f"),
                        "value_score": st.column_config.NumberColumn("Value Score", format="%.2f"),
                        "differential_score": st.column_config.NumberColumn("Diff Score", format="%.2f")
                    }
                )
            else:
                st.info("Value analysis data not available")
    
    def _render_defender_analysis(self, df):
        """Comprehensive defender analysis"""
        st.subheader("🛡️ Defender Analysis")
        
        # Filter for defenders
        if 'element_type' in df.columns:
            def_df = df[df['element_type'] == 2].copy()
        elif 'position_name' in df.columns:
            def_df = df[df['position_name'] == 'Defender'].copy()
        else:
            st.warning("Position data not available")
            return
        
        if def_df.empty:
            st.info("No defenders found in current filter selection")
            return
        
        # Defender-specific tabs
        def_tab1, def_tab2, def_tab3, def_tab4 = st.tabs([
            "🛡️ Defensive Stats",
            "⚽ Attack Contribution",
            "📈 Advanced Metrics",
            "💎 Value Analysis"
        ])
        
        with def_tab1:
            st.write("**🛡️ Defensive Statistics**")
            
            # Defensive columns
            def_cols = ['web_name', 'team_short_name']
            def_stats = ['clean_sheets', 'goals_conceded', 'clearances_blocks_interceptions', 'recoveries', 'tackles', 'defensive_contribution']
            per_90_stats = ['clean_sheets_per_90', 'defensive_contribution_per_90']
            
            available_def = [col for col in def_cols + def_stats + per_90_stats if col in def_df.columns]
            
            if len(available_def) > 2:
                def_stats_df = def_df[available_def].sort_values('clean_sheets' if 'clean_sheets' in def_df.columns else available_def[0], ascending=False)
                
                st.dataframe(
                    def_stats_df.head(10),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "web_name": "Defender",
                        "team_short_name": "Team",
                        "clean_sheets": "Clean Sheets",
                        "goals_conceded": "Goals Conceded",
                        "clearances_blocks_interceptions": "CBI",
                        "recoveries": "Recoveries",
                        "tackles": "Tackles",
                        "defensive_contribution": "Def Contribution",
                        "clean_sheets_per_90": st.column_config.NumberColumn("CS/90", format="%.2f"),
                        "defensive_contribution_per_90": st.column_config.NumberColumn("Def/90", format="%.2f")
                    }
                )
            else:
                st.info("Defensive statistics not available")
        
        with def_tab2:
            st.write("**⚽ Attack Contribution**")
            
            # Attack columns
            att_cols = ['web_name', 'team_short_name']
            att_stats = ['goals_scored', 'assists', 'expected_goals', 'expected_assists', 'expected_goal_involvements']
            
            available_att = [col for col in att_cols + att_stats if col in def_df.columns]
            
            if len(available_att) > 2:
                att_df = def_df[available_att].sort_values('expected_goal_involvements' if 'expected_goal_involvements' in def_df.columns else 'goals_scored' if 'goals_scored' in def_df.columns else available_att[0], ascending=False)
                
                st.dataframe(
                    att_df.head(10),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "web_name": "Defender",
                        "team_short_name": "Team",
                        "goals_scored": "Goals",
                        "assists": "Assists",
                        "expected_goals": st.column_config.NumberColumn("xG", format="%.2f"),
                        "expected_assists": st.column_config.NumberColumn("xA", format="%.2f"),
                        "expected_goal_involvements": st.column_config.NumberColumn("xGI", format="%.2f")
                    }
                )
            else:
                st.info("Attack contribution data not available")
        
        with def_tab3:
            st.write("**📈 Advanced Metrics**")
            
            # Advanced metrics
            adv_cols = ['web_name', 'team_short_name']
            adv_stats = ['bps', 'bonus', 'influence', 'creativity', 'threat']
            
            available_adv = [col for col in adv_cols + adv_stats if col in def_df.columns]
            
            if len(available_adv) > 2:
                adv_df = def_df[available_adv].sort_values('bps' if 'bps' in def_df.columns else available_adv[0], ascending=False)
                
                st.dataframe(
                    adv_df.head(10),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "web_name": "Defender",
                        "team_short_name": "Team",
                        "bps": "BPS",
                        "bonus": "Bonus",
                        "influence": st.column_config.NumberColumn("Influence", format="%.1f"),
                        "creativity": st.column_config.NumberColumn("Creativity", format="%.1f"),
                        "threat": st.column_config.NumberColumn("Threat", format="%.1f")
                    }
                )
            else:
                st.info("Advanced metrics not available")
        
        with def_tab4:
            st.write("**💎 Value Analysis**")
            self._render_position_value_analysis(def_df, "Defender")
    
    def _render_midfielder_analysis(self, df):
        """Comprehensive midfielder analysis"""
        st.subheader("⚽ Midfielder Analysis")
        
        # Filter for midfielders
        if 'element_type' in df.columns:
            mid_df = df[df['element_type'] == 3].copy()
        elif 'position_name' in df.columns:
            mid_df = df[df['position_name'] == 'Midfielder'].copy()
        else:
            st.warning("Position data not available")
            return
        
        if mid_df.empty:
            st.info("No midfielders found in current filter selection")
            return
        
        # Midfielder-specific tabs
        mid_tab1, mid_tab2, mid_tab3, mid_tab4 = st.tabs([
            "⚽ Attack Contribution",
            "🎨 Creativity & Threat",
            "🛡️ Defensive Support",
            "💎 Value Analysis"
        ])
        
        with mid_tab1:
            st.write("**⚽ Attack Contribution**")
            
            # Attack columns with per 90 stats
            att_cols = ['web_name', 'team_short_name']
            att_stats = ['goals_scored', 'assists', 'expected_goals', 'expected_assists', 'expected_goal_involvements']
            per_90_stats = ['expected_goals_per_90', 'expected_assists_per_90', 'expected_goal_involvements_per_90']
            
            available_att = [col for col in att_cols + att_stats + per_90_stats if col in mid_df.columns]
            
            if len(available_att) > 2:
                att_df = mid_df[available_att].sort_values('expected_goal_involvements' if 'expected_goal_involvements' in mid_df.columns else 'goals_scored' if 'goals_scored' in mid_df.columns else available_att[0], ascending=False)
                
                st.dataframe(
                    att_df.head(10),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "web_name": "Midfielder",
                        "team_short_name": "Team",
                        "goals_scored": "Goals",
                        "assists": "Assists",
                        "expected_goals": st.column_config.NumberColumn("xG", format="%.2f"),
                        "expected_assists": st.column_config.NumberColumn("xA", format="%.2f"),
                        "expected_goal_involvements": st.column_config.NumberColumn("xGI", format="%.2f"),
                        "expected_goals_per_90": st.column_config.NumberColumn("xG/90", format="%.2f"),
                        "expected_assists_per_90": st.column_config.NumberColumn("xA/90", format="%.2f"),
                        "expected_goal_involvements_per_90": st.column_config.NumberColumn("xGI/90", format="%.2f")
                    }
                )
            else:
                st.info("Attack contribution data not available")
        
        with mid_tab2:
            st.write("**🎨 Creativity & Threat**")
            
            # Creativity columns
            cre_cols = ['web_name', 'team_short_name']
            cre_stats = ['creativity', 'threat', 'influence']
            
            available_cre = [col for col in cre_cols + cre_stats if col in mid_df.columns]
            
            if len(available_cre) > 2:
                cre_df = mid_df[available_cre].sort_values('creativity' if 'creativity' in mid_df.columns else available_cre[0], ascending=False)
                
                st.dataframe(
                    cre_df.head(10),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "web_name": "Midfielder",
                        "team_short_name": "Team",
                        "creativity": st.column_config.NumberColumn("Creativity", format="%.1f"),
                        "threat": st.column_config.NumberColumn("Threat", format="%.1f"),
                        "influence": st.column_config.NumberColumn("Influence", format="%.1f")
                    }
                )
            else:
                st.info("Creativity & threat data not available")
        
        with mid_tab3:
            st.write("**🛡️ Defensive Support**")
            
            # Defensive columns
            def_cols = ['web_name', 'team_short_name']
            def_stats = ['recoveries', 'tackles']
            
            available_def = [col for col in def_cols + def_stats if col in mid_df.columns]
            
            if len(available_def) > 2:
                def_df = mid_df[available_def].sort_values('recoveries' if 'recoveries' in mid_df.columns else available_def[0], ascending=False)
                
                st.dataframe(
                    def_df.head(10),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "web_name": "Midfielder",
                        "team_short_name": "Team",
                        "recoveries": "Recoveries",
                        "tackles": "Tackles"
                    }
                )
            else:
                st.info("Defensive support data not available")
        
        with mid_tab4:
            st.write("**💎 Value Analysis**")
            self._render_position_value_analysis(mid_df, "Midfielder")
    
    def _render_forward_analysis(self, df):
        """Comprehensive forward analysis"""
        st.subheader("🎯 Forward Analysis")
        
        # Filter for forwards
        if 'element_type' in df.columns:
            fwd_df = df[df['element_type'] == 4].copy()
        elif 'position_name' in df.columns:
            fwd_df = df[df['position_name'] == 'Forward'].copy()
        else:
            st.warning("Position data not available")
            return
        
        if fwd_df.empty:
            st.info("No forwards found in current filter selection")
            return
        
        # Forward-specific tabs
        fwd_tab1, fwd_tab2, fwd_tab3 = st.tabs([
            "⚽ Attack Contribution",
            "🎯 Threat & Bonus",
            "💎 Value Analysis"
        ])
        
        with fwd_tab1:
            st.write("**⚽ Attack Contribution**")
            
            # Attack columns with per 90 stats
            att_cols = ['web_name', 'team_short_name']
            att_stats = ['goals_scored', 'assists', 'expected_goals', 'expected_assists', 'expected_goal_involvements']
            per_90_stats = ['expected_goals_per_90', 'expected_assists_per_90', 'expected_goal_involvements_per_90']
            
            available_att = [col for col in att_cols + att_stats + per_90_stats if col in fwd_df.columns]
            
            if len(available_att) > 2:
                att_df = fwd_df[available_att].sort_values('goals_scored' if 'goals_scored' in fwd_df.columns else available_att[0], ascending=False)
                
                st.dataframe(
                    att_df.head(10),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "web_name": "Forward",
                        "team_short_name": "Team",
                        "goals_scored": "Goals",
                        "assists": "Assists", 
                        "expected_goals": st.column_config.NumberColumn("xG", format="%.2f"),
                        "expected_assists": st.column_config.NumberColumn("xA", format="%.2f"),
                        "expected_goal_involvements": st.column_config.NumberColumn("xGI", format="%.2f"),
                        "expected_goals_per_90": st.column_config.NumberColumn("xG/90", format="%.2f"),
                        "expected_assists_per_90": st.column_config.NumberColumn("xA/90", format="%.2f"),
                        "expected_goal_involvements_per_90": st.column_config.NumberColumn("xGI/90", format="%.2f")
                    }
                )
            else:
                st.info("Attack contribution data not available")
        
        with fwd_tab2:
            st.write("**🎯 Threat & Bonus**")
            
            # Threat columns
            threat_cols = ['web_name', 'team_short_name']
            threat_stats = ['threat', 'bps', 'bonus']
            
            available_threat = [col for col in threat_cols + threat_stats if col in fwd_df.columns]
            
            if len(available_threat) > 2:
                threat_df = fwd_df[available_threat].sort_values('threat' if 'threat' in fwd_df.columns else available_threat[0], ascending=False)
                
                st.dataframe(
                    threat_df.head(10),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "web_name": "Forward",
                        "team_short_name": "Team",
                        "threat": st.column_config.NumberColumn("Threat", format="%.1f"),
                        "bps": "BPS",
                        "bonus": "Bonus"
                    }
                )
            else:
                st.info("Threat & bonus data not available")
        
        with fwd_tab3:
            st.write("**💎 Value Analysis**")
            self._render_position_value_analysis(fwd_df, "Forward")
    
    def _render_position_value_analysis(self, df, position_name):
        """Render value analysis for any position"""
        # Value columns
        val_cols = ['web_name', 'team_short_name', 'cost_millions']
        val_stats = ['value_form', 'value_season', 'points_per_million', 'value_score', 'differential_score']
        
        available_val = [col for col in val_cols + val_stats if col in df.columns]
        
        if len(available_val) > 3:
            val_df = df[available_val].sort_values('points_per_million' if 'points_per_million' in df.columns else available_val[0], ascending=False)
            
            st.dataframe(
                val_df.head(10),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "web_name": position_name,
                    "team_short_name": "Team",
                    "cost_millions": st.column_config.NumberColumn("Price", format="£%.1f"),
                    "value_form": st.column_config.NumberColumn("Value Form", format="%.2f"),
                    "value_season": st.column_config.NumberColumn("Value Season", format="%.2f"),
                    "points_per_million": st.column_config.NumberColumn("PPM", format="%.1f"),
                    "value_score": st.column_config.NumberColumn("Value Score", format="%.2f"),
                    "differential_score": st.column_config.NumberColumn("Diff Score", format="%.2f")
                }
            )
        else:
            st.info("Value analysis data not available")
    
    def _render_ai_player_insights(self, df):
        """AI-powered player insights and recommendations"""
        st.subheader("💡 AI Player Insights & Recommendations")
        
        # Use filtered data if available
        display_df = st.session_state.get('filtered_players_df', df)
        
        if display_df.empty:
            st.warning("No players to analyze. Adjust your filters.")
            return
        
        # AI Insights tabs
        ai_tab1, ai_tab2, ai_tab3, ai_tab4 = st.tabs([
            "🎯 Transfer Targets",
            "📈 Form Analysis", 
            "💎 Hidden Gems",
            "⚠️ Risk Assessment"
        ])
        
        with ai_tab1:
            st.write("**🎯 AI Transfer Target Recommendations**")
            
            # Calculate AI transfer score
            if len(display_df) > 0:
                ai_df = display_df.copy()
                
                # Create AI scoring system
                ai_df['ai_transfer_score'] = 0
                
                # Form component (30%)
                if 'form' in ai_df.columns:
                    ai_df['form_score'] = ai_df['form'] / 10
                    ai_df['ai_transfer_score'] += ai_df['form_score'] * 0.3
                
                # Points per million component (25%)
                if 'points_per_million' in ai_df.columns:
                    ai_df['value_score'] = ai_df['points_per_million'] / ai_df['points_per_million'].max()
                    ai_df['ai_transfer_score'] += ai_df['value_score'] * 0.25
                
                # Total points component (20%)
                if 'total_points' in ai_df.columns:
                    ai_df['points_score'] = ai_df['total_points'] / ai_df['total_points'].max()
                    ai_df['ai_transfer_score'] += ai_df['points_score'] * 0.2
                
                # Ownership differential (15%)
                if 'selected_by_percent' in ai_df.columns:
                    ai_df['diff_score'] = (100 - ai_df['selected_by_percent']) / 100
                    ai_df['ai_transfer_score'] += ai_df['diff_score'] * 0.15
                
                # Minutes played reliability (10%)
                if 'minutes' in ai_df.columns:
                    ai_df['minutes_score'] = ai_df['minutes'] / ai_df['minutes'].max()
                    ai_df['ai_transfer_score'] += ai_df['minutes_score'] * 0.1
                
                # Get top AI recommendations
                top_ai_targets = ai_df.nlargest(10, 'ai_transfer_score')
                
                for i, (_, player) in enumerate(top_ai_targets.iterrows(), 1):
                    with st.expander(f"{i}. {player['web_name']} - AI Score: {player['ai_transfer_score']:.3f}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Position:** {player.get('position_name', 'N/A')}")
                            st.write(f"**Team:** {player.get('team_short_name', 'N/A')}")
                            st.write(f"**Price:** £{player.get('cost_millions', 0):.1f}m")
                            st.write(f"**Total Points:** {player.get('total_points', 0)}")
                        
                        with col2:
                            st.write(f"**Form:** {player.get('form', 0):.1f}")
                            st.write(f"**PPM:** {player.get('points_per_million', 0):.1f}")
                            st.write(f"**Ownership:** {player.get('selected_by_percent', 0):.1f}%")
                            st.write(f"**Minutes:** {player.get('minutes', 0)}")
                        
                        # AI reasoning
                        reasons = []
                        if player.get('form', 0) > 7:
                            reasons.append("🔥 Excellent recent form")
                        if player.get('points_per_million', 0) > 8:
                            reasons.append("💰 Great value for money")
                        if player.get('selected_by_percent', 0) < 15:
                            reasons.append("💎 Low ownership differential")
                        if player.get('minutes', 0) > 1500:
                            reasons.append("🛡️ High playing time reliability")
                        
                        if reasons:
                            st.write("**AI Reasoning:**")
                            for reason in reasons:
                                st.write(f"• {reason}")
            else:
                st.info("No player data available for AI analysis")
        
        with ai_tab2:
            st.write("**📈 AI Form Analysis**")
            
            if 'form' in display_df.columns:
                # Form trends
                hot_form = display_df[display_df['form'] >= 7].sort_values('form', ascending=False)
                cold_form = display_df[display_df['form'] <= 3].sort_values('form', ascending=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**🔥 Hot Form Players (7.0+)**")
                    if not hot_form.empty:
                        for _, player in hot_form.head(5).iterrows():
                            st.write(f"• **{player['web_name']}** - Form: {player['form']:.1f}")
                    else:
                        st.info("No players with exceptional form found")
                
                with col2:
                    st.write("**❄️ Cold Form Players (3.0-)**")
                    if not cold_form.empty:
                        for _, player in cold_form.head(5).iterrows():
                            st.write(f"• **{player['web_name']}** - Form: {player['form']:.1f}")
                    else:
                        st.info("No players with poor form found")
            else:
                st.info("Form data not available")
        
        with ai_tab3:
            st.write("**💎 Hidden Gems Discovery**")
            
            # Find hidden gems: low ownership, good performance
            if 'selected_by_percent' in display_df.columns and 'total_points' in display_df.columns:
                hidden_gems = display_df[
                    (display_df['selected_by_percent'] < 10) &
                    (display_df['total_points'] > 30)
                ].copy()
                
                if not hidden_gems.empty:
                    # Calculate gem score
                    hidden_gems['gem_score'] = (
                        (hidden_gems.get('points_per_million', 0) / hidden_gems.get('points_per_million', 1).max()) * 0.4 +
                        (hidden_gems.get('form', 0) / 10) * 0.3 +
                        ((10 - hidden_gems['selected_by_percent']) / 10) * 0.3
                    )
                    
                    top_gems = hidden_gems.nlargest(5, 'gem_score')
                    
                    for i, (_, player) in enumerate(top_gems.iterrows(), 1):
                        st.write(f"**{i}. {player['web_name']}** ({player.get('position_name', 'N/A')})")
                        st.write(f"💎 Gem Score: {player['gem_score']:.3f} | "
                                f"👥 {player['selected_by_percent']:.1f}% owned | "
                                f"📊 {player['total_points']} pts")
                        st.divider()
                else:
                    st.info("No hidden gems found with current filters")
            else:
                st.info("Insufficient data for hidden gem analysis")
        
        with ai_tab4:
            st.write("**⚠️ AI Risk Assessment**")
            
            # Risk factors analysis
            risk_factors = []
            
            if 'news' in display_df.columns:
                injured_players = display_df[
                    display_df['news'].str.contains('injured|doubt|miss', case=False, na=False)
                ]
                if not injured_players.empty:
                    risk_factors.append(f"🏥 {len(injured_players)} players with injury concerns")
            
            if 'form' in display_df.columns:
                poor_form = display_df[display_df['form'] < 3]
                if not poor_form.empty:
                    risk_factors.append(f"📉 {len(poor_form)} players in very poor form")
            
            if 'minutes' in display_df.columns:
                low_minutes = display_df[display_df['minutes'] < 500]
                if not low_minutes.empty:
                    risk_factors.append(f"⏱️ {len(low_minutes)} players with limited game time")
            
            if risk_factors:
                st.write("**⚠️ Current Risk Factors:**")
                for factor in risk_factors:
                    st.write(f"• {factor}")
            else:
                st.success("✅ No major risk factors detected in current player selection")
            
            # High ownership template warnings
            if 'selected_by_percent' in display_df.columns:
                template_players = display_df[display_df['selected_by_percent'] > 50]
                if not template_players.empty:
                    st.write("**👥 Template Player Warnings:**")
                    st.info(f"You have {len(template_players)} highly owned players (50%+). "
                           f"Consider differentials for rank climbing.")

