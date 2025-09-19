"""
Visualization Services for FPL Analytics Application
Following Single Responsibility Principle - each service handles specific visualization type
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
from abc import ABC, abstractmethod

from models.domain_models import Player, Team, FDRData, Position


class IVisualizationService(ABC):
    """Base interface for visualization services"""
    
    @abstractmethod
    def create_chart(self, data: Any, **kwargs) -> go.Figure:
        """Create chart from data"""
        pass


class FDRVisualizationService(IVisualizationService):
    """Service for creating FDR-related visualizations"""
    
    def __init__(self):
        self.fdr_colors = {
            1: '#00FF87',  # Very Easy
            2: '#01FF70',  # Easy
            3: '#FFDC00',  # Average
            4: '#FF851B',  # Hard
            5: '#FF4136'   # Very Hard
        }
        
        self.fdr_labels = {
            1: 'Very Easy',
            2: 'Easy', 
            3: 'Average',
            4: 'Hard',
            5: 'Very Hard'
        }
    
    def create_chart(self, data: List[FDRData], chart_type: str = 'heatmap', **kwargs) -> go.Figure:
        """Create FDR visualization"""
        if chart_type == 'heatmap':
            return self._create_heatmap(data, **kwargs)
        elif chart_type == 'bar':
            return self._create_bar_chart(data, **kwargs)
        elif chart_type == 'scatter':
            return self._create_scatter_plot(data, **kwargs)
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")
    
    def _create_heatmap(self, fdr_data: List[FDRData], fdr_type: str = 'combined', **kwargs) -> go.Figure:
        """Create FDR heatmap"""
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'team': fdr.team_short_name,
                'fixture': fdr.fixture_number,
                'opponent': fdr.opponent_short_name,
                'fdr': getattr(fdr, f'{fdr_type}_fdr', fdr.combined_fdr),
                'home_away': 'H' if fdr.is_home else 'A'
            }
            for fdr in fdr_data
        ])
        
        # Create pivot table
        pivot_data = df.pivot_table(
            index='team', 
            columns='fixture', 
            values='fdr', 
            aggfunc='first'
        )
        
        # Create hover text
        hover_df = df.pivot_table(
            index='team',
            columns='fixture', 
            values='opponent',
            aggfunc='first'
        )
        
        home_away_df = df.pivot_table(
            index='team',
            columns='fixture',
            values='home_away',
            aggfunc='first'
        )
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=[f'GW{i}' for i in pivot_data.columns],
            y=pivot_data.index,
            text=[[f"{opponent} ({ha})" 
                   for opponent, ha in zip(hover_row, ha_row)]
                  for hover_row, ha_row in zip(hover_df.values, home_away_df.values)],
            colorscale=[[i/4, color] for i, color in enumerate(self.fdr_colors.values())],
            zmin=1, zmax=5,
            hovertemplate='<b>%{y}</b><br>%{x}<br>FDR: %{z}<br>vs %{text}<extra></extra>',
            colorbar=dict(
                title="FDR",
                tickvals=list(self.fdr_colors.keys()),
                ticktext=list(self.fdr_labels.values())
            )
        ))
        
        fig.update_layout(
            title=f'{fdr_type.title()} FDR Heatmap',
            xaxis_title="Gameweek",
            yaxis_title="Team",
            height=600,
            font=dict(size=10)
        )
        
        return fig
    
    def _create_bar_chart(self, fdr_data: List[FDRData], **kwargs) -> go.Figure:
        """Create FDR bar chart"""
        # Aggregate by team
        team_fdr = {}
        for fdr in fdr_data:
            if fdr.team_short_name not in team_fdr:
                team_fdr[fdr.team_short_name] = []
            team_fdr[fdr.team_short_name].append(fdr.combined_fdr)
        
        teams = list(team_fdr.keys())
        avg_fdr = [np.mean(fdrs) for fdrs in team_fdr.values()]
        
        # Sort by FDR
        sorted_data = sorted(zip(teams, avg_fdr), key=lambda x: x[1])
        teams, avg_fdr = zip(*sorted_data)
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(teams),
                y=list(avg_fdr),
                marker=dict(
                    color=list(avg_fdr),
                    colorscale=[[0, self.fdr_colors[1]], [1, self.fdr_colors[5]]],
                    cmin=1, cmax=5
                ),
                text=[f"{fdr:.2f}" for fdr in avg_fdr],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Average FDR by Team",
            xaxis_title="Team",
            yaxis_title="Average FDR",
            height=500
        )
        
        return fig
    
    def _create_scatter_plot(self, fdr_data: List[FDRData], **kwargs) -> go.Figure:
        """Create FDR scatter plot (attack vs defense)"""
        df = pd.DataFrame([
            {
                'team': fdr.team_short_name,
                'attack_fdr': fdr.attack_fdr,
                'defense_fdr': fdr.defense_fdr,
                'combined_fdr': fdr.combined_fdr
            }
            for fdr in fdr_data
        ])
        
        # Aggregate by team
        team_avg = df.groupby('team').agg({
            'attack_fdr': 'mean',
            'defense_fdr': 'mean',
            'combined_fdr': 'mean'
        }).reset_index()
        
        fig = go.Figure(data=go.Scatter(
            x=team_avg['attack_fdr'],
            y=team_avg['defense_fdr'],
            mode='markers+text',
            text=team_avg['team'],
            textposition="top center",
            marker=dict(
                size=12,
                color=team_avg['combined_fdr'],
                colorscale=[[0, self.fdr_colors[1]], [1, self.fdr_colors[5]]],
                cmin=1, cmax=5,
                showscale=True,
                colorbar=dict(title="Combined FDR")
            ),
            hovertemplate='<b>%{text}</b><br>Attack FDR: %{x}<br>Defense FDR: %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Attack vs Defense FDR",
            xaxis_title="Attack FDR (Higher = Harder to score against)",
            yaxis_title="Defense FDR (Higher = Harder to keep clean sheet)",
            height=500
        )
        
        return fig


class PlayerVisualizationService(IVisualizationService):
    """Service for creating player-related visualizations"""
    
    def create_chart(self, data: List[Player], chart_type: str = 'scatter', **kwargs) -> go.Figure:
        """Create player visualization"""
        if chart_type == 'scatter':
            return self._create_scatter_plot(data, **kwargs)
        elif chart_type == 'bar':
            return self._create_bar_chart(data, **kwargs)
        elif chart_type == 'position_analysis':
            return self._create_position_analysis(data, **kwargs)
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")
    
    def _create_scatter_plot(self, players: List[Player], x_metric: str = 'cost_millions', 
                           y_metric: str = 'total_points', **kwargs) -> go.Figure:
        """Create player scatter plot"""
        df = pd.DataFrame([
            {
                'name': player.web_name,
                'team': player.team_short_name,
                'position': player.position_name,
                'cost_millions': player.cost_millions,
                'total_points': player.total_points,
                'points_per_game': player.points_per_game,
                'form': player.form,
                'selected_by_percent': player.selected_by_percent,
                'points_per_million': player.points_per_million
            }
            for player in players
        ])
        
        position_colors = {
            'GK': '#FF6B6B',
            'DEF': '#4ECDC4', 
            'MID': '#45B7D1',
            'FWD': '#96CEB4'
        }
        
        fig = go.Figure()
        
        for position in df['position'].unique():
            pos_data = df[df['position'] == position]
            
            fig.add_trace(go.Scatter(
                x=pos_data[x_metric],
                y=pos_data[y_metric],
                mode='markers',
                name=position,
                marker=dict(
                    size=8,
                    color=position_colors.get(position, '#666666'),
                    opacity=0.7
                ),
                text=pos_data['name'],
                hovertemplate='<b>%{text}</b><br>' +
                             f'{x_metric}: %{{x}}<br>' +
                             f'{y_metric}: %{{y}}<br>' +
                             'Team: ' + pos_data['team'] + '<extra></extra>'
            ))
        
        fig.update_layout(
            title=f"{y_metric.replace('_', ' ').title()} vs {x_metric.replace('_', ' ').title()}",
            xaxis_title=x_metric.replace('_', ' ').title(),
            yaxis_title=y_metric.replace('_', ' ').title(),
            height=500
        )
        
        return fig
    
    def _create_bar_chart(self, players: List[Player], metric: str = 'total_points', 
                         top_n: int = 20, **kwargs) -> go.Figure:
        """Create top players bar chart"""
        df = pd.DataFrame([
            {
                'name': player.web_name,
                'team': player.team_short_name,
                'position': player.position_name,
                'metric_value': getattr(player, metric, 0)
            }
            for player in players
        ])
        
        # Sort and take top N
        df_sorted = df.nlargest(top_n, 'metric_value')
        
        position_colors = {
            'GK': '#FF6B6B',
            'DEF': '#4ECDC4',
            'MID': '#45B7D1', 
            'FWD': '#96CEB4'
        }
        
        colors = [position_colors.get(pos, '#666666') for pos in df_sorted['position']]
        
        fig = go.Figure(data=[
            go.Bar(
                x=df_sorted['name'],
                y=df_sorted['metric_value'],
                marker_color=colors,
                text=df_sorted['metric_value'],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>' +
                             f'{metric.replace("_", " ").title()}: %{{y}}<br>' +
                             'Team: ' + df_sorted['team'] + '<br>' +
                             'Position: ' + df_sorted['position'] + '<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title=f"Top {top_n} Players by {metric.replace('_', ' ').title()}",
            xaxis_title="Player",
            yaxis_title=metric.replace('_', ' ').title(),
            height=500,
            xaxis_tickangle=-45
        )
        
        return fig
    
    def _create_position_analysis(self, players: List[Player], **kwargs) -> go.Figure:
        """Create position-based analysis chart"""
        df = pd.DataFrame([
            {
                'position': player.position_name,
                'cost_millions': player.cost_millions,
                'total_points': player.total_points,
                'points_per_million': player.points_per_million
            }
            for player in players
        ])
        
        position_stats = df.groupby('position').agg({
            'cost_millions': 'mean',
            'total_points': 'mean', 
            'points_per_million': 'mean'
        }).reset_index()
        
        fig = go.Figure(data=[
            go.Bar(
                name='Avg Cost (£m)',
                x=position_stats['position'],
                y=position_stats['cost_millions'],
                yaxis='y',
                marker_color='lightblue'
            ),
            go.Bar(
                name='Avg Points',
                x=position_stats['position'],
                y=position_stats['total_points'],
                yaxis='y2',
                marker_color='orange'
            )
        ])
        
        fig.update_layout(
            title="Position Analysis - Cost vs Points",
            xaxis_title="Position",
            yaxis=dict(title="Average Cost (£m)", side="left"),
            yaxis2=dict(title="Average Points", side="right", overlaying="y"),
            height=500,
            barmode='group'
        )
        
        return fig


class TeamComparisonService(IVisualizationService):
    """Service for creating team comparison visualizations"""
    
    def create_chart(self, data: List[Team], chart_type: str = 'radar', **kwargs) -> go.Figure:
        """Create team comparison chart"""
        if chart_type == 'radar':
            return self._create_radar_chart(data, **kwargs)
        elif chart_type == 'bar_comparison':
            return self._create_bar_comparison(data, **kwargs)
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")
    
    def _create_radar_chart(self, teams: List[Team], metrics: List[str] = None, **kwargs) -> go.Figure:
        """Create team comparison radar chart"""
        if metrics is None:
            metrics = ['strength_overall_home', 'strength_overall_away', 
                      'strength_attack_home', 'strength_attack_away',
                      'strength_defence_home', 'strength_defence_away']
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        
        for i, team in enumerate(teams[:6]):  # Limit to 6 teams for readability
            values = []
            for metric in metrics:
                value = getattr(team, metric, 0)
                values.append(value)
            
            # Close the radar chart
            values.append(values[0])
            metric_labels = [m.replace('_', ' ').title() for m in metrics]
            metric_labels.append(metric_labels[0])
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metric_labels,
                fill='toself',
                name=team.short_name,
                line_color=colors[i % len(colors)]
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 6]  # Typical FPL strength range
                )
            ),
            title="Team Strength Comparison",
            height=600
        )
        
        return fig
    
    def _create_bar_comparison(self, teams: List[Team], metric: str = 'strength_overall_home', **kwargs) -> go.Figure:
        """Create team comparison bar chart"""
        df = pd.DataFrame([
            {
                'team': team.short_name,
                'value': getattr(team, metric, 0)
            }
            for team in teams
        ])
        
        df_sorted = df.sort_values('value', ascending=False)
        
        fig = go.Figure(data=[
            go.Bar(
                x=df_sorted['team'],
                y=df_sorted['value'],
                marker_color='skyblue',
                text=df_sorted['value'].round(1),
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title=f"Team Comparison - {metric.replace('_', ' ').title()}",
            xaxis_title="Team",
            yaxis_title=metric.replace('_', ' ').title(),
            height=500,
            xaxis_tickangle=-45
        )
        
        return fig


class StatisticalAnalysisService:
    """Service for statistical analysis and distributions"""
    
    def create_distribution_chart(self, data: List[float], title: str = "Distribution", 
                                bins: int = 20) -> go.Figure:
        """Create distribution histogram"""
        fig = go.Figure(data=[
            go.Histogram(
                x=data,
                nbinsx=bins,
                marker_color='lightblue',
                opacity=0.7
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Value",
            yaxis_title="Frequency",
            height=400
        )
        
        return fig
    
    def create_correlation_heatmap(self, df: pd.DataFrame, columns: List[str] = None) -> go.Figure:
        """Create correlation heatmap"""
        if columns:
            corr_df = df[columns].corr()
        else:
            corr_df = df.select_dtypes(include=[np.number]).corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_df.values,
            x=corr_df.columns,
            y=corr_df.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_df.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate='<b>%{x} vs %{y}</b><br>Correlation: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Correlation Heatmap",
            height=500,
            width=500
        )
        
        return fig
