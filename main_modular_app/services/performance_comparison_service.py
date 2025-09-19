"""
Performance Comparison Service - Compare team performance against benchmarks
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import streamlit as st


class PerformanceComparisonService:
    """Service for comparing team performance against various benchmarks"""
    
    def __init__(self):
        # Benchmark data (would be fetched from APIs in production)
        self.top_10k_averages = {
            'total_points': 1800,
            'gw_average': 65,
            'captain_success_rate': 0.75,
            'transfer_success_rate': 0.68,
            'value_efficiency': 7.2
        }
        
        self.overall_averages = {
            'total_points': 1400,
            'gw_average': 52,
            'captain_success_rate': 0.55,
            'transfer_success_rate': 0.45,
            'value_efficiency': 5.8
        }
        
        # Historical season patterns
        self.historical_patterns = {
            'early_season': {'gw_range': (1, 10), 'avg_points': 55},
            'mid_season': {'gw_range': (11, 25), 'avg_points': 58},
            'late_season': {'gw_range': (26, 38), 'avg_points': 62}
        }
    
    def generate_performance_comparison(self, team_data: Dict, players_df: pd.DataFrame) -> Dict:
        """Generate comprehensive performance comparison analysis"""
        
        current_gw = team_data.get('gameweek', 20)
        total_points = team_data.get('summary_overall_points', 0)
        overall_rank = team_data.get('summary_overall_rank', 0)
        gw_points = team_data.get('summary_event_points', 0)
        
        # Calculate key metrics
        avg_ppg = total_points / max(current_gw, 1)
        percentile = self._calculate_percentile(overall_rank) if overall_rank else 50
        
        # Compare against benchmarks
        top_10k_comparison = self._compare_against_benchmark(
            team_data, self.top_10k_averages, "Top 10K"
        )
        
        overall_comparison = self._compare_against_benchmark(
            team_data, self.overall_averages, "Overall Average"
        )
        
        # Historical performance analysis
        historical_analysis = self._analyze_historical_performance(team_data, current_gw)
        
        # Squad comparison
        squad_comparison = self._compare_squad_composition(team_data, players_df)
        
        # Performance trends
        trend_analysis = self._analyze_performance_trends(team_data, current_gw)
        
        return {
            'current_metrics': {
                'total_points': total_points,
                'avg_ppg': avg_ppg,
                'percentile': percentile,
                'current_gw': current_gw
            },
            'top_10k_comparison': top_10k_comparison,
            'overall_comparison': overall_comparison,
            'historical_analysis': historical_analysis,
            'squad_comparison': squad_comparison,
            'trend_analysis': trend_analysis,
            'recommendations': self._generate_comparison_recommendations(
                top_10k_comparison, overall_comparison, trend_analysis
            )
        }
    
    def _calculate_percentile(self, rank: int, total_players: int = 8000000) -> float:
        """Calculate percentile based on rank"""
        return max(0, (1 - (rank / total_players)) * 100)
    
    def _compare_against_benchmark(self, team_data: Dict, benchmark: Dict, benchmark_name: str) -> Dict:
        """Compare team performance against a specific benchmark"""
        current_gw = team_data.get('gameweek', 20)
        total_points = team_data.get('summary_overall_points', 0)
        avg_ppg = total_points / max(current_gw, 1)
        
        comparison = {
            'benchmark_name': benchmark_name,
            'metrics': {}
        }
        
        # Points comparison
        points_diff = total_points - (benchmark['total_points'] * current_gw / 38)
        comparison['metrics']['points'] = {
            'your_score': total_points,
            'benchmark': benchmark['total_points'] * current_gw / 38,
            'difference': points_diff,
            'performance': 'above' if points_diff > 0 else 'below'
        }
        
        # Average points per gameweek
        ppg_diff = avg_ppg - benchmark['gw_average']
        comparison['metrics']['ppg'] = {
            'your_score': avg_ppg,
            'benchmark': benchmark['gw_average'],
            'difference': ppg_diff,
            'performance': 'above' if ppg_diff > 0 else 'below'
        }
        
        # Overall performance rating
        total_metrics = len(comparison['metrics'])
        above_benchmark = sum(1 for m in comparison['metrics'].values() if m['performance'] == 'above')
        
        comparison['overall_rating'] = {
            'above_benchmark_count': above_benchmark,
            'total_metrics': total_metrics,
            'percentage': (above_benchmark / total_metrics) * 100 if total_metrics > 0 else 0
        }
        
        return comparison
    
    def _analyze_historical_performance(self, team_data: Dict, current_gw: int) -> Dict:
        """Analyze performance against historical patterns"""
        total_points = team_data.get('summary_overall_points', 0)
        avg_ppg = total_points / max(current_gw, 1)
        
        # Determine current season phase
        if current_gw <= 10:
            phase = 'early_season'
        elif current_gw <= 25:
            phase = 'mid_season'
        else:
            phase = 'late_season'
        
        expected_avg = self.historical_patterns[phase]['avg_points']
        performance_vs_historical = avg_ppg - expected_avg
        
        return {
            'current_phase': phase.replace('_', ' ').title(),
            'your_avg_ppg': avg_ppg,
            'historical_avg': expected_avg,
            'difference': performance_vs_historical,
            'performance': 'above' if performance_vs_historical > 0 else 'below',
            'phase_analysis': self._get_phase_analysis(phase, performance_vs_historical)
        }
    
    def _compare_squad_composition(self, team_data: Dict, players_df: pd.DataFrame) -> Dict:
        """Compare squad composition against typical top performers"""
        picks = team_data.get('picks', [])
        
        if not picks or players_df.empty:
            return {'error': 'No squad data available'}
        
        squad_players = []
        for pick in picks:
            player_info = players_df[players_df['id'] == pick['element']]
            if not player_info.empty:
                player = player_info.iloc[0]
                squad_players.append({
                    'price': float(player.get('now_cost', 0)) / 10,
                    'ownership': float(player.get('selected_by_percent', 0)),
                    'form': float(player.get('form', 0)),
                    'position': player.get('position_name', 'Unknown')
                })
        
        if not squad_players:
            return {'error': 'No valid squad players found'}
        
        # Calculate squad metrics
        avg_price = np.mean([p['price'] for p in squad_players])
        avg_ownership = np.mean([p['ownership'] for p in squad_players])
        avg_form = np.mean([p['form'] for p in squad_players])
        
        # Count premium players (>£9m)
        premium_count = len([p for p in squad_players if p['price'] > 9])
        
        # Count template players (>50% ownership)
        template_count = len([p for p in squad_players if p['ownership'] > 50])
        
        # Count differential players (<10% ownership)
        differential_count = len([p for p in squad_players if p['ownership'] < 10])
        
        return {
            'avg_price': avg_price,
            'avg_ownership': avg_ownership,
            'avg_form': avg_form,
            'premium_count': premium_count,
            'template_count': template_count,
            'differential_count': differential_count,
            'squad_style': self._determine_squad_style(template_count, differential_count, premium_count)
        }
    
    def _analyze_performance_trends(self, team_data: Dict, current_gw: int) -> Dict:
        """Analyze performance trends and momentum"""
        total_points = team_data.get('summary_overall_points', 0)
        gw_points = team_data.get('summary_event_points', 0)
        rank = team_data.get('summary_overall_rank', 0)
        
        avg_ppg = total_points / max(current_gw, 1)
        
        # Momentum analysis
        if gw_points > avg_ppg * 1.2:
            momentum = 'strong_positive'
        elif gw_points > avg_ppg * 1.05:
            momentum = 'positive'
        elif gw_points < avg_ppg * 0.8:
            momentum = 'negative'
        elif gw_points < avg_ppg * 0.95:
            momentum = 'slight_negative'
        else:
            momentum = 'stable'
        
        # Projected season end
        projected_total = avg_ppg * 38
        
        # Rank trajectory (simplified)
        if rank:
            if rank < 100000:
                trajectory = 'elite'
            elif rank < 500000:
                trajectory = 'excellent'
            elif rank < 1000000:
                trajectory = 'good'
            elif rank < 2000000:
                trajectory = 'average'
            else:
                trajectory = 'needs_improvement'
        else:
            trajectory = 'unknown'
        
        return {
            'momentum': momentum,
            'momentum_description': self._get_momentum_description(momentum),
            'projected_total': projected_total,
            'trajectory': trajectory,
            'trajectory_description': self._get_trajectory_description(trajectory),
            'weeks_remaining': 38 - current_gw
        }
    
    def _generate_comparison_recommendations(self, top_10k_comp: Dict, overall_comp: Dict, trend_analysis: Dict) -> List[str]:
        """Generate recommendations based on comparison analysis"""
        recommendations = []
        
        # Performance recommendations
        if top_10k_comp['overall_rating']['percentage'] < 50:
            recommendations.append("📈 Focus on consistency to match top 10K performance levels")
        
        if overall_comp['overall_rating']['percentage'] > 80:
            recommendations.append("🎯 You're performing well above average - aim for top 10K benchmarks")
        
        # Momentum recommendations
        momentum = trend_analysis['momentum']
        if momentum in ['negative', 'slight_negative']:
            recommendations.append("⚠️ Recent form concerning - review captain choices and transfers")
        elif momentum == 'strong_positive':
            recommendations.append("🔥 Excellent momentum - maintain current strategy")
        
        # Trajectory recommendations
        trajectory = trend_analysis['trajectory']
        if trajectory == 'needs_improvement':
            recommendations.append("💪 Significant room for improvement - consider bold strategic changes")
        elif trajectory in ['excellent', 'elite']:
            recommendations.append("🏆 Strong position - focus on maintaining rank")
        
        return recommendations
    
    def _get_phase_analysis(self, phase: str, performance_diff: float) -> str:
        """Get analysis text for current season phase"""
        if phase == 'early_season':
            if performance_diff > 5:
                return "Excellent start - building strong foundation for the season"
            elif performance_diff < -5:
                return "Slow start - focus on template players and consistency"
            else:
                return "Steady start - room for improvement as season progresses"
        
        elif phase == 'mid_season':
            if performance_diff > 3:
                return "Strong mid-season form - well positioned for final push"
            elif performance_diff < -3:
                return "Mid-season struggles - consider strategic changes"
            else:
                return "Average mid-season performance - potential for improvement"
        
        else:  # late_season
            if performance_diff > 2:
                return "Excellent late-season form - strong finish expected"
            elif performance_diff < -2:
                return "Late-season difficulties - focus on differential picks"
            else:
                return "Steady late-season performance"
    
    def _determine_squad_style(self, template_count: int, differential_count: int, premium_count: int) -> str:
        """Determine squad style based on composition"""
        if template_count >= 8:
            return "Template Heavy"
        elif differential_count >= 5:
            return "Differential Heavy"
        elif premium_count >= 4:
            return "Premium Heavy"
        elif template_count >= 5 and differential_count >= 3:
            return "Balanced"
        else:
            return "Standard"
    
    def _get_momentum_description(self, momentum: str) -> str:
        """Get description for momentum status"""
        descriptions = {
            'strong_positive': "🚀 Excellent recent form - significantly above average",
            'positive': "📈 Good recent form - above your season average",
            'stable': "➡️ Consistent performance - maintaining season average",
            'slight_negative': "📉 Slight dip in form - below season average",
            'negative': "⚠️ Poor recent form - significantly below average"
        }
        return descriptions.get(momentum, "Unknown momentum")
    
    def _get_trajectory_description(self, trajectory: str) -> str:
        """Get description for rank trajectory"""
        descriptions = {
            'elite': "🏆 Elite performance - Top 100K manager",
            'excellent': "🥇 Excellent performance - Top 500K manager", 
            'good': "👍 Good performance - Top 1M manager",
            'average': "📊 Average performance - Top 2M manager",
            'needs_improvement': "📈 Below average - significant improvement needed",
            'unknown': "❓ Rank data not available"
        }
        return descriptions.get(trajectory, "Unknown trajectory")