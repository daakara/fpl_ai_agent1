"""
Comprehensive Test Cases for Service Components
Enhanced modularization testing for FPL Analytics App
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import requests
from datetime import datetime, timedelta


class TestFPLDataService:
    """Comprehensive tests for FPL Data Service"""
    
    def test_initialization(self):
        """Test service initialization"""
        from services.fpl_data_service import FPLDataService
        service = FPLDataService()
        assert service is not None
        assert hasattr(service, 'base_url')
    
    @patch('requests.get')
    def test_load_fpl_data_success(self, mock_get, sample_players_data, sample_teams_data):
        """Test successful data loading from FPL API"""
        # Mock successful API response
        mock_response = Mock()
        mock_response.json.return_value = {
            'elements': sample_players_data.to_dict('records'),
            'teams': sample_teams_data.to_dict('records'),
            'events': [{'id': 1, 'name': 'Gameweek 1', 'finished': True}],
            'element_types': [
                {'id': 1, 'plural_name': 'Goalkeepers'},
                {'id': 2, 'plural_name': 'Defenders'},
                {'id': 3, 'plural_name': 'Midfielders'},
                {'id': 4, 'plural_name': 'Forwards'}
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        from services.fpl_data_service import FPLDataService
        service = FPLDataService()
        
        result = service.load_fpl_data()
        assert result is not None
        assert 'elements' in result
        assert 'teams' in result
        mock_get.assert_called_once()
    
    @patch('requests.get')
    def test_load_fpl_data_api_error(self, mock_get):
        """Test API error handling"""
        mock_get.side_effect = requests.RequestException("API Error")
        
        from services.fpl_data_service import FPLDataService
        service = FPLDataService()
        
        result = service.load_fpl_data()
        assert result is None or result == {}
    
    @patch('requests.get')
    def test_load_fpl_data_timeout(self, mock_get):
        """Test timeout handling"""
        mock_get.side_effect = requests.Timeout("Request timeout")
        
        from services.fpl_data_service import FPLDataService
        service = FPLDataService()
        
        result = service.load_fpl_data()
        assert result is None or result == {}
    
    @patch('requests.get')
    def test_load_player_history(self, mock_get):
        """Test loading individual player history"""
        mock_response = Mock()
        mock_response.json.return_value = {
            'history': [
                {'round': 1, 'total_points': 10, 'minutes': 90},
                {'round': 2, 'total_points': 8, 'minutes': 90}
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        from services.fpl_data_service import FPLDataService
        service = FPLDataService()
        
        result = service.load_player_history(1)
        assert result is not None
        assert 'history' in result


class TestDataServices:
    """Tests for general data services"""
    
    def test_data_processing_service_init(self):
        """Test data processing service initialization"""
        from services.data_services import DataProcessingService
        service = DataProcessingService()
        assert service is not None
    
    def test_filter_players_by_position(self, sample_players_data):
        """Test filtering players by position"""
        from services.data_services import DataProcessingService
        service = DataProcessingService()
        
        filtered = service.filter_players_by_position(sample_players_data, [3, 4])  # Mid, Forward
        assert len(filtered) == 5  # All test players are mid/forward
        assert all(pos in [3, 4] for pos in filtered['element_type'])
    
    def test_calculate_value_metrics(self, sample_players_data):
        """Test value calculation methods"""
        from services.data_services import DataProcessingService
        service = DataProcessingService()
        
        result = service.calculate_value_metrics(sample_players_data)
        assert 'points_per_million' in result.columns
        assert 'value_score' in result.columns
        assert len(result) == len(sample_players_data)
    
    def test_sort_players_by_criteria(self, sample_players_data):
        """Test player sorting functionality"""
        from services.data_services import DataProcessingService
        service = DataProcessingService()
        
        sorted_data = service.sort_players(sample_players_data, 'total_points', ascending=False)
        assert sorted_data.iloc[0]['web_name'] == 'Haaland'  # Highest points
        assert len(sorted_data) == len(sample_players_data)


class TestFixtureService:
    """Tests for fixture analysis service"""
    
    def test_fixture_service_init(self):
        """Test fixture service initialization"""
        from services.fixture_service import FixtureService
        service = FixtureService()
        assert service is not None
    
    def test_calculate_fdr_scores(self, sample_fixtures_data, sample_teams_data):
        """Test FDR calculation"""
        from services.fixture_service import FixtureService
        service = FixtureService()
        
        fdr_data = service.calculate_fdr_scores(sample_fixtures_data, sample_teams_data)
        assert len(fdr_data) == len(sample_fixtures_data)
        assert 'fdr_score' in fdr_data.columns or 'team_h_difficulty' in fdr_data.columns
    
    def test_get_upcoming_fixtures(self, sample_fixtures_data):
        """Test getting upcoming fixtures"""
        from services.fixture_service import FixtureService
        service = FixtureService()
        
        upcoming = service.get_upcoming_fixtures(sample_fixtures_data, gameweeks=3)
        assert len(upcoming) <= len(sample_fixtures_data)
        # Should only include unfinished fixtures
        if len(upcoming) > 0:
            assert not upcoming['finished'].any()
    
    def test_analyze_fixture_difficulty(self, sample_fixtures_data, sample_teams_data):
        """Test fixture difficulty analysis"""
        from services.fixture_service import FixtureService
        service = FixtureService()
        
        analysis = service.analyze_fixture_difficulty(sample_fixtures_data, sample_teams_data)
        assert analysis is not None
        assert isinstance(analysis, (dict, pd.DataFrame))


class TestVisualizationServices:
    """Tests for visualization services"""
    
    def test_visualization_service_init(self):
        """Test visualization service initialization"""
        from services.visualization_services import VisualizationService
        service = VisualizationService()
        assert service is not None
    
    def test_create_player_comparison_chart(self, sample_players_data):
        """Test player comparison chart creation"""
        from services.visualization_services import VisualizationService
        service = VisualizationService()
        
        chart = service.create_player_comparison_chart(
            sample_players_data.head(3), 
            'total_points', 
            'now_cost'
        )
        assert chart is not None
        # Should be a plotly figure
        assert hasattr(chart, 'data') or hasattr(chart, 'to_dict')
    
    def test_create_fdr_heatmap(self, sample_fixtures_data, sample_teams_data):
        """Test FDR heatmap creation"""
        from services.visualization_services import VisualizationService
        service = VisualizationService()
        
        heatmap = service.create_fdr_heatmap(sample_fixtures_data, sample_teams_data)
        assert heatmap is not None
        assert hasattr(heatmap, 'data') or hasattr(heatmap, 'to_dict')
    
    def test_create_form_trend_chart(self, sample_players_data):
        """Test form trend chart creation"""
        from services.visualization_services import VisualizationService
        service = VisualizationService()
        
        # Mock historical data
        trend_data = pd.DataFrame({
            'gameweek': [1, 2, 3, 4, 5],
            'points': [10, 8, 12, 6, 9],
            'player_name': ['Salah'] * 5
        })
        
        chart = service.create_form_trend_chart(trend_data)
        assert chart is not None
        assert hasattr(chart, 'data') or hasattr(chart, 'to_dict')


class TestIntegrationScenarios:
    """Integration tests for service interactions"""
    
    @patch('requests.get')
    def test_full_data_pipeline(self, mock_get, sample_players_data, sample_teams_data, sample_fixtures_data):
        """Test complete data loading and processing pipeline"""
        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = {
            'elements': sample_players_data.to_dict('records'),
            'teams': sample_teams_data.to_dict('records'),
            'events': [{'id': 1, 'name': 'Gameweek 1'}],
            'element_types': [{'id': 1, 'plural_name': 'Goalkeepers'}]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Test the pipeline
        from services.fpl_data_service import FPLDataService
        from services.data_services import DataProcessingService
        
        # Load data
        data_service = FPLDataService()
        raw_data = data_service.load_fpl_data()
        assert raw_data is not None
        
        # Process data
        processing_service = DataProcessingService()
        players_df = pd.DataFrame(raw_data['elements'])
        processed_data = processing_service.calculate_value_metrics(players_df)
        
        assert len(processed_data) > 0
        assert 'points_per_million' in processed_data.columns
    
    def test_service_error_propagation(self):
        """Test error handling across service boundaries"""
        from services.fpl_data_service import FPLDataService
        from services.data_services import DataProcessingService
        
        data_service = FPLDataService()
        processing_service = DataProcessingService()
        
        # Test with invalid data
        empty_df = pd.DataFrame()
        
        # Should handle empty dataframe gracefully
        try:
            result = processing_service.calculate_value_metrics(empty_df)
            # Should return empty dataframe or handle gracefully
            assert isinstance(result, pd.DataFrame)
        except Exception as e:
            # Should be a controlled exception, not a crash
            assert isinstance(e, (ValueError, KeyError))


class TestPerformanceAndCaching:
    """Tests for performance and caching functionality"""
    
    def test_data_caching_behavior(self, sample_players_data):
        """Test that data caching works properly"""
        from services.data_services import DataProcessingService
        service = DataProcessingService()
        
        # Process same data multiple times
        result1 = service.calculate_value_metrics(sample_players_data)
        result2 = service.calculate_value_metrics(sample_players_data)
        
        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_large_dataset_handling(self):
        """Test handling of larger datasets"""
        # Create larger test dataset
        large_data = pd.DataFrame({
            'id': range(1000),
            'web_name': [f'Player_{i}' for i in range(1000)],
            'total_points': np.random.randint(0, 300, 1000),
            'now_cost': np.random.randint(40, 150, 1000),
            'element_type': np.random.randint(1, 5, 1000)
        })
        
        from services.data_services import DataProcessingService
        service = DataProcessingService()
        
        # Should handle large dataset efficiently
        start_time = datetime.now()
        result = service.calculate_value_metrics(large_data)
        end_time = datetime.now()
        
        # Should complete within reasonable time (adjust threshold as needed)
        processing_time = (end_time - start_time).total_seconds()
        assert processing_time < 10.0  # Should complete within 10 seconds
        assert len(result) == 1000
