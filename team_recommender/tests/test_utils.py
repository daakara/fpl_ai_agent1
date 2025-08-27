import pandas as pd
import pytest
from team_recommender.src.utils import calculate_team_chemistry_bonus
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
import requests
import httpx
from utils import retry_with_backoff, async_retry_with_backoff

def test_calculate_team_chemistry_bonus():
    team = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'team_name': ['Team A', 'Team A', 'Team B', 'Team B', 'Team A'],
        'expected_points_next_5': [5, 6, 7, 8, 5]
    })

    # Test with players from the same team
    bonus = calculate_team_chemistry_bonus(team)
    assert bonus >= 0  # Chemistry bonus should be non-negative

    # Test with players from different teams
    team['team_name'] = ['Team A', 'Team B', 'Team C', 'Team D', 'Team E']
    bonus = calculate_team_chemistry_bonus(team)
    assert bonus == 0  # No chemistry bonus for different teams

def test_some_other_utility_function():
    # Placeholder for another utility function test
    pass  # This test is intentionally left blank for future implementation.
    def test_retry_with_backoff_success_no_retry():
        """Test that successful function executes without retry"""
        mock_func = Mock(return_value="success")
        decorated_func = retry_with_backoff()(mock_func)
        
        result = decorated_func("arg1", kwarg1="value1")
        
        assert result == "success"
        assert mock_func.call_count == 1
        mock_func.assert_called_once_with("arg1", kwarg1="value1")

    def test_retry_with_backoff_success_after_failures():
        """Test that function succeeds after some failures"""
        mock_func = Mock(side_effect=[
            requests.exceptions.RequestException("error"),
            requests.exceptions.RequestException("error"),
            "success"
        ])
        decorated_func = retry_with_backoff(retries=3)(mock_func)
        
        result = decorated_func()
        
        assert result == "success"
        assert mock_func.call_count == 3

    def test_retry_with_backoff_all_retries_exhausted():
        """Test that function fails after all retries are exhausted"""
        mock_func = Mock(side_effect=requests.exceptions.RequestException("persistent error"))
        decorated_func = retry_with_backoff(retries=2)(mock_func)
        
        with pytest.raises(requests.exceptions.RequestException):
            decorated_func()
        
        assert mock_func.call_count == 2

    @patch('time.sleep')
    def test_retry_with_backoff_timing(mock_sleep):
        """Test that backoff timing increases exponentially"""
        mock_func = Mock(side_effect=[
            requests.exceptions.RequestException("error"),
            requests.exceptions.RequestException("error"),
            "success"
        ])
        decorated_func = retry_with_backoff(retries=3, backoff_in_seconds=2)(mock_func)
        
        result = decorated_func()
        
        assert result == "success"
        # Check that sleep was called with increasing intervals
        expected_calls = [2, 4]  # 2, then 2*2
        actual_calls = [call[0][0] for call in mock_sleep.call_args_list]
        assert actual_calls == expected_calls

    def test_retry_with_backoff_custom_exceptions():
        """Test with custom exception types"""
        class CustomError(Exception):
            pass
        
        mock_func = Mock(side_effect=[CustomError("error"), "success"])
        decorated_func = retry_with_backoff(
            retries=2, 
            catch_exceptions=(CustomError,)
        )(mock_func)
        
        result = decorated_func()
        
        assert result == "success"
        assert mock_func.call_count == 2

    def test_retry_with_backoff_unhandled_exception():
        """Test that unhandled exceptions are not retried"""
        class UnhandledException(Exception):
            pass
        
        mock_func = Mock(side_effect=UnhandledException("unhandled"))
        decorated_func = retry_with_backoff(
            catch_exceptions=(requests.exceptions.RequestException,)
        )(mock_func)
        
        with pytest.raises(UnhandledException):
            decorated_func()
        
        assert mock_func.call_count == 1

    @pytest.mark.asyncio
    async def test_async_retry_with_backoff_success_no_retry():
        """Test that successful async function executes without retry"""
        mock_func = AsyncMock(return_value="async_success")
        decorated_func = async_retry_with_backoff()(mock_func)
        
        result = await decorated_func("arg1", kwarg1="value1")
        
        assert result == "async_success"
        assert mock_func.call_count == 1
        mock_func.assert_called_once_with("arg1", kwarg1="value1")

    @pytest.mark.asyncio
    async def test_async_retry_with_backoff_success_after_failures():
        """Test that async function succeeds after some failures"""
        mock_func = AsyncMock(side_effect=[
            httpx.RequestError("error"),
            httpx.RequestError("error"),
            "async_success"
        ])
        decorated_func = async_retry_with_backoff(retries=3)(mock_func)
        
        result = await decorated_func()
        
        assert result == "async_success"
        assert mock_func.call_count == 3

    @pytest.mark.asyncio
    async def test_async_retry_with_backoff_all_retries_exhausted():
        """Test that async function fails after all retries are exhausted"""
        mock_func = AsyncMock(side_effect=httpx.RequestError("persistent error"))
        decorated_func = async_retry_with_backoff(retries=2)(mock_func)
        
        with pytest.raises(httpx.RequestError):
            await decorated_func()
        
        assert mock_func.call_count == 2

    @pytest.mark.asyncio
    @patch('asyncio.sleep')
    async def test_async_retry_with_backoff_timing(mock_sleep):
        """Test that async backoff timing increases exponentially"""
        mock_func = AsyncMock(side_effect=[
            httpx.RequestError("error"),
            httpx.RequestError("error"),
            "async_success"
        ])
        decorated_func = async_retry_with_backoff(retries=3, backoff_in_seconds=1)(mock_func)
        
        result = await decorated_func()
        
        assert result == "async_success"
        # Check that asyncio.sleep was called with increasing intervals
        expected_calls = [1, 2]  # 1, then 1*2
        actual_calls = [call[0][0] for call in mock_sleep.call_args_list]
        assert actual_calls == expected_calls

    @pytest.mark.asyncio
    async def test_async_retry_with_backoff_custom_exceptions():
        """Test async decorator with custom exception types"""
        class AsyncCustomError(Exception):
            pass
        
        mock_func = AsyncMock(side_effect=[AsyncCustomError("error"), "async_success"])
        decorated_func = async_retry_with_backoff(
            retries=2, 
            catch_exceptions=(AsyncCustomError,)
        )(mock_func)
        
        result = await decorated_func()
        
        assert result == "async_success"
        assert mock_func.call_count == 2

    @pytest.mark.asyncio
    async def test_async_retry_with_backoff_unhandled_exception():
        """Test that unhandled async exceptions are not retried"""
        class UnhandledAsyncException(Exception):
            pass
        
        mock_func = AsyncMock(side_effect=UnhandledAsyncException("unhandled"))
        decorated_func = async_retry_with_backoff(
            catch_exceptions=(httpx.RequestError,)
        )(mock_func)
        
        with pytest.raises(UnhandledAsyncException):
            await decorated_func()
        
        assert mock_func.call_count == 1

    def test_retry_with_backoff_zero_retries():
        """Test behavior with zero retries"""
        mock_func = Mock(side_effect=requests.exceptions.RequestException("error"))
        decorated_func = retry_with_backoff(retries=1)(mock_func)
        
        with pytest.raises(requests.exceptions.RequestException):
            decorated_func()
        
        assert mock_func.call_count == 1

    @pytest.mark.asyncio
    async def test_async_retry_with_backoff_zero_retries():
        """Test async behavior with zero retries"""
        mock_func = AsyncMock(side_effect=httpx.RequestError("error"))
        decorated_func = async_retry_with_backoff(retries=1)(mock_func)
        
        with pytest.raises(httpx.RequestError):
            await decorated_func()
        
        assert mock_func.call_count == 1

    def test_retry_with_backoff_preserves_function_metadata():
        """Test that decorator preserves original function metadata"""
        def original_func():
            """Original docstring"""
            return "result"
        
        decorated_func = retry_with_backoff()(original_func)
        
        assert decorated_func.__name__ == "original_func"
        assert decorated_func.__doc__ == "Original docstring"

    @pytest.mark.asyncio
    async def test_async_retry_with_backoff_preserves_function_metadata():
        """Test that async decorator preserves original function metadata"""
        async def original_async_func():
            """Original async docstring"""
            return "async_result"
        
        decorated_func = async_retry_with_backoff()(original_async_func)
        
        assert decorated_func.__name__ == "original_async_func"
        assert decorated_func.__doc__ == "Original async docstring"