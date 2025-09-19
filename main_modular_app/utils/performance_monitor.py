"""
Performance Monitoring and Optimization System
"""
import time
import psutil
import streamlit as st
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from contextlib import contextmanager
import functools
import threading

@dataclass
class PerformanceMetric:
    """Performance metric data structure"""
    function_name: str
    execution_time: float
    memory_usage: float
    timestamp: datetime
    args_hash: str
    status: str  # 'success', 'error', 'cached'

class PerformanceMonitor:
    """Monitor and track application performance"""
    
    def __init__(self, max_metrics: int = 1000):
        self.metrics: List[PerformanceMetric] = []
        self.max_metrics = max_metrics
        self._lock = threading.Lock()
    
    def record_metric(self, metric: PerformanceMetric):
        """Record a performance metric"""
        with self._lock:
            self.metrics.append(metric)
            # Keep only recent metrics
            if len(self.metrics) > self.max_metrics:
                self.metrics = self.metrics[-self.max_metrics:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics"""
        if not self.metrics:
            return {"message": "No performance data available"}
        
        recent_metrics = [m for m in self.metrics 
                         if m.timestamp > datetime.now() - timedelta(hours=1)]
        
        df = pd.DataFrame([
            {
                'function': m.function_name,
                'execution_time': m.execution_time,
                'memory_usage': m.memory_usage,
                'status': m.status
            } for m in recent_metrics
        ])
        
        if df.empty:
            return {"message": "No recent performance data"}
        
        summary = {
            'total_calls': len(df),
            'avg_execution_time': df['execution_time'].mean(),
            'max_execution_time': df['execution_time'].max(),
            'avg_memory_usage': df['memory_usage'].mean(),
            'cache_hit_rate': len(df[df['status'] == 'cached']) / len(df) * 100,
            'slowest_functions': df.groupby('function')['execution_time'].mean().nlargest(5).to_dict(),
            'memory_intensive_functions': df.groupby('function')['memory_usage'].mean().nlargest(5).to_dict()
        }
        
        return summary
    
    def get_metrics_dataframe(self) -> pd.DataFrame:
        """Get metrics as DataFrame for analysis"""
        return pd.DataFrame([
            {
                'function': m.function_name,
                'execution_time': m.execution_time,
                'memory_usage': m.memory_usage,
                'timestamp': m.timestamp,
                'status': m.status
            } for m in self.metrics
        ])

# Global performance monitor
performance_monitor = PerformanceMonitor()

@contextmanager
def performance_tracking(function_name: str, args_hash: str = ""):
    """Context manager for tracking function performance"""
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    status = "success"
    
    try:
        yield
    except Exception as e:
        status = "error"
        raise
    finally:
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        metric = PerformanceMetric(
            function_name=function_name,
            execution_time=end_time - start_time,
            memory_usage=end_memory - start_memory,
            timestamp=datetime.now(),
            args_hash=args_hash,
            status=status
        )
        
        performance_monitor.record_metric(metric)

def monitor_performance(track_memory: bool = True):
    """Decorator to monitor function performance"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            args_hash = str(hash(str(args) + str(sorted(kwargs.items()))))
            
            with performance_tracking(func.__name__, args_hash):
                result = func(*args, **kwargs)
            
            return result
        
        return wrapper
    return decorator

def display_performance_dashboard():
    """Display performance monitoring dashboard"""
    st.subheader("🚀 Performance Dashboard")
    
    summary = performance_monitor.get_performance_summary()
    
    if "message" in summary:
        st.info(summary["message"])
        return
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Calls", summary['total_calls'])
    
    with col2:
        st.metric("Avg Response Time", f"{summary['avg_execution_time']:.2f}s")
    
    with col3:
        st.metric("Cache Hit Rate", f"{summary['cache_hit_rate']:.1f}%")
    
    with col4:
        st.metric("Avg Memory Usage", f"{summary['avg_memory_usage']:.1f}MB")
    
    # Performance charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Slowest Functions")
        if summary['slowest_functions']:
            slow_df = pd.DataFrame(
                list(summary['slowest_functions'].items()),
                columns=['Function', 'Avg Time (s)']
            )
            st.bar_chart(slow_df.set_index('Function'))
    
    with col2:
        st.subheader("Memory Intensive Functions")
        if summary['memory_intensive_functions']:
            memory_df = pd.DataFrame(
                list(summary['memory_intensive_functions'].items()),
                columns=['Function', 'Avg Memory (MB)']
            )
            st.bar_chart(memory_df.set_index('Function'))
    
    # Detailed metrics
    if st.checkbox("Show Detailed Metrics"):
        df = performance_monitor.get_metrics_dataframe()
        if not df.empty:
            st.dataframe(df.tail(50))

def optimize_streamlit_performance():
    """Apply Streamlit-specific performance optimizations"""
    
    # Session state optimization
    if 'performance_optimized' not in st.session_state:
        # Enable experimental features for better performance
        if hasattr(st, 'experimental_memo'):
            st.experimental_memo.clear()
        
        # Configure session state for performance
        st.session_state.performance_optimized = True
    
    # Memory cleanup
    if len(st.session_state) > 50:  # Arbitrary threshold
        # Clean up old session state entries
        keys_to_remove = []
        for key in st.session_state.keys():
            if key.startswith('temp_') and hasattr(st.session_state[key], 'timestamp'):
                if datetime.now() - st.session_state[key].timestamp > timedelta(minutes=30):
                    keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del st.session_state[key]