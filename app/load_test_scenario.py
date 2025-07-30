import os
import logging
from typing import Dict, Any
# @Author: Swastik N. (2025)

logger = logging.getLogger(__name__)

class LoadTestScenario:
    """Define different load testing scenarios"""
    
    @staticmethod
    def light_load_scenario() -> Dict[str, Any]:
        """Light load scenario configuration"""
        return {
            'name': 'light_load',
            'num_services': 3,
            'instances_per_service': 5,
            'concurrent_requests': 10,
            'request_rate_per_second': 50,
            'test_duration_seconds': 60,
            'user_sessions': 20
        }
    
    @staticmethod
    def medium_load_scenario() -> Dict[str, Any]:
        """Medium load scenario configuration"""
        return {
            'name': 'medium_load',
            'num_services': 5,
            'instances_per_service': 10,
            'concurrent_requests': 50,
            'request_rate_per_second': 200,
            'test_duration_seconds': 300,
            'user_sessions': 100
        }
    
    @staticmethod
    def heavy_load_scenario() -> Dict[str, Any]:
        """Heavy load scenario configuration"""
        return {
            'name': 'heavy_load',
            'num_services': 10,
            'instances_per_service': 20,
            'concurrent_requests': 200,
            'request_rate_per_second': 1000,
            'test_duration_seconds': 600,
            'user_sessions': 500
        }
    
    @staticmethod
    def burst_load_scenario() -> Dict[str, Any]:
        """Burst load scenario configuration"""
        return {
            'name': 'burst_load',
            'num_services': 5,
            'instances_per_service': 15,
            'concurrent_requests': 500,
            'request_rate_per_second': 2000,
            'test_duration_seconds': 120,
            'user_sessions': 1000,
            'burst_pattern': True
        }