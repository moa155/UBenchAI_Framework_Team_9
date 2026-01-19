"""
Tests for the Web Interface module.
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

from inferbench.core.models import ServiceStatus, RunStatus


class TestWebApp:
    """Tests for Flask web application."""
    
    @pytest.fixture
    def app(self):
        """Create test Flask app."""
        with patch('inferbench.interface.web.app.get_server_manager'), \
             patch('inferbench.interface.web.app.get_client_manager'), \
             patch('inferbench.interface.web.app.get_monitor_manager'), \
             patch('inferbench.interface.web.app.get_log_manager'):
            from inferbench.interface.web.app import create_app
            app = create_app({"TESTING": True})
            return app
    
    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return app.test_client()
    
    def test_health_check(self, client):
        """Should return healthy status."""
        response = client.get('/health')
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
        assert data['version'] == '0.1.0'
    
    def test_index_page(self, client):
        """Should return index page."""
        response = client.get('/')
        
        assert response.status_code == 200
        assert b'InferBench' in response.data
    
    def test_services_page(self, client):
        """Should return services page."""
        response = client.get('/services')
        
        assert response.status_code == 200
        assert b'Services' in response.data
    
    def test_benchmarks_page(self, client):
        """Should return benchmarks page."""
        response = client.get('/benchmarks')
        
        assert response.status_code == 200
        assert b'Benchmarks' in response.data
    
    def test_monitoring_page(self, client):
        """Should return monitoring page."""
        response = client.get('/monitoring')
        
        assert response.status_code == 200
        assert b'Monitoring' in response.data
    
    def test_logs_page(self, client):
        """Should return logs page."""
        response = client.get('/logs')
        
        assert response.status_code == 200
        assert b'Logs' in response.data


class TestServicesAPI:
    """Tests for Services API endpoints."""
    
    def test_list_services(self):
        """Should list services."""
        mock_service = MagicMock()
        mock_service.id = "svc-001"
        mock_service.recipe_name = "vllm-inference"
        mock_service.status = ServiceStatus.RUNNING
        mock_service.node = "mel2091"
        mock_service.slurm_job_id = "12345678"
        mock_service.created_at = datetime.now()
        mock_service.endpoints = {"api": "http://mel2091:8000"}
        
        mock_manager = MagicMock()
        mock_manager.list_services.return_value = [mock_service]
        
        with patch('inferbench.interface.web.app.get_server_manager', return_value=mock_manager), \
             patch('inferbench.interface.web.app.get_client_manager'), \
             patch('inferbench.interface.web.app.get_monitor_manager'), \
             patch('inferbench.interface.web.app.get_log_manager'):
            from inferbench.interface.web.app import create_app
            app = create_app({"TESTING": True})
            client = app.test_client()
            
            response = client.get('/api/services')
            
            assert response.status_code == 200
            data = response.get_json()
            assert 'services' in data
            assert data['total'] == 1
            assert data['services'][0]['recipe_name'] == 'vllm-inference'
    
    def test_get_service(self):
        """Should get service details."""
        mock_service = MagicMock()
        mock_service.id = "svc-001"
        mock_service.recipe_name = "vllm-inference"
        mock_service.status = ServiceStatus.RUNNING
        mock_service.node = "mel2091"
        mock_service.slurm_job_id = "12345678"
        mock_service.created_at = datetime.now()
        mock_service.started_at = datetime.now()
        mock_service.endpoints = {"api": "http://mel2091:8000"}
        mock_service.error_message = None
        
        mock_manager = MagicMock()
        mock_manager.get_service_status.return_value = mock_service
        
        with patch('inferbench.interface.web.app.get_server_manager', return_value=mock_manager), \
             patch('inferbench.interface.web.app.get_client_manager'), \
             patch('inferbench.interface.web.app.get_monitor_manager'), \
             patch('inferbench.interface.web.app.get_log_manager'):
            from inferbench.interface.web.app import create_app
            app = create_app({"TESTING": True})
            client = app.test_client()
            
            response = client.get('/api/services/svc-001')
            
            assert response.status_code == 200
            data = response.get_json()
            assert data['id'] == 'svc-001'
            assert data['status'] == 'running'


class TestBenchmarksAPI:
    """Tests for Benchmarks API endpoints."""
    
    def test_list_benchmarks(self):
        """Should list benchmark runs."""
        mock_run = MagicMock()
        mock_run.id = "run-001"
        mock_run.recipe_name = "llm-stress-test"
        mock_run.status = RunStatus.COMPLETED
        mock_run.slurm_job_id = "87654321"
        mock_run.target_service_id = None
        mock_run.created_at = datetime.now()
        mock_run.completed_at = datetime.now()
        mock_run.results_path = "/path/to/results"
        
        mock_manager = MagicMock()
        mock_manager.list_runs.return_value = [mock_run]
        
        with patch('inferbench.interface.web.app.get_server_manager'), \
             patch('inferbench.interface.web.app.get_client_manager', return_value=mock_manager), \
             patch('inferbench.interface.web.app.get_monitor_manager'), \
             patch('inferbench.interface.web.app.get_log_manager'):
            from inferbench.interface.web.app import create_app
            app = create_app({"TESTING": True})
            client = app.test_client()
            
            response = client.get('/api/benchmarks')
            
            assert response.status_code == 200
            data = response.get_json()
            assert 'runs' in data
            assert data['total'] == 1
    
    def test_get_benchmark_results(self):
        """Should get benchmark results."""
        mock_manager = MagicMock()
        mock_manager.get_run_results.return_value = {
            "summary": {"total_requests": 100, "success_rate": 95.0}
        }
        
        with patch('inferbench.interface.web.app.get_server_manager'), \
             patch('inferbench.interface.web.app.get_client_manager', return_value=mock_manager), \
             patch('inferbench.interface.web.app.get_monitor_manager'), \
             patch('inferbench.interface.web.app.get_log_manager'):
            from inferbench.interface.web.app import create_app
            app = create_app({"TESTING": True})
            client = app.test_client()
            
            response = client.get('/api/benchmarks/run-001/results')
            
            assert response.status_code == 200
            data = response.get_json()
            assert data['summary']['total_requests'] == 100


class TestDashboardAPI:
    """Tests for Dashboard API endpoints."""
    
    def test_dashboard_stats(self):
        """Should return dashboard statistics."""
        mock_server = MagicMock()
        mock_server.list_services.return_value = []
        
        mock_client = MagicMock()
        mock_client.list_runs.return_value = []
        
        mock_monitor = MagicMock()
        mock_monitor.list_monitors.return_value = []
        
        with patch('inferbench.interface.web.app.get_server_manager', return_value=mock_server), \
             patch('inferbench.interface.web.app.get_client_manager', return_value=mock_client), \
             patch('inferbench.interface.web.app.get_monitor_manager', return_value=mock_monitor), \
             patch('inferbench.interface.web.app.get_log_manager'):
            from inferbench.interface.web.app import create_app
            app = create_app({"TESTING": True})
            client = app.test_client()
            
            response = client.get('/api/dashboard/stats')
            
            assert response.status_code == 200
            data = response.get_json()
            assert 'services' in data
            assert 'benchmarks' in data
            assert 'monitors' in data
            assert 'timestamp' in data
