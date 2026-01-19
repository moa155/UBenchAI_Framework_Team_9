"""
Tests for the Monitor Manager module.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from inferbench.monitors.manager import MonitorManager
from inferbench.core.models import (
    MonitorInstance,
    MonitorRecipe,
    ServiceInstance,
    ServiceStatus,
    ServerRecipe,
    ContainerSpec,
    MetricsSpec,
)
from inferbench.core.exceptions import MonitorError


class TestMonitorManager:
    """Tests for MonitorManager class."""
    
    @pytest.fixture
    def mock_recipe_loader(self):
        """Create a mock recipe loader."""
        loader = MagicMock()
        loader.list_recipes.return_value = ["default-monitor", "vllm-monitor"]
        return loader
    
    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock SLURM orchestrator."""
        orchestrator = MagicMock()
        orchestrator.submit_job.return_value = "11111111"
        orchestrator.generate_batch_script.return_value = "#!/bin/bash\necho 'test'"
        orchestrator.cancel_job.return_value = True
        orchestrator.get_job_status.return_value = ServiceStatus.RUNNING
        orchestrator.get_job_node.return_value = "mel2091"
        return orchestrator
    
    @pytest.fixture
    def sample_monitor_recipe(self):
        """Create a sample monitor recipe."""
        return MonitorRecipe(
            name="test-monitor",
            description="Test monitoring stack",
            targets=[],
            prometheus={"port": 9090},
            grafana={"port": 3000, "admin_password": "admin"},
            scrape_interval=15,
            retention="7d",
        )
    
    @pytest.fixture
    def sample_service(self):
        """Create a sample service for targeting."""
        server_recipe = ServerRecipe(
            name="test-server",
            container=ContainerSpec(image="/path/to/image.sif"),
            metrics=MetricsSpec(enabled=True, port=8000, endpoint="/metrics"),
        )
        return ServiceInstance(
            id="svc-001",
            recipe_name="test-server",
            recipe=server_recipe,
            status=ServiceStatus.RUNNING,
            node="mel2091",
        )
    
    @pytest.fixture
    def manager(self, mock_recipe_loader, mock_orchestrator, tmp_path):
        """Create a monitor manager with mocked dependencies."""
        with patch('inferbench.monitors.manager.get_config') as mock_config:
            config = MagicMock()
            config.logs_dir = tmp_path / "logs"
            config.results_dir = tmp_path / "results"
            mock_config.return_value = config
            
            with patch('inferbench.monitors.manager.get_service_registry') as mock_svc_reg:
                mock_svc_reg.return_value = MagicMock()
                
                return MonitorManager(
                    recipe_loader=mock_recipe_loader,
                    orchestrator=mock_orchestrator,
                )
    
    def test_list_available_recipes(self, manager, mock_recipe_loader):
        """Should list available monitor recipes."""
        recipes = manager.list_available_recipes()
        
        assert recipes == ["default-monitor", "vllm-monitor"]
        mock_recipe_loader.list_recipes.assert_called_once()
    
    def test_list_monitors_empty(self, manager):
        """Should return empty list when no monitors."""
        monitors = manager.list_monitors()
        
        assert monitors == []
    
    def test_start_monitor_success(
        self, manager, mock_recipe_loader, mock_orchestrator, 
        sample_monitor_recipe
    ):
        """Should start a monitor successfully."""
        mock_recipe_loader.load_monitor.return_value = sample_monitor_recipe
        
        monitor = manager.start_monitor("test-monitor", wait_for_ready=False)
        
        assert monitor is not None
        assert monitor.recipe_name == "test-monitor"
        assert monitor.prometheus_job_id == "11111111"
        mock_orchestrator.submit_job.assert_called()
    
    def test_start_monitor_with_targets(
        self, manager, mock_recipe_loader, mock_orchestrator,
        sample_monitor_recipe, sample_service
    ):
        """Should resolve and add targets."""
        mock_recipe_loader.load_monitor.return_value = sample_monitor_recipe
        manager.service_registry.get.return_value = sample_service
        
        monitor = manager.start_monitor(
            "test-monitor",
            target_ids=["svc-001"],
            wait_for_ready=False
        )
        
        assert "svc-001" in monitor.targets
    
    def test_stop_monitor_success(
        self, manager, mock_recipe_loader, mock_orchestrator,
        sample_monitor_recipe
    ):
        """Should stop a monitor successfully."""
        mock_recipe_loader.load_monitor.return_value = sample_monitor_recipe
        
        # First start a monitor
        monitor = manager.start_monitor("test-monitor", wait_for_ready=False)
        
        # Then stop it
        result = manager.stop_monitor(monitor.id)
        
        assert result is True
        mock_orchestrator.cancel_job.assert_called()
    
    def test_stop_monitor_not_found(self, manager):
        """Should raise error when monitor not found."""
        with pytest.raises(MonitorError) as exc_info:
            manager.stop_monitor("unknown")
        
        assert "not found" in str(exc_info.value)
    
    def test_get_monitor_status(
        self, manager, mock_recipe_loader, mock_orchestrator,
        sample_monitor_recipe
    ):
        """Should get monitor status."""
        mock_recipe_loader.load_monitor.return_value = sample_monitor_recipe
        
        monitor = manager.start_monitor("test-monitor", wait_for_ready=False)
        
        status = manager.get_monitor_status(monitor.id)
        
        assert status.id == monitor.id
        assert status.status == ServiceStatus.RUNNING
    
    def test_add_target(
        self, manager, mock_recipe_loader, mock_orchestrator,
        sample_monitor_recipe, sample_service, tmp_path
    ):
        """Should add target to existing monitor."""
        mock_recipe_loader.load_monitor.return_value = sample_monitor_recipe
        manager.service_registry.get.return_value = sample_service
        
        # Start monitor
        monitor = manager.start_monitor("test-monitor", wait_for_ready=False)
        
        # Add target
        result = manager.add_target(monitor.id, "svc-001")
        
        assert result is True
        assert "svc-001" in manager._monitors[monitor.id].targets
    
    def test_remove_target(
        self, manager, mock_recipe_loader, mock_orchestrator,
        sample_monitor_recipe, sample_service
    ):
        """Should remove target from monitor."""
        mock_recipe_loader.load_monitor.return_value = sample_monitor_recipe
        manager.service_registry.get.return_value = sample_service
        
        # Start with target
        monitor = manager.start_monitor(
            "test-monitor",
            target_ids=["svc-001"],
            wait_for_ready=False
        )
        
        # Remove target
        result = manager.remove_target(monitor.id, "svc-001")
        
        assert result is True
        assert "svc-001" not in manager._monitors[monitor.id].targets
    
    def test_resolve_targets(self, manager, sample_service):
        """Should resolve service IDs to scrape targets."""
        manager.service_registry.get.return_value = sample_service
        
        targets = manager._resolve_targets(["svc-001"])
        
        assert len(targets) == 1
        assert "mel2091:8000" in targets[0]["targets"]
        assert targets[0]["labels"]["service_id"] == "svc-001"
    
    def test_resolve_targets_service_not_running(self, manager, sample_service):
        """Should skip non-running services."""
        sample_service.status = ServiceStatus.STOPPED
        manager.service_registry.get.return_value = sample_service
        
        targets = manager._resolve_targets(["svc-001"])
        
        assert len(targets) == 0
    
    def test_generate_prometheus_config(self, manager, sample_monitor_recipe, tmp_path):
        """Should generate valid Prometheus config."""
        targets = [
            {
                "targets": ["mel2091:8000"],
                "labels": {"service_id": "svc-001", "job": "test"}
            }
        ]
        
        config = manager._generate_prometheus_config(
            sample_monitor_recipe,
            targets,
            tmp_path
        )
        
        assert "scrape_interval: 15s" in config
        assert "inferbench_services" in config
        assert "prometheus" in config
    
    def test_generate_vllm_dashboard(self, manager):
        """Should generate vLLM dashboard JSON."""
        dashboard = manager._generate_vllm_dashboard()
        
        assert dashboard["title"] == "vLLM Metrics"
        assert len(dashboard["panels"]) == 4
        assert dashboard["uid"] == "inferbench-vllm"


class TestMonitorManagerIntegration:
    """Integration-style tests for MonitorManager."""
    
    @pytest.fixture
    def manager_with_real_loader(self, tmp_path):
        """Create manager with real recipe loader."""
        recipes_dir = tmp_path / "recipes" / "monitors"
        recipes_dir.mkdir(parents=True)
        
        recipe_file = recipes_dir / "test-monitor.yaml"
        recipe_file.write_text("""
name: test-monitor
type: monitor
description: Test monitoring for integration tests
targets: []
prometheus:
  port: 9090
grafana:
  port: 3000
  admin_password: testpass
scrape_interval: 15
retention: "1d"
""")
        
        from inferbench.core.recipe_loader import RecipeLoader
        loader = RecipeLoader(tmp_path / "recipes")
        
        with patch('inferbench.monitors.manager.get_config') as mock_config:
            config = MagicMock()
            config.logs_dir = tmp_path / "logs"
            config.results_dir = tmp_path / "results"
            mock_config.return_value = config
            
            orchestrator = MagicMock()
            orchestrator.submit_job.return_value = "99999999"
            orchestrator.generate_batch_script.return_value = "#!/bin/bash\necho test"
            orchestrator.get_job_status.return_value = ServiceStatus.RUNNING
            orchestrator.get_job_node.return_value = "mel2091"
            
            with patch('inferbench.monitors.manager.get_service_registry'):
                return MonitorManager(
                    recipe_loader=loader,
                    orchestrator=orchestrator,
                )
    
    def test_start_monitor_with_real_recipe(self, manager_with_real_loader):
        """Should start monitor using real recipe file."""
        monitor = manager_with_real_loader.start_monitor(
            "test-monitor",
            wait_for_ready=False
        )
        
        assert monitor.recipe_name == "test-monitor"
        assert monitor.recipe.description == "Test monitoring for integration tests"
        assert monitor.prometheus_job_id == "99999999"
