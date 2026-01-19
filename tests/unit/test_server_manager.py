"""
Tests for the Server Manager module.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from datetime import datetime

from inferbench.servers.manager import ServerManager
from inferbench.core.models import (
    ServiceInstance,
    ServiceStatus,
    ServerRecipe,
    ContainerSpec,
    ResourceSpec,
)
from inferbench.core.exceptions import (
    ServiceStartError,
    ServiceNotFoundError,
    RecipeNotFoundError,
)


class TestServerManager:
    """Tests for ServerManager class."""
    
    @pytest.fixture
    def mock_recipe_loader(self):
        """Create a mock recipe loader."""
        loader = MagicMock()
        loader.list_recipes.return_value = ["vllm-inference", "ollama-inference"]
        return loader
    
    @pytest.fixture
    def mock_registry(self):
        """Create a mock service registry."""
        registry = MagicMock()
        registry.get_all.return_value = []
        registry.get_running.return_value = []
        return registry
    
    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock SLURM orchestrator."""
        orchestrator = MagicMock()
        orchestrator.submit_job.return_value = "12345678"
        orchestrator.get_job_status.return_value = ServiceStatus.RUNNING
        orchestrator.get_job_node.return_value = "mel2091"
        orchestrator.cancel_job.return_value = True
        # Must return a string for batch script
        orchestrator.generate_batch_script.return_value = "#!/bin/bash\necho 'test'"
        return orchestrator
    
    @pytest.fixture
    def mock_runtime(self):
        """Create a mock Apptainer runtime."""
        runtime = MagicMock()
        runtime.generate_exec_script.return_value = "apptainer exec /path/to/image.sif bash -c 'python server.py'"
        return runtime
    
    @pytest.fixture
    def sample_recipe(self):
        """Create a sample server recipe."""
        return ServerRecipe(
            name="test-server",
            description="Test server",
            container=ContainerSpec(image="/path/to/image.sif"),
            resources=ResourceSpec(nodes=1, gpus=1, memory="32G"),
            environment={"MODEL": "test"},
            command="python -m server",
        )
    
    @pytest.fixture
    def manager(self, mock_recipe_loader, mock_registry, mock_orchestrator, mock_runtime, tmp_path):
        """Create a server manager with mocked dependencies."""
        with patch('inferbench.servers.manager.get_config') as mock_config:
            config = MagicMock()
            config.logs_dir = tmp_path / "logs"
            config.results_dir = tmp_path / "results"
            mock_config.return_value = config
            
            return ServerManager(
                recipe_loader=mock_recipe_loader,
                registry=mock_registry,
                orchestrator=mock_orchestrator,
                runtime=mock_runtime,
            )
    
    def test_list_available_recipes(self, manager, mock_recipe_loader):
        """Should list available server recipes."""
        recipes = manager.list_available_recipes()
        
        assert recipes == ["vllm-inference", "ollama-inference"]
        mock_recipe_loader.list_recipes.assert_called_once()
    
    def test_list_services_empty(self, manager, mock_registry):
        """Should return empty list when no services."""
        services = manager.list_services()
        
        assert services == []
        mock_registry.get_all.assert_called_once()
    
    def test_list_services_running_only(self, manager, mock_registry):
        """Should filter running services."""
        manager.list_services(running_only=True)
        
        mock_registry.get_running.assert_called_once()
    
    def test_start_service_success(
        self, manager, mock_recipe_loader, mock_registry, 
        mock_orchestrator, sample_recipe
    ):
        """Should start a service successfully."""
        mock_recipe_loader.load_server.return_value = sample_recipe
        
        # Don't wait for ready in test
        service = manager.start_service("test-server", wait_for_ready=False)
        
        assert service is not None
        assert service.recipe_name == "test-server"
        assert service.slurm_job_id == "12345678"
        mock_orchestrator.submit_job.assert_called_once()
        mock_registry.register.assert_called()
    
    def test_start_service_recipe_not_found(self, manager, mock_recipe_loader):
        """Should raise error when recipe not found."""
        mock_recipe_loader.load_server.side_effect = RecipeNotFoundError("unknown", "server")
        
        with pytest.raises(RecipeNotFoundError):
            manager.start_service("unknown")
    
    def test_stop_service_success(self, manager, mock_registry, mock_orchestrator, sample_recipe):
        """Should stop a service successfully."""
        # Setup mock service
        service = ServiceInstance(
            id="test-001",
            recipe_name="test-server",
            recipe=sample_recipe,
            status=ServiceStatus.RUNNING,
            slurm_job_id="12345678",
        )
        mock_registry.get.return_value = service
        
        result = manager.stop_service("test-001")
        
        assert result is True
        mock_orchestrator.cancel_job.assert_called_once_with("12345678")
        mock_registry.update_status.assert_called()
    
    def test_stop_service_not_found(self, manager, mock_registry):
        """Should raise error when service not found."""
        mock_registry.get.side_effect = ServiceNotFoundError("unknown")
        mock_registry.get_by_job_id.return_value = None
        
        with pytest.raises(ServiceNotFoundError):
            manager.stop_service("unknown")
    
    def test_stop_service_already_stopped(self, manager, mock_registry, sample_recipe):
        """Should handle already stopped service."""
        service = ServiceInstance(
            id="test-001",
            recipe_name="test-server",
            recipe=sample_recipe,
            status=ServiceStatus.STOPPED,
        )
        mock_registry.get.return_value = service
        
        result = manager.stop_service("test-001")
        
        assert result is True
    
    def test_get_service_status(self, manager, mock_registry, mock_orchestrator, sample_recipe):
        """Should get service status."""
        service = ServiceInstance(
            id="test-001",
            recipe_name="test-server",
            recipe=sample_recipe,
            status=ServiceStatus.RUNNING,
            slurm_job_id="12345678",
        )
        mock_registry.get.return_value = service
        
        result = manager.get_service_status("test-001")
        
        assert result.id == "test-001"
        assert result.status == ServiceStatus.RUNNING
    
    def test_get_service_status_by_job_id(self, manager, mock_registry, sample_recipe):
        """Should find service by SLURM job ID."""
        service = ServiceInstance(
            id="test-001",
            recipe_name="test-server",
            recipe=sample_recipe,
            slurm_job_id="12345678",
        )
        mock_registry.get.side_effect = ServiceNotFoundError("12345678")
        mock_registry.get_by_job_id.return_value = service
        
        result = manager.get_service_status("12345678")
        
        assert result.slurm_job_id == "12345678"
    
    def test_get_service_logs(self, manager, mock_registry, mock_orchestrator, sample_recipe, tmp_path):
        """Should get service logs."""
        service = ServiceInstance(
            id="test-001",
            recipe_name="test-server",
            recipe=sample_recipe,
            slurm_job_id="12345678",
        )
        mock_registry.get.return_value = service
        mock_orchestrator.get_job_output.return_value = "Log content here"
        
        logs = manager.get_service_logs("test-001", lines=50)
        
        assert logs == "Log content here"
    
    def test_check_health_not_running(self, manager, mock_registry, sample_recipe):
        """Should report unhealthy if service not running."""
        service = ServiceInstance(
            id="test-001",
            recipe_name="test-server",
            recipe=sample_recipe,
            status=ServiceStatus.STOPPED,
        )
        mock_registry.get.return_value = service
        
        result = manager.check_health("test-001")
        
        assert result["healthy"] is False
        assert "not running" in result["message"]
    
    def test_apply_config_overrides(self, manager, sample_recipe):
        """Should apply configuration overrides."""
        overrides = {
            "environment": {"NEW_VAR": "value"},
            "resources": {"memory": "64G"},
        }
        
        modified = manager._apply_overrides(sample_recipe, overrides)
        
        assert modified.environment.get("NEW_VAR") == "value"
        # Original MODEL should still be there
        assert modified.environment.get("MODEL") == "test"


class TestServerManagerIntegration:
    """Integration-style tests for ServerManager."""
    
    @pytest.fixture
    def manager_with_real_loader(self, tmp_path):
        """Create manager with real recipe loader."""
        # Create recipe directory
        recipes_dir = tmp_path / "recipes" / "servers"
        recipes_dir.mkdir(parents=True)
        
        # Create a test recipe
        recipe_file = recipes_dir / "test-service.yaml"
        recipe_file.write_text("""
name: test-service
type: server
description: Test service for integration tests
container:
  image: /path/to/test.sif
resources:
  nodes: 1
  gpus: 1
  memory: 16G
  time: "01:00:00"
network:
  ports:
    - name: api
      port: 8000
command: echo "Hello"
""")
        
        from inferbench.core.recipe_loader import RecipeLoader
        loader = RecipeLoader(tmp_path / "recipes")
        
        with patch('inferbench.servers.manager.get_config') as mock_config:
            config = MagicMock()
            config.logs_dir = tmp_path / "logs"
            config.results_dir = tmp_path / "results"
            mock_config.return_value = config
            
            # Mock orchestrator and runtime
            orchestrator = MagicMock()
            orchestrator.submit_job.return_value = "99999999"
            orchestrator.generate_batch_script.return_value = "#!/bin/bash\necho 'test'"
            runtime = MagicMock()
            runtime.generate_exec_script.return_value = "echo test"
            
            return ServerManager(
                recipe_loader=loader,
                orchestrator=orchestrator,
                runtime=runtime,
            )
    
    def test_start_service_with_real_recipe(self, manager_with_real_loader):
        """Should start service using real recipe file."""
        service = manager_with_real_loader.start_service(
            "test-service",
            wait_for_ready=False
        )
        
        assert service.recipe_name == "test-service"
        assert service.recipe.description == "Test service for integration tests"
        assert service.slurm_job_id == "99999999"
