"""
Tests for the Client Manager module.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from inferbench.clients.manager import ClientManager
from inferbench.core.models import (
    ClientRun,
    RunStatus,
    ClientRecipe,
    ResourceSpec,
)
from inferbench.core.exceptions import (
    ClientRunError,
    ClientNotFoundError,
    RecipeNotFoundError,
)


class TestClientManager:
    """Tests for ClientManager class."""
    
    @pytest.fixture
    def mock_recipe_loader(self):
        """Create a mock recipe loader."""
        loader = MagicMock()
        loader.list_recipes.return_value = ["llm-stress-test", "simple-benchmark"]
        return loader
    
    @pytest.fixture
    def mock_registry(self):
        """Create a mock run registry."""
        registry = MagicMock()
        registry.get_all.return_value = []
        registry.get_active.return_value = []
        return registry
    
    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock SLURM orchestrator."""
        orchestrator = MagicMock()
        orchestrator.submit_job.return_value = "87654321"
        orchestrator.generate_batch_script.return_value = "#!/bin/bash\necho 'test'"
        orchestrator.cancel_job.return_value = True
        return orchestrator
    
    @pytest.fixture
    def mock_runtime(self):
        """Create a mock Apptainer runtime."""
        runtime = MagicMock()
        runtime.generate_exec_script.return_value = "apptainer exec test"
        return runtime
    
    @pytest.fixture
    def sample_client_recipe(self):
        """Create a sample client recipe."""
        return ClientRecipe(
            name="test-client",
            description="Test benchmark client",
            resources=ResourceSpec(nodes=1, gpus=0, memory="16G"),
            target={"url": "http://localhost:8000"},
            workload={
                "type": "stress-test",
                "pattern": {"rate": 10, "duration": 60},
                "request": {"endpoint": "/v1/completions", "method": "POST"},
                "dataset": {"prompts": ["Hello", "World"]},
            },
        )
    
    @pytest.fixture
    def manager(self, mock_recipe_loader, mock_registry, mock_orchestrator, mock_runtime, tmp_path):
        """Create a client manager with mocked dependencies."""
        with patch('inferbench.clients.manager.get_config') as mock_config:
            config = MagicMock()
            config.logs_dir = tmp_path / "logs"
            config.results_dir = tmp_path / "results"
            mock_config.return_value = config
            
            with patch('inferbench.clients.manager.get_service_registry'):
                return ClientManager(
                    recipe_loader=mock_recipe_loader,
                    registry=mock_registry,
                    orchestrator=mock_orchestrator,
                    runtime=mock_runtime,
                )
    
    def test_list_available_recipes(self, manager, mock_recipe_loader):
        """Should list available client recipes."""
        recipes = manager.list_available_recipes()
        
        assert recipes == ["llm-stress-test", "simple-benchmark"]
        mock_recipe_loader.list_recipes.assert_called_once()
    
    def test_list_runs_empty(self, manager, mock_registry):
        """Should return empty list when no runs."""
        runs = manager.list_runs()
        
        assert runs == []
        mock_registry.get_all.assert_called_once()
    
    def test_list_runs_active_only(self, manager, mock_registry):
        """Should filter active runs."""
        manager.list_runs(active_only=True)
        
        mock_registry.get_active.assert_called_once()
    
    def test_run_client_success(
        self, manager, mock_recipe_loader, mock_registry, 
        mock_orchestrator, sample_client_recipe
    ):
        """Should start a client run successfully."""
        mock_recipe_loader.load_client.return_value = sample_client_recipe
        
        run = manager.run_client("test-client", wait_for_completion=False)
        
        assert run is not None
        assert run.recipe_name == "test-client"
        assert run.slurm_job_id == "87654321"
        mock_orchestrator.submit_job.assert_called_once()
        mock_registry.register.assert_called()
    
    def test_run_client_recipe_not_found(self, manager, mock_recipe_loader):
        """Should raise error when recipe not found."""
        mock_recipe_loader.load_client.side_effect = RecipeNotFoundError("unknown", "client")
        
        with pytest.raises(RecipeNotFoundError):
            manager.run_client("unknown")
    
    def test_run_client_with_target(
        self, manager, mock_recipe_loader, sample_client_recipe, tmp_path
    ):
        """Should resolve target service endpoint."""
        mock_recipe_loader.load_client.return_value = sample_client_recipe
        
        # Mock service registry to return a service
        with patch.object(manager, 'service_registry') as mock_svc_reg:
            mock_service = MagicMock()
            mock_service.get_endpoint.return_value = "http://mel2091:8000"
            mock_svc_reg.get.return_value = mock_service
            
            run = manager.run_client(
                "test-client", 
                target_service_id="svc-001",
                wait_for_completion=False
            )
            
            assert run.target_service_id == "svc-001"
    
    def test_stop_run_success(self, manager, mock_registry, mock_orchestrator, sample_client_recipe):
        """Should stop a run successfully."""
        run = ClientRun(
            id="run-001",
            recipe_name="test-client",
            recipe=sample_client_recipe,
            status=RunStatus.RUNNING,
            slurm_job_id="87654321",
        )
        mock_registry.get.return_value = run
        
        result = manager.stop_run("run-001")
        
        assert result is True
        mock_orchestrator.cancel_job.assert_called_once_with("87654321")
    
    def test_stop_run_not_found(self, manager, mock_registry):
        """Should raise error when run not found."""
        mock_registry.get.side_effect = ClientNotFoundError("unknown")
        
        with pytest.raises(ClientNotFoundError):
            manager.stop_run("unknown")
    
    def test_stop_run_not_active(self, manager, mock_registry, sample_client_recipe):
        """Should handle stopping already completed run."""
        run = ClientRun(
            id="run-001",
            recipe_name="test-client",
            recipe=sample_client_recipe,
            status=RunStatus.COMPLETED,
        )
        mock_registry.get.return_value = run
        
        result = manager.stop_run("run-001")
        
        assert result is True
    
    def test_get_run_status(self, manager, mock_registry, sample_client_recipe):
        """Should get run status."""
        run = ClientRun(
            id="run-001",
            recipe_name="test-client",
            recipe=sample_client_recipe,
            status=RunStatus.RUNNING,
            slurm_job_id="87654321",
        )
        mock_registry.get.return_value = run
        
        result = manager.get_run_status("run-001")
        
        assert result.id == "run-001"
    
    def test_get_run_results_not_found(self, manager, mock_registry, sample_client_recipe):
        """Should return None when no results."""
        run = ClientRun(
            id="run-001",
            recipe_name="test-client",
            recipe=sample_client_recipe,
            results_path=None,
        )
        mock_registry.get.return_value = run
        
        result = manager.get_run_results("run-001")
        
        assert result is None
    
    def test_get_run_results_success(self, manager, mock_registry, sample_client_recipe, tmp_path):
        """Should return results when available."""
        import json
        
        results_dir = tmp_path / "results" / "clients" / "run-001"
        results_dir.mkdir(parents=True)
        results_file = results_dir / "benchmark_results.json"
        results_file.write_text(json.dumps({
            "benchmark": "test",
            "summary": {"total_requests": 100}
        }))
        
        run = ClientRun(
            id="run-001",
            recipe_name="test-client",
            recipe=sample_client_recipe,
            results_path=str(results_dir),
        )
        mock_registry.get.return_value = run
        
        result = manager.get_run_results("run-001")
        
        assert result is not None
        assert result["benchmark"] == "test"
        assert result["summary"]["total_requests"] == 100
    
    def test_build_http_benchmark_script(self, manager, sample_client_recipe, tmp_path):
        """Should generate valid benchmark script."""
        results_dir = tmp_path / "results"
        
        script = manager._build_http_benchmark_script(
            sample_client_recipe,
            "http://localhost:8000",
            results_dir,
        )
        
        assert "TARGET_ENDPOINT" in script
        assert "REQUEST_RATE = 10" in script
        assert "DURATION = 60" in script
        assert "run_benchmark" in script


class TestClientManagerIntegration:
    """Integration-style tests for ClientManager."""
    
    @pytest.fixture
    def manager_with_real_loader(self, tmp_path):
        """Create manager with real recipe loader."""
        recipes_dir = tmp_path / "recipes" / "clients"
        recipes_dir.mkdir(parents=True)
        
        recipe_file = recipes_dir / "test-benchmark.yaml"
        recipe_file.write_text("""
name: test-benchmark
type: client
description: Test benchmark for integration tests
resources:
  nodes: 1
  gpus: 0
  memory: 8G
  time: "00:30:00"
target:
  url: http://localhost:8000
workload:
  type: stress-test
  pattern:
    rate: 5
    duration: 30
  request:
    endpoint: /health
    method: GET
  dataset:
    prompts:
      - "test prompt"
""")
        
        from inferbench.core.recipe_loader import RecipeLoader
        loader = RecipeLoader(tmp_path / "recipes")
        
        with patch('inferbench.clients.manager.get_config') as mock_config:
            config = MagicMock()
            config.logs_dir = tmp_path / "logs"
            config.results_dir = tmp_path / "results"
            mock_config.return_value = config
            
            orchestrator = MagicMock()
            orchestrator.submit_job.return_value = "99999999"
            orchestrator.generate_batch_script.return_value = "#!/bin/bash\necho test"
            
            with patch('inferbench.clients.manager.get_service_registry'):
                return ClientManager(
                    recipe_loader=loader,
                    orchestrator=orchestrator,
                )
    
    def test_run_client_with_real_recipe(self, manager_with_real_loader):
        """Should run client using real recipe file."""
        run = manager_with_real_loader.run_client(
            "test-benchmark",
            wait_for_completion=False
        )
        
        assert run.recipe_name == "test-benchmark"
        assert run.recipe.description == "Test benchmark for integration tests"
        assert run.slurm_job_id == "99999999"
