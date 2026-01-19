"""
Tests for the registry module.
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path

from inferbench.core.registry import ServiceRegistry, RunRegistry
from inferbench.core.models import (
    ServiceInstance,
    ServiceStatus,
    ClientRun,
    RunStatus,
    ServerRecipe,
    ClientRecipe,
    ContainerSpec,
)
from inferbench.core.exceptions import ServiceNotFoundError, ClientNotFoundError


@pytest.fixture
def sample_server_recipe():
    """Create a sample server recipe for testing."""
    return ServerRecipe(
        name="test-recipe",
        container=ContainerSpec(image="/path/to/image.sif"),
    )


@pytest.fixture
def sample_client_recipe():
    """Create a sample client recipe for testing."""
    return ClientRecipe(name="test-client")


class TestServiceRegistry:
    """Tests for ServiceRegistry class."""
    
    @pytest.fixture
    def registry(self, tmp_path):
        """Create a service registry with temp persistence."""
        return ServiceRegistry(persistence_path=tmp_path / "services.json")
    
    @pytest.fixture
    def sample_service(self, sample_server_recipe):
        """Create a sample service instance."""
        return ServiceInstance(
            id="test-001",
            recipe_name="test-recipe",
            recipe=sample_server_recipe,
            status=ServiceStatus.PENDING,
        )
    
    def test_register_service(self, registry, sample_service):
        """Should register a new service."""
        result = registry.register(sample_service)
        
        assert result is True
        assert len(registry.get_all()) == 1
    
    def test_get_service(self, registry, sample_service):
        """Should retrieve a registered service."""
        registry.register(sample_service)
        
        service = registry.get("test-001")
        
        assert service.id == "test-001"
        assert service.recipe_name == "test-recipe"
    
    def test_get_service_not_found(self, registry):
        """Should raise ServiceNotFoundError for missing service."""
        with pytest.raises(ServiceNotFoundError):
            registry.get("nonexistent")
    
    def test_unregister_service(self, registry, sample_service):
        """Should unregister a service."""
        registry.register(sample_service)
        result = registry.unregister("test-001")
        
        assert result is True
        assert len(registry.get_all()) == 0
    
    def test_unregister_nonexistent(self, registry):
        """Should return False when unregistering nonexistent service."""
        result = registry.unregister("nonexistent")
        assert result is False
    
    def test_get_by_job_id(self, registry, sample_service):
        """Should find service by SLURM job ID."""
        sample_service.slurm_job_id = "12345"
        registry.register(sample_service)
        
        service = registry.get_by_job_id("12345")
        
        assert service is not None
        assert service.id == "test-001"
    
    def test_get_by_job_id_not_found(self, registry, sample_service):
        """Should return None for nonexistent job ID."""
        registry.register(sample_service)
        
        service = registry.get_by_job_id("99999")
        
        assert service is None
    
    def test_get_running_services(self, registry, sample_server_recipe):
        """Should filter running services."""
        # Create multiple services with different statuses
        service1 = ServiceInstance(
            id="run-001",
            recipe_name="test",
            recipe=sample_server_recipe,
            status=ServiceStatus.RUNNING,
        )
        service2 = ServiceInstance(
            id="stop-001",
            recipe_name="test",
            recipe=sample_server_recipe,
            status=ServiceStatus.STOPPED,
        )
        service3 = ServiceInstance(
            id="run-002",
            recipe_name="test",
            recipe=sample_server_recipe,
            status=ServiceStatus.RUNNING,
        )
        
        registry.register(service1)
        registry.register(service2)
        registry.register(service3)
        
        running = registry.get_running()
        
        assert len(running) == 2
        assert all(s.is_running() for s in running)
    
    def test_update_status(self, registry, sample_service):
        """Should update service status."""
        registry.register(sample_service)
        
        result = registry.update_status("test-001", ServiceStatus.RUNNING)
        
        assert result is True
        service = registry.get("test-001")
        assert service.status == ServiceStatus.RUNNING
        assert service.started_at is not None
    
    def test_update_status_with_error(self, registry, sample_service):
        """Should set error message when updating to error status."""
        registry.register(sample_service)
        
        registry.update_status("test-001", ServiceStatus.ERROR, "Something failed")
        
        service = registry.get("test-001")
        assert service.status == ServiceStatus.ERROR
        assert service.error_message == "Something failed"
    
    def test_get_by_recipe(self, registry, sample_server_recipe):
        """Should filter services by recipe name."""
        service1 = ServiceInstance(
            id="s1",
            recipe_name="recipe-a",
            recipe=sample_server_recipe,
        )
        service2 = ServiceInstance(
            id="s2",
            recipe_name="recipe-b",
            recipe=sample_server_recipe,
        )
        service3 = ServiceInstance(
            id="s3",
            recipe_name="recipe-a",
            recipe=sample_server_recipe,
        )
        
        registry.register(service1)
        registry.register(service2)
        registry.register(service3)
        
        recipe_a_services = registry.get_by_recipe("recipe-a")
        
        assert len(recipe_a_services) == 2
    
    def test_persistence(self, tmp_path, sample_server_recipe):
        """Should persist and reload state."""
        persistence_path = tmp_path / "services.json"
        
        # Create registry and add service
        registry1 = ServiceRegistry(persistence_path=persistence_path)
        service = ServiceInstance(
            id="persist-001",
            recipe_name="test",
            recipe=sample_server_recipe,
            status=ServiceStatus.RUNNING,
        )
        registry1.register(service)
        
        # Create new registry with same persistence path
        registry2 = ServiceRegistry(persistence_path=persistence_path)
        
        # Should have loaded the service
        assert len(registry2.get_all()) == 1
        loaded_service = registry2.get("persist-001")
        assert loaded_service.recipe_name == "test"
    
    def test_cleanup_stale(self, registry, sample_server_recipe):
        """Should remove stale stopped services."""
        # Create a stopped service with old timestamp
        old_service = ServiceInstance(
            id="old-001",
            recipe_name="test",
            recipe=sample_server_recipe,
            status=ServiceStatus.STOPPED,
        )
        old_service.stopped_at = datetime.now() - timedelta(hours=48)
        registry.register(old_service)
        
        # Create a recent stopped service
        recent_service = ServiceInstance(
            id="recent-001",
            recipe_name="test",
            recipe=sample_server_recipe,
            status=ServiceStatus.STOPPED,
        )
        recent_service.stopped_at = datetime.now() - timedelta(hours=1)
        registry.register(recent_service)
        
        # Cleanup with 24 hour threshold
        removed = registry.cleanup_stale(max_age_hours=24)
        
        assert removed == 1
        assert len(registry.get_all()) == 1


class TestRunRegistry:
    """Tests for RunRegistry class."""
    
    @pytest.fixture
    def registry(self, tmp_path):
        """Create a run registry with temp persistence."""
        return RunRegistry(persistence_path=tmp_path / "runs.json")
    
    @pytest.fixture
    def sample_run(self, sample_client_recipe):
        """Create a sample client run."""
        return ClientRun(
            id="run-001",
            recipe_name="test-client",
            recipe=sample_client_recipe,
            status=RunStatus.SUBMITTED,
        )
    
    def test_register_run(self, registry, sample_run):
        """Should register a new run."""
        result = registry.register(sample_run)
        
        assert result is True
        assert len(registry.get_all()) == 1
    
    def test_get_run(self, registry, sample_run):
        """Should retrieve a registered run."""
        registry.register(sample_run)
        
        run = registry.get("run-001")
        
        assert run.id == "run-001"
        assert run.recipe_name == "test-client"
    
    def test_get_run_not_found(self, registry):
        """Should raise ClientNotFoundError for missing run."""
        with pytest.raises(ClientNotFoundError):
            registry.get("nonexistent")
    
    def test_get_active_runs(self, registry, sample_client_recipe):
        """Should filter active runs."""
        run1 = ClientRun(
            id="active-001",
            recipe_name="test",
            recipe=sample_client_recipe,
            status=RunStatus.RUNNING,
        )
        run2 = ClientRun(
            id="done-001",
            recipe_name="test",
            recipe=sample_client_recipe,
            status=RunStatus.COMPLETED,
        )
        run3 = ClientRun(
            id="active-002",
            recipe_name="test",
            recipe=sample_client_recipe,
            status=RunStatus.QUEUED,
        )
        
        registry.register(run1)
        registry.register(run2)
        registry.register(run3)
        
        active = registry.get_active()
        
        assert len(active) == 2
        assert all(r.is_active() for r in active)
    
    def test_update_status(self, registry, sample_run):
        """Should update run status."""
        registry.register(sample_run)
        
        registry.update_status("run-001", RunStatus.RUNNING)
        
        run = registry.get("run-001")
        assert run.status == RunStatus.RUNNING
        assert run.started_at is not None
    
    def test_completion_timestamp(self, registry, sample_run):
        """Should set completion timestamp when finishing."""
        registry.register(sample_run)
        
        registry.update_status("run-001", RunStatus.COMPLETED)
        
        run = registry.get("run-001")
        assert run.completed_at is not None
