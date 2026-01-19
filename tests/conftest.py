"""
Pytest configuration and fixtures for InferBench Framework tests.
"""

import pytest
from pathlib import Path

from inferbench.core.config import Config
from inferbench.core.models import (
    ServerRecipe,
    ClientRecipe,
    MonitorRecipe,
    ContainerSpec,
    ResourceSpec,
    ServiceInstance,
    ClientRun,
    ServiceStatus,
    RunStatus,
)


@pytest.fixture
def temp_config(tmp_path):
    """Create a temporary configuration for testing."""
    return Config(
        base_dir=tmp_path,
        config_dir=tmp_path / "config",
        recipes_dir=tmp_path / "recipes",
        results_dir=tmp_path / "results",
        logs_dir=tmp_path / "logs",
    )


@pytest.fixture
def sample_recipe_dir(tmp_path):
    """Create a sample recipes directory with test recipes."""
    recipes_dir = tmp_path / "recipes"
    servers_dir = recipes_dir / "servers"
    servers_dir.mkdir(parents=True)
    
    # Create a sample recipe
    sample_recipe = servers_dir / "test-server.yaml"
    sample_recipe.write_text("""
name: test-server
type: server
description: Test server for unit tests
container:
  image: /path/to/test.sif
resources:
  nodes: 1
  gpus: 1
  memory: 16G
""")
    
    return recipes_dir


@pytest.fixture
def sample_container_spec():
    """Create a sample container specification."""
    return ContainerSpec(
        image="/path/to/image.sif",
        runtime="apptainer",
        binds=["/data:/data:ro", "/models:/models"]
    )


@pytest.fixture
def sample_resources():
    """Create sample resource requirements."""
    return ResourceSpec(
        nodes=1,
        gpus=1,
        memory="32G",
        time="02:00:00",
        partition="gpu"
    )


@pytest.fixture
def sample_server_recipe(sample_container_spec, sample_resources):
    """Create a sample server recipe."""
    return ServerRecipe(
        name="test-server",
        description="Test server recipe",
        container=sample_container_spec,
        resources=sample_resources,
        environment={"MODEL": "test-model"},
        command="python -m server"
    )


@pytest.fixture
def sample_client_recipe():
    """Create a sample client recipe."""
    return ClientRecipe(
        name="test-client",
        description="Test client recipe",
        resources=ResourceSpec(memory="16G", gpus=0),
        target={"url": "http://localhost:8000"},
        workload={"type": "stress-test"}
    )


@pytest.fixture
def sample_service_instance(sample_server_recipe):
    """Create a sample service instance."""
    return ServiceInstance(
        id="test-svc-001",
        recipe_name="test-server",
        recipe=sample_server_recipe,
        status=ServiceStatus.RUNNING,
        slurm_job_id="12345678",
        node="mel2091",
        endpoints={"api": "http://mel2091:8000"}
    )


@pytest.fixture
def sample_client_run(sample_client_recipe):
    """Create a sample client run."""
    return ClientRun(
        id="test-run-001",
        recipe_name="test-client",
        recipe=sample_client_recipe,
        status=RunStatus.RUNNING,
        slurm_job_id="87654321"
    )
