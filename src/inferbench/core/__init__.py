"""
Core infrastructure for InferBench Framework.

Provides configuration management, data models, recipe loading,
registries, and orchestration components.
"""

from inferbench.core.config import Config, get_config, set_config
from inferbench.core.exceptions import (
    InferBenchError,
    RecipeError,
    RecipeNotFoundError,
    RecipeValidationError,
    RecipeParseError,
    ServiceError,
    ServiceNotFoundError,
    ServiceStartError,
    ServiceStopError,
    ClientError,
    ClientNotFoundError,
    MonitorError,
    OrchestratorError,
    SlurmError,
    ContainerError,
)
from inferbench.core.models import (
    ServiceStatus,
    RunStatus,
    RecipeType,
    ResourceSpec,
    PortSpec,
    NetworkSpec,
    HealthCheckSpec,
    MetricsSpec,
    ContainerSpec,
    ServerRecipe,
    ClientRecipe,
    MonitorRecipe,
    ServiceInstance,
    ClientRun,
    MonitorInstance,
)
from inferbench.core.recipe_loader import RecipeLoader, get_recipe_loader
from inferbench.core.registry import (
    ServiceRegistry,
    RunRegistry,
    get_service_registry,
    get_run_registry,
)
from inferbench.core.slurm import SlurmOrchestrator, SlurmJobInfo, get_slurm_orchestrator
from inferbench.core.apptainer import ApptainerRuntime, get_apptainer_runtime

__all__ = [
    # Config
    "Config",
    "get_config",
    "set_config",
    # Exceptions
    "InferBenchError",
    "RecipeError",
    "RecipeNotFoundError",
    "RecipeValidationError",
    "RecipeParseError",
    "ServiceError",
    "ServiceNotFoundError",
    "ServiceStartError",
    "ServiceStopError",
    "ClientError",
    "ClientNotFoundError",
    "MonitorError",
    "OrchestratorError",
    "SlurmError",
    "ContainerError",
    # Models
    "ServiceStatus",
    "RunStatus",
    "RecipeType",
    "ResourceSpec",
    "PortSpec",
    "NetworkSpec",
    "HealthCheckSpec",
    "MetricsSpec",
    "ContainerSpec",
    "ServerRecipe",
    "ClientRecipe",
    "MonitorRecipe",
    "ServiceInstance",
    "ClientRun",
    "MonitorInstance",
    # Recipe Loader
    "RecipeLoader",
    "get_recipe_loader",
    # Registries
    "ServiceRegistry",
    "RunRegistry",
    "get_service_registry",
    "get_run_registry",
    # SLURM
    "SlurmOrchestrator",
    "SlurmJobInfo",
    "get_slurm_orchestrator",
    # Apptainer
    "ApptainerRuntime",
    "get_apptainer_runtime",
]
