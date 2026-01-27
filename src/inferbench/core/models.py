"""
Pydantic models for InferBench Framework.

Defines data models for recipes, services, clients, and monitoring
with full validation support.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional
import uuid

from pydantic import BaseModel, Field, field_validator, model_validator


# =============================================================================
# Enums
# =============================================================================

class ServiceStatus(str, Enum):
    """Status of a service instance."""
    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    UNKNOWN = "unknown"


class RunStatus(str, Enum):
    """Status of a client run."""
    SUBMITTED = "submitted"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    UNKNOWN = "unknown"


class RecipeType(str, Enum):
    """Type of recipe."""
    SERVER = "server"
    CLIENT = "client"
    MONITOR = "monitor"
    BENCHMARK = "benchmark"


# =============================================================================
# Resource Models
# =============================================================================

class ResourceSpec(BaseModel):
    """Resource requirements for SLURM jobs."""
    
    nodes: int = Field(default=1, ge=1, description="Number of nodes")
    gpus: int = Field(default=0, ge=0, description="Number of GPUs per node")
    gpu_type: Optional[str] = Field(default=None, description="GPU type (e.g., a100)")
    cpus_per_task: int = Field(default=4, ge=1, description="CPUs per task")
    memory: str = Field(default="16G", description="Memory per node (e.g., 16G, 32G)")
    time: str = Field(default="01:00:00", description="Time limit (HH:MM:SS)")
    partition: str = Field(default="gpu", description="SLURM partition")
    
    @field_validator("memory")
    @classmethod
    def validate_memory(cls, v: str) -> str:
        """Validate memory format."""
        v = v.upper()
        if not any(v.endswith(suffix) for suffix in ["G", "M", "K", "GB", "MB", "KB"]):
            raise ValueError("Memory must end with G, M, K (e.g., 16G, 32GB)")
        return v
    
    @field_validator("time")
    @classmethod
    def validate_time(cls, v: str) -> str:
        """Validate time format."""
        parts = v.split(":")
        if len(parts) not in [2, 3]:
            raise ValueError("Time must be in HH:MM:SS or MM:SS format")
        return v


class PortSpec(BaseModel):
    """Port specification for a service."""
    
    name: str = Field(description="Port name (e.g., api, metrics)")
    port: int = Field(ge=1, le=65535, description="Port number")
    protocol: str = Field(default="http", description="Protocol (http, https, grpc)")


class NetworkSpec(BaseModel):
    """Network configuration for a service."""
    
    ports: list[PortSpec] = Field(default_factory=list, description="Port mappings")


class HealthCheckSpec(BaseModel):
    """Health check configuration."""
    
    enabled: bool = Field(default=True, description="Whether health check is enabled")
    endpoint: str = Field(default="/health", description="Health check endpoint")
    port: int = Field(default=8000, ge=1, le=65535, description="Port for health check")
    interval: int = Field(default=30, ge=5, description="Check interval in seconds")
    timeout: int = Field(default=10, ge=1, description="Timeout in seconds")
    retries: int = Field(default=3, ge=1, description="Number of retries")
    initial_delay: int = Field(default=30, ge=0, description="Initial delay before first check")


class MetricsSpec(BaseModel):
    """Metrics endpoint configuration."""
    
    enabled: bool = Field(default=True, description="Whether metrics are enabled")
    endpoint: str = Field(default="/metrics", description="Metrics endpoint")
    port: int = Field(default=8000, ge=1, le=65535, description="Metrics port")
    type: str = Field(default="prometheus", description="Metrics format type")
    exporter: Optional[str] = Field(default=None, description="External exporter name")


class ContainerSpec(BaseModel):
    """Container configuration."""
    
    image: str = Field(description="Path to container image (.sif file)")
    runtime: str = Field(default="apptainer", description="Container runtime")
    binds: list[str] = Field(default_factory=list, description="Bind mount paths")
    
    @field_validator("runtime")
    @classmethod
    def validate_runtime(cls, v: str) -> str:
        """Validate container runtime."""
        valid = ["apptainer", "singularity"]
        if v.lower() not in valid:
            raise ValueError(f"Runtime must be one of: {valid}")
        return v.lower()


# =============================================================================
# Recipe Models
# =============================================================================

class BaseRecipe(BaseModel):
    """Base class for all recipes."""
    
    name: str = Field(description="Recipe name")
    type: RecipeType = Field(description="Recipe type")
    description: Optional[str] = Field(default=None, description="Recipe description")
    version: str = Field(default="1.0.0", description="Recipe version")
    labels: dict[str, str] = Field(default_factory=dict, description="Labels for organization")


class ServerRecipe(BaseRecipe):
    """Recipe for deploying a server service."""
    
    type: RecipeType = Field(default=RecipeType.SERVER)
    container: ContainerSpec = Field(description="Container configuration")
    resources: ResourceSpec = Field(default_factory=ResourceSpec, description="Resource requirements")
    network: NetworkSpec = Field(default_factory=NetworkSpec, description="Network configuration")
    environment: dict[str, str] = Field(default_factory=dict, description="Environment variables")
    command: Optional[str] = Field(default=None, description="Command to run in container")
    post_start: list[str] = Field(default_factory=list, description="Commands to run after start")
    healthcheck: HealthCheckSpec = Field(default_factory=HealthCheckSpec, description="Health check config")
    metrics: MetricsSpec = Field(default_factory=MetricsSpec, description="Metrics config")
    
    def get_primary_port(self) -> Optional[int]:
        """Get the primary service port."""
        if self.network.ports:
            return self.network.ports[0].port
        return None


class ClientRecipe(BaseRecipe):
    """Recipe for running a benchmark client."""
    
    type: RecipeType = Field(default=RecipeType.CLIENT)
    container: Optional[ContainerSpec] = Field(default=None, description="Container config (optional)")
    resources: ResourceSpec = Field(default_factory=ResourceSpec, description="Resource requirements")
    target: dict[str, Any] = Field(default_factory=dict, description="Target service configuration")
    workload: dict[str, Any] = Field(default_factory=dict, description="Workload configuration")
    environment: dict[str, str] = Field(default_factory=dict, description="Environment variables")
    command: Optional[str] = Field(default=None, description="Command to run")
    output: dict[str, Any] = Field(default_factory=dict, description="Output configuration")


class MonitorRecipe(BaseRecipe):
    """Recipe for monitoring configuration."""
    
    type: RecipeType = Field(default=RecipeType.MONITOR)
    targets: list[str] = Field(default_factory=list, description="Target services to monitor")
    prometheus: dict[str, Any] = Field(default_factory=dict, description="Prometheus configuration")
    grafana: dict[str, Any] = Field(default_factory=dict, description="Grafana configuration")
    scrape_interval: int = Field(default=15, ge=5, description="Scrape interval in seconds")
    retention: str = Field(default="7d", description="Data retention period")


# =============================================================================
# Instance Models
# =============================================================================

class ServiceInstance(BaseModel):
    """Represents a running service instance."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8], description="Service ID")
    recipe_name: str = Field(description="Name of the recipe used")
    recipe: ServerRecipe = Field(description="Full recipe configuration")
    status: ServiceStatus = Field(default=ServiceStatus.PENDING, description="Current status")
    slurm_job_id: Optional[str] = Field(default=None, description="SLURM job ID")
    node: Optional[str] = Field(default=None, description="Compute node name")
    endpoints: dict[str, str] = Field(default_factory=dict, description="Service endpoints")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation time")
    started_at: Optional[datetime] = Field(default=None, description="Start time")
    stopped_at: Optional[datetime] = Field(default=None, description="Stop time")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    
    def get_endpoint(self, port_name: str = "api") -> Optional[str]:
        """Get endpoint URL for a named port."""
        return self.endpoints.get(port_name)
    
    def is_running(self) -> bool:
        """Check if service is running."""
        return self.status == ServiceStatus.RUNNING


class ClientRun(BaseModel):
    """Represents a client benchmark run."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8], description="Run ID")
    recipe_name: str = Field(description="Name of the recipe used")
    recipe: ClientRecipe = Field(description="Full recipe configuration")
    status: RunStatus = Field(default=RunStatus.SUBMITTED, description="Current status")
    slurm_job_id: Optional[str] = Field(default=None, description="SLURM job ID")
    node: Optional[str] = Field(default=None, description="Compute node name")
    target_service_id: Optional[str] = Field(default=None, description="Target service being tested")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation time")
    started_at: Optional[datetime] = Field(default=None, description="Start time")
    completed_at: Optional[datetime] = Field(default=None, description="Completion time")
    results_path: Optional[str] = Field(default=None, description="Path to results")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    
    def is_active(self) -> bool:
        """Check if run is still active."""
        return self.status in [RunStatus.SUBMITTED, RunStatus.QUEUED, RunStatus.RUNNING]


class MonitorInstance(BaseModel):
    """Represents a running monitor instance."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8], description="Monitor ID")
    recipe_name: str = Field(description="Name of the recipe used")
    recipe: MonitorRecipe = Field(description="Full recipe configuration")
    status: ServiceStatus = Field(default=ServiceStatus.PENDING, description="Current status")
    prometheus_job_id: Optional[str] = Field(default=None, description="Prometheus SLURM job ID")
    grafana_job_id: Optional[str] = Field(default=None, description="Grafana SLURM job ID")
    prometheus_url: Optional[str] = Field(default=None, description="Prometheus URL")
    grafana_url: Optional[str] = Field(default=None, description="Grafana URL")
    targets: list[str] = Field(default_factory=list, description="Monitored service IDs")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation time")
