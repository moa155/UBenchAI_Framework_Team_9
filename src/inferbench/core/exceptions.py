"""
Custom exceptions for InferBench Framework.

Provides a hierarchy of exceptions for different error conditions
that can occur during framework operation.
"""


class InferBenchError(Exception):
    """Base exception for all InferBench errors."""
    
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


# =============================================================================
# Recipe Errors
# =============================================================================

class RecipeError(InferBenchError):
    """Base exception for recipe-related errors."""
    pass


class RecipeNotFoundError(RecipeError):
    """Raised when a recipe cannot be found."""
    
    def __init__(self, recipe_name: str, recipe_type: str):
        super().__init__(
            f"Recipe '{recipe_name}' of type '{recipe_type}' not found",
            {"recipe_name": recipe_name, "recipe_type": recipe_type}
        )


class RecipeValidationError(RecipeError):
    """Raised when recipe validation fails."""
    
    def __init__(self, recipe_name: str, errors: list[str]):
        super().__init__(
            f"Recipe '{recipe_name}' validation failed: {'; '.join(errors)}",
            {"recipe_name": recipe_name, "errors": errors}
        )


class RecipeParseError(RecipeError):
    """Raised when recipe YAML parsing fails."""
    
    def __init__(self, recipe_path: str, error: str):
        super().__init__(
            f"Failed to parse recipe at '{recipe_path}': {error}",
            {"recipe_path": recipe_path, "parse_error": error}
        )


# =============================================================================
# Service Errors
# =============================================================================

class ServiceError(InferBenchError):
    """Base exception for service-related errors."""
    pass


class ServiceNotFoundError(ServiceError):
    """Raised when a service cannot be found."""
    
    def __init__(self, service_id: str):
        super().__init__(
            f"Service '{service_id}' not found",
            {"service_id": service_id}
        )


class ServiceStartError(ServiceError):
    """Raised when a service fails to start."""
    
    def __init__(self, recipe_name: str, reason: str):
        super().__init__(
            f"Failed to start service '{recipe_name}': {reason}",
            {"recipe_name": recipe_name, "reason": reason}
        )


class ServiceStopError(ServiceError):
    """Raised when a service fails to stop."""
    
    def __init__(self, service_id: str, reason: str):
        super().__init__(
            f"Failed to stop service '{service_id}': {reason}",
            {"service_id": service_id, "reason": reason}
        )


class ServiceHealthCheckError(ServiceError):
    """Raised when a service health check fails."""
    
    def __init__(self, service_id: str, endpoint: str, status_code: int | None = None):
        super().__init__(
            f"Health check failed for service '{service_id}' at '{endpoint}'",
            {"service_id": service_id, "endpoint": endpoint, "status_code": status_code}
        )


# =============================================================================
# Client Errors
# =============================================================================

class ClientError(InferBenchError):
    """Base exception for client-related errors."""
    pass


class ClientRunError(ClientError):
    """Raised when a client run fails."""
    
    def __init__(self, run_id: str, reason: str):
        super().__init__(
            f"Client run '{run_id}' failed: {reason}",
            {"run_id": run_id, "reason": reason}
        )


class ClientNotFoundError(ClientError):
    """Raised when a client run cannot be found."""
    
    def __init__(self, run_id: str):
        super().__init__(
            f"Client run '{run_id}' not found",
            {"run_id": run_id}
        )


# =============================================================================
# Monitor Errors
# =============================================================================

class MonitorError(InferBenchError):
    """Base exception for monitor-related errors."""
    pass


class MonitorStartError(MonitorError):
    """Raised when monitoring fails to start."""
    
    def __init__(self, reason: str):
        super().__init__(
            f"Failed to start monitoring: {reason}",
            {"reason": reason}
        )


class MetricsCollectionError(MonitorError):
    """Raised when metrics collection fails."""
    
    def __init__(self, target: str, reason: str):
        super().__init__(
            f"Failed to collect metrics from '{target}': {reason}",
            {"target": target, "reason": reason}
        )


# =============================================================================
# Orchestrator Errors
# =============================================================================

class OrchestratorError(InferBenchError):
    """Base exception for orchestrator-related errors."""
    pass


class SlurmError(OrchestratorError):
    """Raised when SLURM operations fail."""
    
    def __init__(self, operation: str, reason: str, job_id: str | None = None):
        super().__init__(
            f"SLURM {operation} failed: {reason}",
            {"operation": operation, "reason": reason, "job_id": job_id}
        )


class ContainerError(OrchestratorError):
    """Raised when container operations fail."""
    
    def __init__(self, operation: str, image: str, reason: str):
        super().__init__(
            f"Container {operation} failed for '{image}': {reason}",
            {"operation": operation, "image": image, "reason": reason}
        )


# =============================================================================
# Configuration Errors
# =============================================================================

class ConfigurationError(InferBenchError):
    """Raised when configuration is invalid or missing."""
    
    def __init__(self, setting: str, reason: str):
        super().__init__(
            f"Configuration error for '{setting}': {reason}",
            {"setting": setting, "reason": reason}
        )
