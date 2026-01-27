"""
Registry for tracking running services and client runs.

Provides thread-safe storage and retrieval of service instances,
client runs, and monitor instances.
"""

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from inferbench.core.config import get_config
from inferbench.core.exceptions import ServiceNotFoundError, ClientNotFoundError
from inferbench.core.models import (
    ServiceInstance,
    ServiceStatus,
    ClientRun,
    RunStatus,
    MonitorInstance,
)
from inferbench.utils.logging import get_logger

logger = get_logger(__name__)


class ServiceRegistry:
    """
    Registry for tracking running service instances.
    
    Thread-safe storage with optional persistence to disk.
    """
    
    def __init__(self, persistence_path: Optional[Path] = None):
        """
        Initialize the service registry.
        
        Args:
            persistence_path: Optional path to persist registry state
        """
        self._services: dict[str, ServiceInstance] = {}
        self._lock = threading.RLock()
        self._persistence_path = persistence_path
        
        # Load persisted state if available
        if persistence_path and persistence_path.exists():
            self._load_state()
    
    def register(self, service: ServiceInstance) -> bool:
        """
        Register a new service instance.
        
        Args:
            service: Service instance to register
            
        Returns:
            True if registered successfully
        """
        with self._lock:
            if service.id in self._services:
                logger.warning(f"Service {service.id} already registered, updating")
            
            self._services[service.id] = service
            logger.info(f"Registered service: {service.id} ({service.recipe_name})")
            self._persist_state()
            return True
    
    def unregister(self, service_id: str) -> bool:
        """
        Remove a service from the registry.
        
        Args:
            service_id: ID of service to remove
            
        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if service_id not in self._services:
                logger.warning(f"Service {service_id} not found in registry")
                return False
            
            del self._services[service_id]
            logger.info(f"Unregistered service: {service_id}")
            self._persist_state()
            return True
    
    def get(self, service_id: str) -> ServiceInstance:
        """
        Get a service by ID.
        
        Args:
            service_id: Service ID to look up
            
        Returns:
            Service instance
            
        Raises:
            ServiceNotFoundError: If service not found
        """
        with self._lock:
            if service_id not in self._services:
                raise ServiceNotFoundError(service_id)
            return self._services[service_id]
    
    def get_by_job_id(self, job_id: str) -> Optional[ServiceInstance]:
        """
        Get a service by SLURM job ID.
        
        Args:
            job_id: SLURM job ID
            
        Returns:
            Service instance or None if not found
        """
        with self._lock:
            for service in self._services.values():
                if service.slurm_job_id == job_id:
                    return service
            return None
    
    def get_all(self) -> list[ServiceInstance]:
        """Get all registered services."""
        with self._lock:
            return list(self._services.values())
    
    def get_running(self) -> list[ServiceInstance]:
        """Get all running services."""
        with self._lock:
            return [s for s in self._services.values() if s.is_running()]
    
    def get_by_recipe(self, recipe_name: str) -> list[ServiceInstance]:
        """Get all services using a specific recipe."""
        with self._lock:
            return [s for s in self._services.values() if s.recipe_name == recipe_name]
    
    def update_status(
        self, 
        service_id: str, 
        status: ServiceStatus,
        error_message: Optional[str] = None
    ) -> bool:
        """
        Update the status of a service.
        
        Args:
            service_id: Service ID
            status: New status
            error_message: Optional error message
            
        Returns:
            True if updated successfully
        """
        with self._lock:
            if service_id not in self._services:
                logger.warning(f"Cannot update status: service {service_id} not found")
                return False
            
            service = self._services[service_id]
            old_status = service.status
            service.status = status
            
            if error_message:
                service.error_message = error_message
            
            if status == ServiceStatus.RUNNING and not service.started_at:
                service.started_at = datetime.now()
            elif status in [ServiceStatus.STOPPED, ServiceStatus.ERROR]:
                service.stopped_at = datetime.now()
            
            logger.info(f"Service {service_id} status: {old_status} -> {status}")
            self._persist_state()
            return True
    
    def update_node(self, service_id: str, node: str) -> bool:
        """Update the node where a service is running."""
        with self._lock:
            if service_id not in self._services:
                return False
            self._services[service_id].node = node
            self._persist_state()
            return True
    
    def update_endpoints(self, service_id: str, endpoints: dict[str, str]) -> bool:
        """Update the endpoints for a service."""
        with self._lock:
            if service_id not in self._services:
                return False
            self._services[service_id].endpoints = endpoints
            self._persist_state()
            return True
    
    def cleanup_stale(self, max_age_hours: int = 24) -> int:
        """
        Remove stale stopped/error services older than max_age_hours.
        
        Returns:
            Number of services removed
        """
        with self._lock:
            now = datetime.now()
            stale_ids = []
            
            for service_id, service in self._services.items():
                if service.status in [ServiceStatus.STOPPED, ServiceStatus.ERROR]:
                    if service.stopped_at:
                        age = (now - service.stopped_at).total_seconds() / 3600
                        if age > max_age_hours:
                            stale_ids.append(service_id)
            
            for service_id in stale_ids:
                del self._services[service_id]
            
            if stale_ids:
                logger.info(f"Cleaned up {len(stale_ids)} stale services")
                self._persist_state()
            
            return len(stale_ids)
    
    def _persist_state(self) -> None:
        """Persist registry state to disk."""
        if not self._persistence_path:
            return
        
        try:
            self._persistence_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                service_id: service.model_dump(mode="json")
                for service_id, service in self._services.items()
            }
            with open(self._persistence_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to persist registry state: {e}")
    
    def _load_state(self) -> None:
        """Load registry state from disk."""
        if not self._persistence_path or not self._persistence_path.exists():
            return
        
        try:
            with open(self._persistence_path, "r") as f:
                data = json.load(f)
            
            for service_id, service_data in data.items():
                try:
                    self._services[service_id] = ServiceInstance(**service_data)
                except Exception as e:
                    logger.warning(f"Failed to load service {service_id}: {e}")
            
            logger.info(f"Loaded {len(self._services)} services from persistent state")
        except Exception as e:
            logger.error(f"Failed to load registry state: {e}")


class RunRegistry:
    """
    Registry for tracking client benchmark runs.
    
    Thread-safe storage with optional persistence.
    """
    
    def __init__(self, persistence_path: Optional[Path] = None):
        """Initialize the run registry."""
        self._runs: dict[str, ClientRun] = {}
        self._lock = threading.RLock()
        self._persistence_path = persistence_path
        
        if persistence_path and persistence_path.exists():
            self._load_state()
    
    def register(self, run: ClientRun) -> bool:
        """Register a new client run."""
        with self._lock:
            if run.id in self._runs:
                logger.warning(f"Run {run.id} already registered, updating")
            
            self._runs[run.id] = run
            logger.info(f"Registered run: {run.id} ({run.recipe_name})")
            self._persist_state()
            return True
    
    def unregister(self, run_id: str) -> bool:
        """Remove a run from the registry."""
        with self._lock:
            if run_id not in self._runs:
                return False
            del self._runs[run_id]
            self._persist_state()
            return True
    
    def get(self, run_id: str) -> ClientRun:
        """Get a run by ID."""
        with self._lock:
            if run_id not in self._runs:
                raise ClientNotFoundError(run_id)
            return self._runs[run_id]
    
    def get_all(self) -> list[ClientRun]:
        """Get all registered runs."""
        with self._lock:
            return list(self._runs.values())
    
    def get_active(self) -> list[ClientRun]:
        """Get all active runs."""
        with self._lock:
            return [r for r in self._runs.values() if r.is_active()]
    
    def update_status(
        self, 
        run_id: str, 
        status: RunStatus,
        error_message: Optional[str] = None
    ) -> bool:
        """Update the status of a run."""
        with self._lock:
            if run_id not in self._runs:
                return False
            
            run = self._runs[run_id]
            run.status = status
            
            if error_message:
                run.error_message = error_message
            
            if status == RunStatus.RUNNING and not run.started_at:
                run.started_at = datetime.now()
            elif status in [RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELED]:
                run.completed_at = datetime.now()
            
            self._persist_state()
            return True
    
    def _persist_state(self) -> None:
        """Persist registry state to disk."""
        if not self._persistence_path:
            return
        
        try:
            self._persistence_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                run_id: run.model_dump(mode="json")
                for run_id, run in self._runs.items()
            }
            with open(self._persistence_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to persist run registry state: {e}")
    
    def _load_state(self) -> None:
        """Load registry state from disk."""
        if not self._persistence_path or not self._persistence_path.exists():
            return
        
        try:
            with open(self._persistence_path, "r") as f:
                data = json.load(f)
            
            for run_id, run_data in data.items():
                try:
                    self._runs[run_id] = ClientRun(**run_data)
                except Exception as e:
                    logger.warning(f"Failed to load run {run_id}: {e}")
        except Exception as e:
            logger.error(f"Failed to load run registry state: {e}")


# Global registry instances
_service_registry: Optional[ServiceRegistry] = None
_run_registry: Optional[RunRegistry] = None


def get_service_registry() -> ServiceRegistry:
    """Get the global service registry instance."""
    global _service_registry
    if _service_registry is None:
        config = get_config()
        persistence_path = config.logs_dir / "service_registry.json"
        _service_registry = ServiceRegistry(persistence_path)
    return _service_registry


def get_run_registry() -> RunRegistry:
    """Get the global run registry instance."""
    global _run_registry
    if _run_registry is None:
        config = get_config()
        persistence_path = config.logs_dir / "run_registry.json"
        _run_registry = RunRegistry(persistence_path)
    return _run_registry
