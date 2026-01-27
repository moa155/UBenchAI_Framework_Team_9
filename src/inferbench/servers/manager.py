"""
Server Manager for InferBench Framework.

Handles the complete lifecycle of AI services including deployment,
health checking, and shutdown on SLURM-managed HPC clusters.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from inferbench.core.config import get_config
from inferbench.core.exceptions import (
    ServiceStartError,
    ServiceStopError,
    ServiceNotFoundError,
    ServiceHealthCheckError,
    RecipeNotFoundError,
)
from inferbench.core.models import (
    ServiceInstance,
    ServiceStatus,
    ServerRecipe,
    RecipeType,
)
from inferbench.core.recipe_loader import RecipeLoader, get_recipe_loader
from inferbench.core.registry import ServiceRegistry, get_service_registry
from inferbench.core.slurm import SlurmOrchestrator, get_slurm_orchestrator
from inferbench.core.apptainer import ApptainerRuntime, get_apptainer_runtime
from inferbench.utils.logging import get_logger

logger = get_logger(__name__)


class ServerManager:
    """
    Manages AI service deployment and lifecycle on HPC clusters.
    
    Coordinates between recipe loading, SLURM job submission,
    container execution, and service health monitoring.
    """
    
    def __init__(
        self,
        recipe_loader: Optional[RecipeLoader] = None,
        registry: Optional[ServiceRegistry] = None,
        orchestrator: Optional[SlurmOrchestrator] = None,
        runtime: Optional[ApptainerRuntime] = None,
    ):
        """
        Initialize the server manager.
        
        Args:
            recipe_loader: Recipe loader instance
            registry: Service registry instance
            orchestrator: SLURM orchestrator instance
            runtime: Apptainer runtime instance
        """
        self.config = get_config()
        self.recipe_loader = recipe_loader or get_recipe_loader()
        self.registry = registry or get_service_registry()
        self.orchestrator = orchestrator or get_slurm_orchestrator()
        self.runtime = runtime or get_apptainer_runtime()
        
        # Ensure required directories exist
        self._setup_directories()
        
        logger.info("ServerManager initialized")
    
    def _setup_directories(self) -> None:
        """Create required directories for server operations."""
        dirs = [
            self.config.logs_dir / "servers",
            self.config.results_dir / "servers",
            Path("/tmp/inferbench/servers"),
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _get_work_dir(self, service_id: str) -> Path:
        """Get the working directory for a service."""
        work_dir = self.config.logs_dir / "servers" / service_id
        work_dir.mkdir(parents=True, exist_ok=True)
        return work_dir
    
    def _generate_endpoint_file_content(
        self, 
        service: ServiceInstance,
        node: str,
        port: int
    ) -> str:
        """Generate content for the endpoint file."""
        return f"""# InferBench Service Endpoint
# Generated: {datetime.now().isoformat()}
SERVICE_ID={service.id}
SERVICE_NAME={service.recipe_name}
SLURM_JOB_ID={service.slurm_job_id}
NODE={node}
PORT={port}
ENDPOINT=http://{node}:{port}
"""
    
    def _build_server_command(self, recipe: ServerRecipe) -> str:
        """Build the complete command to run the server."""
        # Generate container exec command
        container_cmd = self.runtime.generate_exec_script(
            container_spec=recipe.container,
            resources=recipe.resources,
            command=recipe.command or "echo 'No command specified'",
            environment=recipe.environment,
        )
        
        return container_cmd
    
    def start_service(
        self,
        recipe_name: str,
        config_overrides: Optional[dict] = None,
        wait_for_ready: bool = True,
        timeout: int = 300,
    ) -> ServiceInstance:
        """
        Start a service from a recipe.
        
        Args:
            recipe_name: Name of the server recipe
            config_overrides: Optional configuration overrides
            wait_for_ready: Whether to wait for service to be ready
            timeout: Timeout in seconds for waiting
            
        Returns:
            ServiceInstance object
            
        Raises:
            RecipeNotFoundError: If recipe doesn't exist
            ServiceStartError: If service fails to start
        """
        logger.info(f"Starting service with recipe: {recipe_name}")
        
        # Load and validate recipe
        try:
            recipe = self.recipe_loader.load_server(recipe_name)
        except RecipeNotFoundError:
            raise
        except Exception as e:
            raise ServiceStartError(recipe_name, f"Failed to load recipe: {e}")
        
        # Apply config overrides
        if config_overrides:
            recipe = self._apply_overrides(recipe, config_overrides)
        
        # Create service instance
        service = ServiceInstance(
            recipe_name=recipe_name,
            recipe=recipe,
            status=ServiceStatus.PENDING,
        )
        
        # Register the service
        self.registry.register(service)
        
        try:
            # Get working directory
            work_dir = self._get_work_dir(service.id)
            
            # Build the server command
            server_command = self._build_server_command(recipe)
            
            # Get the primary port
            primary_port = recipe.get_primary_port() or 8000
            
            # Add endpoint file creation to the command
            endpoint_file = f"/tmp/inferbench/servers/{service.id}_endpoint.txt"
            
            # Build setup commands
            setup_commands = [
                f"mkdir -p /tmp/inferbench/servers",
                f"export SERVICE_ID={service.id}",
                f"export SERVICE_PORT={primary_port}",
                # Write endpoint file with node info
                f'echo "SERVICE_ID={service.id}" > {endpoint_file}',
                f'echo "NODE=$SLURM_NODELIST" >> {endpoint_file}',
                f'echo "PORT={primary_port}" >> {endpoint_file}',
                f'echo "ENDPOINT=http://$SLURM_NODELIST:{primary_port}" >> {endpoint_file}',
            ]
            
            # Add post-start commands if any
            if recipe.post_start:
                setup_commands.extend(recipe.post_start)
            
            # Generate batch script
            batch_script = self.orchestrator.generate_batch_script(
                job_name=f"inferbench-{recipe_name}-{service.id}",
                command=server_command,
                resources=recipe.resources,
                environment=recipe.environment,
                output_dir=work_dir,
                setup_commands=setup_commands,
            )
            
            # Save batch script for debugging
            script_path = work_dir / "job.sh"
            script_path.write_text(batch_script)
            logger.debug(f"Batch script saved to: {script_path}")
            
            # Submit the job
            self.registry.update_status(service.id, ServiceStatus.STARTING)
            job_id = self.orchestrator.submit_job(
                script_content=batch_script,
                script_name=f"server_{service.id}.sh",
                work_dir=work_dir,
            )
            
            # Update service with job ID
            service.slurm_job_id = job_id
            self.registry.register(service)  # Update
            
            logger.info(f"Service {service.id} submitted as SLURM job {job_id}")
            
            # Wait for service to be ready if requested
            if wait_for_ready:
                self._wait_for_ready(service, timeout)
            
            return service
            
        except Exception as e:
            self.registry.update_status(
                service.id, 
                ServiceStatus.ERROR,
                error_message=str(e)
            )
            raise ServiceStartError(recipe_name, str(e))
    
    def _apply_overrides(
        self, 
        recipe: ServerRecipe, 
        overrides: dict
    ) -> ServerRecipe:
        """Apply configuration overrides to a recipe."""
        recipe_dict = recipe.model_dump()
        
        # Deep merge overrides
        for key, value in overrides.items():
            if key in recipe_dict:
                if isinstance(recipe_dict[key], dict) and isinstance(value, dict):
                    recipe_dict[key].update(value)
                else:
                    recipe_dict[key] = value
        
        return ServerRecipe(**recipe_dict)
    
    def _wait_for_ready(
        self, 
        service: ServiceInstance, 
        timeout: int = 300
    ) -> bool:
        """
        Wait for a service to become ready.
        
        Args:
            service: Service instance to wait for
            timeout: Timeout in seconds
            
        Returns:
            True if service is ready
            
        Raises:
            ServiceStartError: If service fails to start within timeout
        """
        logger.info(f"Waiting for service {service.id} to be ready (timeout: {timeout}s)")
        
        start_time = time.time()
        check_interval = 5  # seconds
        
        while time.time() - start_time < timeout:
            # Check SLURM job status
            job_status = self.orchestrator.get_job_status(service.slurm_job_id)
            
            if job_status == ServiceStatus.RUNNING:
                # Get the node
                node = self.orchestrator.get_job_node(service.slurm_job_id)
                if node:
                    service.node = node
                    self.registry.update_node(service.id, node)
                    
                    # Build endpoints
                    primary_port = service.recipe.get_primary_port() or 8000
                    endpoints = {"api": f"http://{node}:{primary_port}"}
                    
                    # Add other ports
                    for port_spec in service.recipe.network.ports:
                        endpoints[port_spec.name] = f"http://{node}:{port_spec.port}"
                    
                    self.registry.update_endpoints(service.id, endpoints)
                    self.registry.update_status(service.id, ServiceStatus.RUNNING)
                    
                    # Update local service object
                    service.status = ServiceStatus.RUNNING
                    service.endpoints = endpoints
                    service.started_at = datetime.now()
                    
                    logger.info(f"Service {service.id} is running on {node}")
                    return True
            
            elif job_status == ServiceStatus.ERROR:
                error_msg = "SLURM job failed"
                self.registry.update_status(service.id, ServiceStatus.ERROR, error_msg)
                raise ServiceStartError(service.recipe_name, error_msg)
            
            elif job_status == ServiceStatus.STOPPED:
                error_msg = "SLURM job completed unexpectedly"
                self.registry.update_status(service.id, ServiceStatus.ERROR, error_msg)
                raise ServiceStartError(service.recipe_name, error_msg)
            
            # Still pending/starting
            logger.debug(f"Service {service.id} status: {job_status}, waiting...")
            time.sleep(check_interval)
        
        # Timeout reached
        error_msg = f"Service did not become ready within {timeout} seconds"
        self.registry.update_status(service.id, ServiceStatus.ERROR, error_msg)
        raise ServiceStartError(service.recipe_name, error_msg)
    
    def stop_service(self, service_id: str, force: bool = False) -> bool:
        """
        Stop a running service.
        
        Args:
            service_id: Service ID or SLURM job ID
            force: Force stop even if graceful shutdown fails
            
        Returns:
            True if stopped successfully
            
        Raises:
            ServiceNotFoundError: If service not found
            ServiceStopError: If stop fails
        """
        logger.info(f"Stopping service: {service_id}")
        
        # Try to find by service ID first
        try:
            service = self.registry.get(service_id)
        except ServiceNotFoundError:
            # Try to find by SLURM job ID
            service = self.registry.get_by_job_id(service_id)
            if not service:
                raise ServiceNotFoundError(service_id)
        
        if service.status in [ServiceStatus.STOPPED, ServiceStatus.ERROR]:
            logger.warning(f"Service {service.id} is already stopped")
            return True
        
        try:
            # Update status
            self.registry.update_status(service.id, ServiceStatus.STOPPING)
            
            # Cancel the SLURM job
            if service.slurm_job_id:
                success = self.orchestrator.cancel_job(service.slurm_job_id)
                if not success and not force:
                    raise ServiceStopError(service.id, "Failed to cancel SLURM job")
            
            # Update final status
            self.registry.update_status(service.id, ServiceStatus.STOPPED)
            
            logger.info(f"Service {service.id} stopped successfully")
            return True
            
        except Exception as e:
            if force:
                self.registry.update_status(service.id, ServiceStatus.STOPPED)
                return True
            raise ServiceStopError(service.id, str(e))
    
    def get_service_status(self, service_id: str) -> ServiceInstance:
        """
        Get the current status of a service.
        
        Args:
            service_id: Service ID or SLURM job ID
            
        Returns:
            Updated ServiceInstance
            
        Raises:
            ServiceNotFoundError: If service not found
        """
        # Try to find by service ID first
        try:
            service = self.registry.get(service_id)
        except ServiceNotFoundError:
            # Try to find by SLURM job ID
            service = self.registry.get_by_job_id(service_id)
            if not service:
                raise ServiceNotFoundError(service_id)
        
        # Update status from SLURM if job is active
        if service.slurm_job_id and service.status not in [
            ServiceStatus.STOPPED, 
            ServiceStatus.ERROR
        ]:
            slurm_status = self.orchestrator.get_job_status(service.slurm_job_id)
            
            if slurm_status != service.status:
                self.registry.update_status(service.id, slurm_status)
                service.status = slurm_status
                
                # Update node if running
                if slurm_status == ServiceStatus.RUNNING:
                    node = self.orchestrator.get_job_node(service.slurm_job_id)
                    if node and node != service.node:
                        service.node = node
                        self.registry.update_node(service.id, node)
        
        return service
    
    def list_services(self, running_only: bool = False) -> list[ServiceInstance]:
        """
        List all services.
        
        Args:
            running_only: If True, only return running services
            
        Returns:
            List of service instances
        """
        if running_only:
            return self.registry.get_running()
        return self.registry.get_all()
    
    def list_available_recipes(self) -> list[str]:
        """List all available server recipes."""
        return self.recipe_loader.list_recipes(RecipeType.SERVER)
    
    def get_service_logs(
        self, 
        service_id: str, 
        lines: int = 100,
        log_type: str = "output"
    ) -> str:
        """
        Get logs for a service.
        
        Args:
            service_id: Service ID
            lines: Number of lines to return
            log_type: Type of log ('output' or 'error')
            
        Returns:
            Log content
        """
        service = self.registry.get(service_id)
        work_dir = self._get_work_dir(service.id)
        
        if log_type == "error":
            return self.orchestrator.get_job_error(
                service.slurm_job_id, 
                work_dir, 
                lines
            )
        else:
            return self.orchestrator.get_job_output(
                service.slurm_job_id, 
                work_dir, 
                lines
            )
    
    def check_health(self, service_id: str) -> dict:
        """
        Check the health of a service.
        
        Args:
            service_id: Service ID
            
        Returns:
            Health check result dict
        """
        service = self.registry.get(service_id)
        
        if service.status != ServiceStatus.RUNNING:
            return {
                "healthy": False,
                "status": service.status.value,
                "message": "Service is not running"
            }
        
        # Get health check config
        healthcheck = service.recipe.healthcheck
        
        if not healthcheck.enabled:
            return {
                "healthy": True,
                "status": "unknown",
                "message": "Health check not enabled"
            }
        
        # Try to perform health check
        try:
            import httpx
            
            endpoint = service.get_endpoint("api")
            if not endpoint:
                return {
                    "healthy": False,
                    "status": "error",
                    "message": "No endpoint available"
                }
            
            health_url = f"{endpoint}{healthcheck.endpoint}"
            
            with httpx.Client(timeout=healthcheck.timeout) as client:
                response = client.get(health_url)
                
                return {
                    "healthy": response.status_code == 200,
                    "status": "healthy" if response.status_code == 200 else "unhealthy",
                    "status_code": response.status_code,
                    "endpoint": health_url,
                }
                
        except Exception as e:
            return {
                "healthy": False,
                "status": "error",
                "message": str(e)
            }


# Global server manager instance
_manager: Optional[ServerManager] = None


def get_server_manager() -> ServerManager:
    """Get the global server manager instance."""
    global _manager
    if _manager is None:
        _manager = ServerManager()
    return _manager
