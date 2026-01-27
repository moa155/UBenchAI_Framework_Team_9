"""
Client Manager for InferBench Framework.

Handles benchmark client execution, workload management,
and results collection on SLURM-managed HPC clusters.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

from inferbench.core.config import get_config
from inferbench.core.exceptions import (
    ClientRunError,
    ClientNotFoundError,
    RecipeNotFoundError,
    ServiceNotFoundError,
)
from inferbench.core.models import (
    ClientRun,
    RunStatus,
    ClientRecipe,
    RecipeType,
    ServiceInstance,
)
from inferbench.core.recipe_loader import RecipeLoader, get_recipe_loader
from inferbench.core.registry import RunRegistry, get_run_registry, get_service_registry
from inferbench.core.slurm import SlurmOrchestrator, get_slurm_orchestrator
from inferbench.core.apptainer import ApptainerRuntime, get_apptainer_runtime
from inferbench.utils.logging import get_logger

logger = get_logger(__name__)


class ClientManager:
    """
    Manages benchmark client execution on HPC clusters.
    
    Coordinates workload submission, monitoring, and results collection.
    """
    
    def __init__(
        self,
        recipe_loader: Optional[RecipeLoader] = None,
        registry: Optional[RunRegistry] = None,
        orchestrator: Optional[SlurmOrchestrator] = None,
        runtime: Optional[ApptainerRuntime] = None,
    ):
        """
        Initialize the client manager.
        
        Args:
            recipe_loader: Recipe loader instance
            registry: Run registry instance
            orchestrator: SLURM orchestrator instance
            runtime: Apptainer runtime instance
        """
        self.config = get_config()
        self.recipe_loader = recipe_loader or get_recipe_loader()
        self.registry = registry or get_run_registry()
        self.service_registry = get_service_registry()
        self.orchestrator = orchestrator or get_slurm_orchestrator()
        self.runtime = runtime or get_apptainer_runtime()
        
        # Ensure required directories exist
        self._setup_directories()
        
        logger.info("ClientManager initialized")
    
    def _setup_directories(self) -> None:
        """Create required directories for client operations."""
        dirs = [
            self.config.logs_dir / "clients",
            self.config.results_dir / "clients",
            Path("/tmp/inferbench/clients"),
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _get_work_dir(self, run_id: str) -> Path:
        """Get the working directory for a client run."""
        work_dir = self.config.logs_dir / "clients" / run_id
        work_dir.mkdir(parents=True, exist_ok=True)
        return work_dir
    
    def _get_results_dir(self, run_id: str) -> Path:
        """Get the results directory for a client run."""
        results_dir = self.config.results_dir / "clients" / run_id
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir
    
    def _resolve_target_endpoint(
        self, 
        recipe: ClientRecipe, 
        target_service_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Resolve the target service endpoint.
        
        Args:
            recipe: Client recipe with target configuration
            target_service_id: Optional specific service ID to target
            
        Returns:
            Target endpoint URL or None
        """
        # If target looks like a URL, use it directly
        if target_service_id and target_service_id.startswith(("http://", "https://")):
            logger.info(f"Using direct target URL: {target_service_id}")
            return target_service_id

        # If specific service ID provided, look it up
        if target_service_id:
            try:
                service = self.service_registry.get(target_service_id)
                endpoint = service.get_endpoint("api")
                if endpoint:
                    logger.info(f"Resolved target endpoint from service {target_service_id}: {endpoint}")
                    return endpoint
            except ServiceNotFoundError:
                logger.warning(f"Target service {target_service_id} not found")
        
        # Check recipe target configuration
        target_config = recipe.target
        
        # Direct URL specified
        if "url" in target_config:
            return target_config["url"]
        
        # Endpoint file specified (written by server)
        if "endpoint_file" in target_config:
            endpoint_file = Path(target_config["endpoint_file"])
            if endpoint_file.exists():
                try:
                    content = endpoint_file.read_text()
                    for line in content.split("\n"):
                        if line.startswith("ENDPOINT="):
                            return line.split("=", 1)[1].strip()
                except Exception as e:
                    logger.warning(f"Failed to read endpoint file: {e}")
        
        return None
    
    def _build_client_command(
        self, 
        recipe: ClientRecipe,
        target_endpoint: Optional[str],
        results_dir: Path,
    ) -> str:
        """Build the command to run the benchmark client."""
        
        # If recipe has a container, use it
        if recipe.container:
            container_cmd = self.runtime.generate_exec_script(
                container_spec=recipe.container,
                resources=recipe.resources,
                command=recipe.command or "echo 'No command specified'",
                environment=recipe.environment,
            )
            return container_cmd
        
        # Otherwise, build a Python-based benchmark script
        workload = recipe.workload
        workload_type = workload.get("type", "simple")
        
        # Generate benchmark script based on workload type
        if workload_type in ["open-loop", "closed-loop", "stress-test"]:
            return self._build_http_benchmark_script(
                recipe, target_endpoint, results_dir
            )
        else:
            # Default: just run the command if specified
            return recipe.command or "echo 'No benchmark command specified'"
    
    def _build_http_benchmark_script(
        self,
        recipe: ClientRecipe,
        target_endpoint: Optional[str],
        results_dir: Path,
    ) -> str:
        """Build an HTTP benchmark script."""
        workload = recipe.workload
        
        # Extract workload parameters
        pattern = workload.get("pattern", {})
        rate = pattern.get("rate", 10)
        duration = pattern.get("duration", 60)
        
        request_config = workload.get("request", {})
        endpoint_path = request_config.get("endpoint", "/v1/completions")
        method = request_config.get("method", "POST")
        
        # Get prompts from dataset
        dataset = workload.get("dataset", {})
        prompts = dataset.get("prompts", ["Hello, how are you?"])
        
        # Build the benchmark script
        script = f'''#!/usr/bin/env python3
"""
InferBench Benchmark Client
Generated for: {recipe.name}
"""

import json
import time
import random
import statistics
import os
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    import urllib.request
    import urllib.error
    HAS_HTTPX = False

# Configuration
TARGET_ENDPOINT = "{target_endpoint or 'http://localhost:8000'}"
ENDPOINT_PATH = "{endpoint_path}"
REQUEST_RATE = {rate}  # requests per second
DURATION = {duration}  # seconds
METHOD = "{method}"
RESULTS_DIR = Path("{results_dir}")

PROMPTS = {json.dumps(prompts)}

def make_request_httpx(client, prompt):
    """Make a request using httpx."""
    url = TARGET_ENDPOINT + ENDPOINT_PATH
    
    body = {{
        "model": os.environ.get("MODEL_NAME", "tinyllama"),
        "prompt": prompt,
        "max_tokens": 100,
        "temperature": 0.7
    }}
    
    start = time.perf_counter()
    try:
        if METHOD == "POST":
            response = client.post(url, json=body, timeout=120)
        else:
            response = client.get(url, timeout=120)
        
        latency = time.perf_counter() - start
        return {{
            "success": response.status_code == 200,
            "status_code": response.status_code,
            "latency": latency,
            "error": None
        }}
    except Exception as e:
        latency = time.perf_counter() - start
        return {{
            "success": False,
            "status_code": 0,
            "latency": latency,
            "error": str(e)
        }}

def make_request_urllib(prompt):
    """Make a request using urllib (fallback)."""
    import urllib.request
    import urllib.error
    
    url = TARGET_ENDPOINT + ENDPOINT_PATH
    
    body = json.dumps({{
        "model": os.environ.get("MODEL_NAME", "tinyllama"),
        "prompt": prompt,
        "max_tokens": 100,
        "temperature": 0.7
    }}).encode()
    
    start = time.perf_counter()
    try:
        req = urllib.request.Request(
            url, 
            data=body if METHOD == "POST" else None,
            headers={{"Content-Type": "application/json"}}
        )
        with urllib.request.urlopen(req, timeout=120) as response:
            latency = time.perf_counter() - start
            return {{
                "success": response.status == 200,
                "status_code": response.status,
                "latency": latency,
                "error": None
            }}
    except Exception as e:
        latency = time.perf_counter() - start
        return {{
            "success": False,
            "status_code": 0,
            "latency": latency,
            "error": str(e)
        }}

def run_benchmark():
    """Run the benchmark."""
    print("=" * 60)
    print("InferBench Benchmark Client")
    print("=" * 60)
    print(f"Target: {{TARGET_ENDPOINT}}")
    print(f"Endpoint: {{ENDPOINT_PATH}}")
    print(f"Rate: {{REQUEST_RATE}} req/s")
    print(f"Duration: {{DURATION}} seconds")
    print("=" * 60)
    
    results = []
    start_time = time.time()
    request_interval = 1.0 / REQUEST_RATE
    request_count = 0
    
    if HAS_HTTPX:
        client = httpx.Client()
        make_request = lambda p: make_request_httpx(client, p)
    else:
        client = None
        make_request = make_request_urllib
    
    try:
        while time.time() - start_time < DURATION:
            prompt = random.choice(PROMPTS)
            result = make_request(prompt)
            results.append(result)
            request_count += 1
            
            if request_count % 10 == 0:
                elapsed = time.time() - start_time
                actual_rate = request_count / elapsed
                print(f"Progress: {{request_count}} requests, {{elapsed:.1f}}s elapsed, {{actual_rate:.1f}} req/s")
            
            # Sleep to maintain rate
            time.sleep(request_interval)
    
    finally:
        if client:
            client.close()
    
    # Calculate statistics
    latencies = [r["latency"] for r in results]
    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]
    
    total_time = time.time() - start_time
    
    stats = {{
        "benchmark": "{recipe.name}",
        "timestamp": datetime.now().isoformat(),
        "target": TARGET_ENDPOINT,
        "config": {{
            "rate": REQUEST_RATE,
            "duration": DURATION,
        }},
        "summary": {{
            "total_requests": len(results),
            "successful_requests": len(successes),
            "failed_requests": len(failures),
            "success_rate": len(successes) / len(results) * 100 if results else 0,
            "total_time_seconds": total_time,
            "actual_throughput": len(results) / total_time if total_time > 0 else 0,
        }},
        "latency": {{
            "min": min(latencies) if latencies else 0,
            "max": max(latencies) if latencies else 0,
            "mean": statistics.mean(latencies) if latencies else 0,
            "median": statistics.median(latencies) if latencies else 0,
            "p95": sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 20 else max(latencies) if latencies else 0,
            "p99": sorted(latencies)[int(len(latencies) * 0.99)] if len(latencies) > 100 else max(latencies) if latencies else 0,
        }},
        "errors": [r["error"] for r in failures[:10]]  # First 10 errors
    }}
    
    # Print summary
    print()
    print("=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Total Requests: {{stats['summary']['total_requests']}}")
    print(f"Successful: {{stats['summary']['successful_requests']}}")
    print(f"Failed: {{stats['summary']['failed_requests']}}")
    print(f"Success Rate: {{stats['summary']['success_rate']:.2f}}%")
    print(f"Throughput: {{stats['summary']['actual_throughput']:.2f}} req/s")
    print()
    print("Latency (seconds):")
    print(f"  Min: {{stats['latency']['min']:.4f}}")
    print(f"  Max: {{stats['latency']['max']:.4f}}")
    print(f"  Mean: {{stats['latency']['mean']:.4f}}")
    print(f"  Median: {{stats['latency']['median']:.4f}}")
    print(f"  P95: {{stats['latency']['p95']:.4f}}")
    print(f"  P99: {{stats['latency']['p99']:.4f}}")
    print("=" * 60)
    
    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_file = RESULTS_DIR / "benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Results saved to: {{results_file}}")
    
    # Also save raw results
    raw_file = RESULTS_DIR / "raw_results.json"
    with open(raw_file, "w") as f:
        json.dump(results, f, indent=2)
    
    return stats

if __name__ == "__main__":
    run_benchmark()
'''
        return script
    
    def run_client(
        self,
        recipe_name: str,
        target_service_id: Optional[str] = None,
        config_overrides: Optional[dict] = None,
        wait_for_completion: bool = False,
        timeout: int = 3600,
    ) -> ClientRun:
        """
        Run a benchmark client from a recipe.
        
        Args:
            recipe_name: Name of the client recipe
            target_service_id: Optional service ID to benchmark
            config_overrides: Optional configuration overrides
            wait_for_completion: Whether to wait for completion
            timeout: Timeout in seconds
            
        Returns:
            ClientRun object
            
        Raises:
            RecipeNotFoundError: If recipe doesn't exist
            ClientRunError: If client fails to start
        """
        logger.info(f"Starting client with recipe: {recipe_name}")
        
        # Load recipe
        try:
            recipe = self.recipe_loader.load_client(recipe_name)
        except RecipeNotFoundError:
            raise
        except Exception as e:
            raise ClientRunError(recipe_name, f"Failed to load recipe: {e}")
        
        # Apply overrides
        if config_overrides:
            recipe = self._apply_overrides(recipe, config_overrides)
        
        # Create client run instance
        run = ClientRun(
            recipe_name=recipe_name,
            recipe=recipe,
            status=RunStatus.SUBMITTED,
            target_service_id=target_service_id,
        )
        
        # Register the run
        self.registry.register(run)
        
        try:
            # Get working directories
            work_dir = self._get_work_dir(run.id)
            results_dir = self._get_results_dir(run.id)
            
            # Resolve target endpoint
            target_endpoint = self._resolve_target_endpoint(recipe, target_service_id)
            
            # Build client command
            client_command = self._build_client_command(recipe, target_endpoint, results_dir)
            
            # Save benchmark script for debugging
            script_file = work_dir / "benchmark_script.py"
            script_file.write_text(client_command)
            
            # Generate batch script
            batch_script = self.orchestrator.generate_batch_script(
                job_name=f"inferbench-client-{recipe_name}-{run.id}",
                command=f"python3 {script_file}",
                resources=recipe.resources,
                environment=recipe.environment,
                output_dir=work_dir,
            )
            
            # Save batch script
            batch_file = work_dir / "job.sh"
            batch_file.write_text(batch_script)
            
            # Submit job
            self.registry.update_status(run.id, RunStatus.QUEUED)
            job_id = self.orchestrator.submit_job(
                script_content=batch_script,
                script_name=f"client_{run.id}.sh",
                work_dir=work_dir,
            )
            
            # Update run with job ID
            run.slurm_job_id = job_id
            run.results_path = str(results_dir)
            self.registry.register(run)
            
            logger.info(f"Client run {run.id} submitted as SLURM job {job_id}")
            
            # Wait for completion if requested
            if wait_for_completion:
                self._wait_for_completion(run, timeout)
            
            return run
            
        except Exception as e:
            self.registry.update_status(run.id, RunStatus.FAILED, str(e))
            raise ClientRunError(recipe_name, str(e))
    
    def _apply_overrides(self, recipe: ClientRecipe, overrides: dict) -> ClientRecipe:
        """Apply configuration overrides to a recipe."""
        recipe_dict = recipe.model_dump()
        
        for key, value in overrides.items():
            if key in recipe_dict:
                if isinstance(recipe_dict[key], dict) and isinstance(value, dict):
                    recipe_dict[key].update(value)
                else:
                    recipe_dict[key] = value
        
        return ClientRecipe(**recipe_dict)
    
    def _wait_for_completion(self, run: ClientRun, timeout: int) -> bool:
        """Wait for a client run to complete."""
        logger.info(f"Waiting for run {run.id} to complete (timeout: {timeout}s)")
        
        start_time = time.time()
        check_interval = 10
        
        while time.time() - start_time < timeout:
            # Check SLURM job status
            slurm_status = self.orchestrator.get_job_status(run.slurm_job_id)
            
            from inferbench.core.models import ServiceStatus
            
            if slurm_status == ServiceStatus.STOPPED:
                # Job completed - check if results exist
                results_file = Path(run.results_path) / "benchmark_results.json"
                if results_file.exists():
                    self.registry.update_status(run.id, RunStatus.COMPLETED)
                    run.status = RunStatus.COMPLETED
                    logger.info(f"Run {run.id} completed successfully")
                    return True
                else:
                    self.registry.update_status(run.id, RunStatus.FAILED, "No results generated")
                    return False
            
            elif slurm_status == ServiceStatus.ERROR:
                self.registry.update_status(run.id, RunStatus.FAILED, "SLURM job failed")
                return False
            
            elif slurm_status == ServiceStatus.RUNNING:
                self.registry.update_status(run.id, RunStatus.RUNNING)
                run.status = RunStatus.RUNNING
            
            time.sleep(check_interval)
        
        # Timeout
        self.registry.update_status(run.id, RunStatus.FAILED, "Timeout waiting for completion")
        return False
    
    def stop_run(self, run_id: str) -> bool:
        """
        Stop a running client.
        
        Args:
            run_id: Run ID to stop
            
        Returns:
            True if stopped successfully
        """
        logger.info(f"Stopping run: {run_id}")
        
        try:
            run = self.registry.get(run_id)
        except ClientNotFoundError:
            raise
        
        if not run.is_active():
            logger.warning(f"Run {run_id} is not active")
            return True
        
        # Cancel SLURM job
        if run.slurm_job_id:
            self.orchestrator.cancel_job(run.slurm_job_id)
        
        self.registry.update_status(run.id, RunStatus.CANCELED)
        logger.info(f"Run {run_id} canceled")
        return True
    
    def get_run_status(self, run_id: str) -> ClientRun:
        """Get the status of a client run."""
        run = self.registry.get(run_id)
        
        # Update from SLURM if active
        if run.is_active() and run.slurm_job_id:
            from inferbench.core.models import ServiceStatus
            slurm_status = self.orchestrator.get_job_status(run.slurm_job_id)
            
            if slurm_status == ServiceStatus.RUNNING:
                self.registry.update_status(run.id, RunStatus.RUNNING)
                run.status = RunStatus.RUNNING
            elif slurm_status == ServiceStatus.STOPPED:
                # Check results
                if run.results_path:
                    results_file = Path(run.results_path) / "benchmark_results.json"
                    if results_file.exists():
                        self.registry.update_status(run.id, RunStatus.COMPLETED)
                        run.status = RunStatus.COMPLETED
                    else:
                        self.registry.update_status(run.id, RunStatus.FAILED)
                        run.status = RunStatus.FAILED
            elif slurm_status == ServiceStatus.ERROR:
                self.registry.update_status(run.id, RunStatus.FAILED)
                run.status = RunStatus.FAILED
        
        return run
    
    def get_run_results(self, run_id: str) -> Optional[dict]:
        """
        Get the results of a completed run.
        
        Args:
            run_id: Run ID
            
        Returns:
            Results dictionary or None
        """
        run = self.registry.get(run_id)
        
        if not run.results_path:
            return None
        
        results_file = Path(run.results_path) / "benchmark_results.json"
        if not results_file.exists():
            return None
        
        try:
            with open(results_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read results: {e}")
            return None
    
    def list_runs(self, active_only: bool = False) -> list[ClientRun]:
        """List all client runs."""
        if active_only:
            return self.registry.get_active()
        return self.registry.get_all()
    
    def list_available_recipes(self) -> list[str]:
        """List available client recipes."""
        return self.recipe_loader.list_recipes(RecipeType.CLIENT)
    
    def get_run_logs(self, run_id: str, lines: int = 100, log_type: str = "output") -> str:
        """Get logs for a client run."""
        run = self.registry.get(run_id)
        work_dir = self._get_work_dir(run.id)
        
        if log_type == "error":
            return self.orchestrator.get_job_error(run.slurm_job_id, work_dir, lines)
        else:
            return self.orchestrator.get_job_output(run.slurm_job_id, work_dir, lines)


# Global client manager instance
_manager: Optional[ClientManager] = None


def get_client_manager() -> ClientManager:
    """Get the global client manager instance."""
    global _manager
    if _manager is None:
        _manager = ClientManager()
    return _manager
