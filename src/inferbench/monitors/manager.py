"""
Monitor Manager for InferBench Framework.

Handles Prometheus and Grafana deployment for real-time metrics
collection and visualization on SLURM-managed HPC clusters.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from inferbench.core.config import get_config
from inferbench.core.exceptions import (
    MonitorStartError,
    MonitorError,
    ServiceNotFoundError,
)
from inferbench.core.models import (
    MonitorInstance,
    MonitorRecipe,
    ServiceStatus,
    RecipeType,
)
from inferbench.core.recipe_loader import RecipeLoader, get_recipe_loader
from inferbench.core.registry import get_service_registry
from inferbench.core.slurm import SlurmOrchestrator, get_slurm_orchestrator
from inferbench.utils.logging import get_logger

logger = get_logger(__name__)


class MonitorManager:
    """
    Manages monitoring stack deployment on HPC clusters.
    
    Coordinates Prometheus and Grafana for metrics collection
    and visualization.
    """
    
    def __init__(
        self,
        recipe_loader: Optional[RecipeLoader] = None,
        orchestrator: Optional[SlurmOrchestrator] = None,
    ):
        """
        Initialize the monitor manager.
        
        Args:
            recipe_loader: Recipe loader instance
            orchestrator: SLURM orchestrator instance
        """
        self.config = get_config()
        self.recipe_loader = recipe_loader or get_recipe_loader()
        self.service_registry = get_service_registry()
        self.orchestrator = orchestrator or get_slurm_orchestrator()
        
        # Monitor instances storage
        self._monitors: dict[str, MonitorInstance] = {}
        
        # Setup directories
        self._setup_directories()
        
        logger.info("MonitorManager initialized")
    
    def _setup_directories(self) -> None:
        """Create required directories for monitoring."""
        dirs = [
            self.config.logs_dir / "monitors",
            self.config.results_dir / "monitors",
            Path("/tmp/inferbench/monitors"),
            Path("/tmp/inferbench/prometheus"),
            Path("/tmp/inferbench/prometheus/targets"),
            Path("/tmp/inferbench/grafana"),
            Path("/tmp/inferbench/grafana/provisioning"),
            Path("/tmp/inferbench/grafana/dashboards"),
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _get_work_dir(self, monitor_id: str) -> Path:
        """Get the working directory for a monitor instance."""
        work_dir = self.config.logs_dir / "monitors" / monitor_id
        work_dir.mkdir(parents=True, exist_ok=True)
        return work_dir
    
    def _resolve_targets(self, target_ids: list[str]) -> list[dict]:
        """
        Resolve target service IDs to scrape targets.
        
        Args:
            target_ids: List of service IDs to monitor
            
        Returns:
            List of target configurations for Prometheus
        """
        targets = []
        
        for target_id in target_ids:
            try:
                service = self.service_registry.get(target_id)
                
                if service.status != ServiceStatus.RUNNING:
                    logger.warning(f"Service {target_id} is not running, skipping")
                    continue
                
                # Get metrics endpoint
                metrics_config = service.recipe.metrics
                if not metrics_config.enabled:
                    logger.warning(f"Service {target_id} has metrics disabled")
                    continue
                
                # Build target
                node = service.node
                port = metrics_config.port
                
                if node and port:
                    targets.append({
                        "targets": [f"{node}:{port}"],
                        "labels": {
                            "service_id": service.id,
                            "service_name": service.recipe_name,
                            "job": f"inferbench_{service.recipe_name}",
                        }
                    })
                    logger.info(f"Added monitoring target: {node}:{port} for {service.recipe_name}")
                    
            except ServiceNotFoundError:
                logger.warning(f"Service {target_id} not found")
            except Exception as e:
                logger.error(f"Error resolving target {target_id}: {e}")
        
        return targets
    
    def _generate_prometheus_config(
        self,
        recipe: MonitorRecipe,
        targets: list[dict],
        work_dir: Path,
    ) -> str:
        """Generate Prometheus configuration file."""
        
        # Write targets file for file-based service discovery
        targets_file = work_dir / "targets.json"
        targets_file.write_text(json.dumps(targets, indent=2))
        
        # Generate prometheus.yml
        config = f"""# InferBench Prometheus Configuration
# Generated: {datetime.now().isoformat()}

global:
  scrape_interval: {recipe.scrape_interval}s
  evaluation_interval: {recipe.scrape_interval}s

scrape_configs:
  # Self-monitoring
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  # InferBench services (file-based discovery)
  - job_name: 'inferbench_services'
    file_sd_configs:
      - files:
          - '{targets_file}'
        refresh_interval: 30s

  # Node exporter (if available)
  - job_name: 'node'
    static_configs:
      - targets: ['localhost:9100']
"""
        
        # Add any custom scrape configs from recipe
        prometheus_config = recipe.prometheus
        if prometheus_config.get("scrape_configs"):
            for scrape_config in prometheus_config["scrape_configs"]:
                job_name = scrape_config.get("job_name", "custom")
                static_configs = scrape_config.get("static_configs", [])
                
                config += f"""
  - job_name: '{job_name}'
    static_configs:
"""
                for sc in static_configs:
                    targets_str = ", ".join(f"'{t}'" for t in sc.get("targets", []))
                    config += f"      - targets: [{targets_str}]\n"
        
        return config
    
    def _generate_grafana_datasource(self, prometheus_url: str) -> str:
        """Generate Grafana datasource provisioning config."""
        return f"""apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: {prometheus_url}
    isDefault: true
    editable: true
"""
    
    def _generate_grafana_dashboard_config(self) -> str:
        """Generate Grafana dashboard provisioning config."""
        return """apiVersion: 1

providers:
  - name: 'InferBench Dashboards'
    orgId: 1
    folder: 'InferBench'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 30
    options:
      path: /tmp/inferbench/grafana/dashboards
"""
    
    def _generate_vllm_dashboard(self) -> dict:
        """Generate a vLLM metrics dashboard for Grafana."""
        return {
            "annotations": {"list": []},
            "editable": True,
            "fiscalYearStartMonth": 0,
            "graphTooltip": 0,
            "id": None,
            "links": [],
            "liveNow": False,
            "panels": [
                {
                    "datasource": {"type": "prometheus", "uid": "prometheus"},
                    "fieldConfig": {
                        "defaults": {"color": {"mode": "palette-classic"}, "unit": "reqps"},
                        "overrides": []
                    },
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                    "id": 1,
                    "options": {"legend": {"displayMode": "list"}, "tooltip": {"mode": "single"}},
                    "targets": [
                        {
                            "expr": "rate(vllm:request_success_total[5m])",
                            "legendFormat": "Success Rate",
                            "refId": "A"
                        }
                    ],
                    "title": "Request Rate",
                    "type": "timeseries"
                },
                {
                    "datasource": {"type": "prometheus", "uid": "prometheus"},
                    "fieldConfig": {
                        "defaults": {"color": {"mode": "palette-classic"}, "unit": "percentunit"},
                        "overrides": []
                    },
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                    "id": 2,
                    "options": {"legend": {"displayMode": "list"}, "tooltip": {"mode": "single"}},
                    "targets": [
                        {
                            "expr": "vllm:gpu_cache_usage_perc",
                            "legendFormat": "GPU Cache Usage",
                            "refId": "A"
                        }
                    ],
                    "title": "GPU Cache Usage",
                    "type": "timeseries"
                },
                {
                    "datasource": {"type": "prometheus", "uid": "prometheus"},
                    "fieldConfig": {
                        "defaults": {"color": {"mode": "palette-classic"}, "unit": "s"},
                        "overrides": []
                    },
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
                    "id": 3,
                    "options": {"legend": {"displayMode": "list"}, "tooltip": {"mode": "single"}},
                    "targets": [
                        {
                            "expr": "histogram_quantile(0.50, rate(vllm:e2e_request_latency_seconds_bucket[5m]))",
                            "legendFormat": "P50",
                            "refId": "A"
                        },
                        {
                            "expr": "histogram_quantile(0.95, rate(vllm:e2e_request_latency_seconds_bucket[5m]))",
                            "legendFormat": "P95",
                            "refId": "B"
                        },
                        {
                            "expr": "histogram_quantile(0.99, rate(vllm:e2e_request_latency_seconds_bucket[5m]))",
                            "legendFormat": "P99",
                            "refId": "C"
                        }
                    ],
                    "title": "Request Latency",
                    "type": "timeseries"
                },
                {
                    "datasource": {"type": "prometheus", "uid": "prometheus"},
                    "fieldConfig": {
                        "defaults": {"color": {"mode": "palette-classic"}, "unit": "short"},
                        "overrides": []
                    },
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
                    "id": 4,
                    "options": {"legend": {"displayMode": "list"}, "tooltip": {"mode": "single"}},
                    "targets": [
                        {
                            "expr": "vllm:num_requests_running",
                            "legendFormat": "Running Requests",
                            "refId": "A"
                        },
                        {
                            "expr": "vllm:num_requests_waiting",
                            "legendFormat": "Waiting Requests",
                            "refId": "B"
                        }
                    ],
                    "title": "Active Requests",
                    "type": "timeseries"
                }
            ],
            "refresh": "5s",
            "schemaVersion": 38,
            "style": "dark",
            "tags": ["inferbench", "vllm", "llm"],
            "templating": {"list": []},
            "time": {"from": "now-15m", "to": "now"},
            "timepicker": {},
            "timezone": "",
            "title": "vLLM Metrics",
            "uid": "inferbench-vllm",
            "version": 1,
            "weekStart": ""
        }
    
    def _generate_prometheus_script(
        self,
        recipe: MonitorRecipe,
        config_file: Path,
        work_dir: Path,
    ) -> str:
        """Generate the Prometheus startup script."""
        port = recipe.prometheus.get("port", 9090)
        retention = recipe.retention
        
        return f'''#!/bin/bash
set -e

echo "========================================="
echo "InferBench Prometheus Starting"
echo "========================================="
echo "Config: {config_file}"
echo "Port: {port}"
echo "Retention: {retention}"
echo "========================================="

# Check if prometheus is available
if ! command -v prometheus &> /dev/null; then
    echo "Prometheus not found, trying to download..."
    cd /tmp
    curl -sLO https://github.com/prometheus/prometheus/releases/download/v2.45.0/prometheus-2.45.0.linux-amd64.tar.gz
    tar xzf prometheus-2.45.0.linux-amd64.tar.gz
    export PATH="/tmp/prometheus-2.45.0.linux-amd64:$PATH"
fi

# Start Prometheus
prometheus \\
    --config.file={config_file} \\
    --storage.tsdb.path={work_dir}/data \\
    --storage.tsdb.retention.time={retention} \\
    --web.listen-address=:${port} \\
    --web.enable-lifecycle &

PROMETHEUS_PID=$!
echo "Prometheus PID: $PROMETHEUS_PID"

# Wait for startup
sleep 10

if ! kill -0 $PROMETHEUS_PID 2>/dev/null; then
    echo "Prometheus failed to start"
    exit 1
fi

COMPUTE_NODE=$(hostname)
echo ""
echo "========================================="
echo "Prometheus Ready!"
echo "========================================="
echo "URL: http://$COMPUTE_NODE:{port}"
echo "Targets: http://$COMPUTE_NODE:{port}/targets"
echo ""
echo "SSH Tunnel: ssh -L {port}:$COMPUTE_NODE:{port} user@login.lxp.lu"
echo "========================================="

# Write endpoint file
echo "PROMETHEUS_URL=http://$COMPUTE_NODE:{port}" > {work_dir}/prometheus_endpoint.txt

# Keep alive
wait $PROMETHEUS_PID
'''
    
    def _generate_grafana_script(
        self,
        recipe: MonitorRecipe,
        prometheus_url: str,
        work_dir: Path,
    ) -> str:
        """Generate the Grafana startup script."""
        grafana_config = recipe.grafana
        port = grafana_config.get("port", 3000)
        admin_password = grafana_config.get("admin_password", "admin")
        
        return f'''#!/bin/bash
set -e

echo "========================================="
echo "InferBench Grafana Starting"
echo "========================================="
echo "Port: {port}"
echo "Prometheus: {prometheus_url}"
echo "========================================="

# Setup provisioning directories
PROV_DIR={work_dir}/provisioning
mkdir -p $PROV_DIR/datasources
mkdir -p $PROV_DIR/dashboards
mkdir -p {work_dir}/dashboards

# Check if grafana-server is available
if ! command -v grafana-server &> /dev/null; then
    echo "Grafana not found in PATH"
    echo "Please ensure Grafana is installed or use a container"
    exit 1
fi

# Start Grafana
GF_PATHS_PROVISIONING=$PROV_DIR \\
GF_SECURITY_ADMIN_PASSWORD={admin_password} \\
GF_SERVER_HTTP_PORT={port} \\
GF_PATHS_DATA={work_dir}/data \\
grafana-server \\
    --homepath=/usr/share/grafana \\
    --config=/etc/grafana/grafana.ini &

GRAFANA_PID=$!
echo "Grafana PID: $GRAFANA_PID"

# Wait for startup
sleep 15

if ! kill -0 $GRAFANA_PID 2>/dev/null; then
    echo "Grafana failed to start"
    exit 1
fi

COMPUTE_NODE=$(hostname)
echo ""
echo "========================================="
echo "Grafana Ready!"
echo "========================================="
echo "URL: http://$COMPUTE_NODE:{port}"
echo "Username: admin"
echo "Password: {admin_password}"
echo ""
echo "SSH Tunnel: ssh -L {port}:$COMPUTE_NODE:{port} user@login.lxp.lu"
echo "========================================="

# Write endpoint file
echo "GRAFANA_URL=http://$COMPUTE_NODE:{port}" > {work_dir}/grafana_endpoint.txt

# Keep alive
wait $GRAFANA_PID
'''
    
    def start_monitor(
        self,
        recipe_name: str,
        target_ids: Optional[list[str]] = None,
        wait_for_ready: bool = True,
        timeout: int = 120,
    ) -> MonitorInstance:
        """
        Start a monitoring stack from a recipe.
        
        Args:
            recipe_name: Name of the monitor recipe
            target_ids: List of service IDs to monitor
            wait_for_ready: Whether to wait for services to be ready
            timeout: Timeout in seconds
            
        Returns:
            MonitorInstance object
        """
        logger.info(f"Starting monitor with recipe: {recipe_name}")
        
        # Load recipe
        try:
            recipe = self.recipe_loader.load_monitor(recipe_name)
        except Exception as e:
            raise MonitorStartError(recipe_name, f"Failed to load recipe: {e}")
        
        # Create monitor instance
        monitor = MonitorInstance(
            recipe_name=recipe_name,
            recipe=recipe,
            status=ServiceStatus.PENDING,
            targets=target_ids or [],
        )
        
        try:
            work_dir = self._get_work_dir(monitor.id)
            
            # Resolve targets
            targets = []
            if target_ids:
                targets = self._resolve_targets(target_ids)
            
            # Generate Prometheus config
            prometheus_config = self._generate_prometheus_config(recipe, targets, work_dir)
            prometheus_config_file = work_dir / "prometheus.yml"
            prometheus_config_file.write_text(prometheus_config)
            
            # Generate Prometheus startup script
            prometheus_script = self._generate_prometheus_script(
                recipe, prometheus_config_file, work_dir
            )
            
            # Generate batch script for Prometheus
            from inferbench.core.models import ResourceSpec
            prometheus_resources = ResourceSpec(
                nodes=1, gpus=0, cpus_per_task=2,
                memory="4G", time="04:00:00", partition="batch"
            )
            
            prometheus_batch = self.orchestrator.generate_batch_script(
                job_name=f"inferbench-prometheus-{monitor.id}",
                command=f"bash -c '{prometheus_script}'",
                resources=prometheus_resources,
                environment={},
                output_dir=work_dir,
            )
            
            # Submit Prometheus job
            prometheus_job_id = self.orchestrator.submit_job(
                script_content=prometheus_batch,
                script_name=f"prometheus_{monitor.id}.sh",
                work_dir=work_dir,
            )
            
            monitor.prometheus_job_id = prometheus_job_id
            logger.info(f"Prometheus job submitted: {prometheus_job_id}")
            
            # Wait for Prometheus to be ready
            if wait_for_ready:
                time.sleep(15)  # Give it time to start
                
                # Get node
                node = self.orchestrator.get_job_node(prometheus_job_id)
                if node:
                    prometheus_port = recipe.prometheus.get("port", 9090)
                    monitor.prometheus_url = f"http://{node}:{prometheus_port}"
                    logger.info(f"Prometheus URL: {monitor.prometheus_url}")
            
            # Update status
            monitor.status = ServiceStatus.RUNNING
            self._monitors[monitor.id] = monitor
            
            logger.info(f"Monitor {monitor.id} started successfully")
            return monitor
            
        except Exception as e:
            monitor.status = ServiceStatus.ERROR
            raise MonitorStartError(recipe_name, str(e))
    
    def stop_monitor(self, monitor_id: str) -> bool:
        """
        Stop a running monitor stack.
        
        Args:
            monitor_id: Monitor ID to stop
            
        Returns:
            True if stopped successfully
        """
        logger.info(f"Stopping monitor: {monitor_id}")
        
        if monitor_id not in self._monitors:
            raise MonitorError(f"Monitor {monitor_id} not found")
        
        monitor = self._monitors[monitor_id]
        
        # Cancel Prometheus job
        if monitor.prometheus_job_id:
            self.orchestrator.cancel_job(monitor.prometheus_job_id)
        
        # Cancel Grafana job
        if monitor.grafana_job_id:
            self.orchestrator.cancel_job(monitor.grafana_job_id)
        
        monitor.status = ServiceStatus.STOPPED
        logger.info(f"Monitor {monitor_id} stopped")
        
        return True
    
    def get_monitor_status(self, monitor_id: str) -> MonitorInstance:
        """Get the status of a monitor instance."""
        if monitor_id not in self._monitors:
            raise MonitorError(f"Monitor {monitor_id} not found")
        
        monitor = self._monitors[monitor_id]
        
        # Update status from SLURM
        if monitor.prometheus_job_id:
            status = self.orchestrator.get_job_status(monitor.prometheus_job_id)
            monitor.status = status
            
            # Update URL if running
            if status == ServiceStatus.RUNNING and not monitor.prometheus_url:
                node = self.orchestrator.get_job_node(monitor.prometheus_job_id)
                if node:
                    port = monitor.recipe.prometheus.get("port", 9090)
                    monitor.prometheus_url = f"http://{node}:{port}"
        
        return monitor
    
    def list_monitors(self, running_only: bool = False) -> list[MonitorInstance]:
        """List all monitor instances."""
        monitors = list(self._monitors.values())
        
        if running_only:
            return [m for m in monitors if m.status == ServiceStatus.RUNNING]
        
        return monitors
    
    def list_available_recipes(self) -> list[str]:
        """List available monitor recipes."""
        return self.recipe_loader.list_recipes(RecipeType.MONITOR)
    
    def add_target(self, monitor_id: str, service_id: str) -> bool:
        """
        Add a service target to an existing monitor.
        
        Args:
            monitor_id: Monitor ID
            service_id: Service ID to add
            
        Returns:
            True if added successfully
        """
        if monitor_id not in self._monitors:
            raise MonitorError(f"Monitor {monitor_id} not found")
        
        monitor = self._monitors[monitor_id]
        
        # Resolve target
        targets = self._resolve_targets([service_id])
        if not targets:
            return False
        
        # Update targets file
        work_dir = self._get_work_dir(monitor_id)
        targets_file = work_dir / "targets.json"
        
        existing_targets = []
        if targets_file.exists():
            existing_targets = json.loads(targets_file.read_text())
        
        existing_targets.extend(targets)
        targets_file.write_text(json.dumps(existing_targets, indent=2))
        
        monitor.targets.append(service_id)
        logger.info(f"Added target {service_id} to monitor {monitor_id}")
        
        return True
    
    def remove_target(self, monitor_id: str, service_id: str) -> bool:
        """Remove a service target from a monitor."""
        if monitor_id not in self._monitors:
            raise MonitorError(f"Monitor {monitor_id} not found")
        
        monitor = self._monitors[monitor_id]
        
        if service_id in monitor.targets:
            monitor.targets.remove(service_id)
            
            # Update targets file
            work_dir = self._get_work_dir(monitor_id)
            targets_file = work_dir / "targets.json"
            
            if targets_file.exists():
                existing = json.loads(targets_file.read_text())
                updated = [t for t in existing if t.get("labels", {}).get("service_id") != service_id]
                targets_file.write_text(json.dumps(updated, indent=2))
            
            logger.info(f"Removed target {service_id} from monitor {monitor_id}")
            return True
        
        return False


# Global monitor manager instance
_manager: Optional[MonitorManager] = None


def get_monitor_manager() -> MonitorManager:
    """Get the global monitor manager instance."""
    global _manager
    if _manager is None:
        _manager = MonitorManager()
    return _manager
