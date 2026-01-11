"""
Web Interface for UBenchAI Framework.

Provides a Flask-based dashboard for monitoring and managing
AI benchmarking services on MeluXina.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from flask import Flask, render_template, jsonify, request, Response
from flask_cors import CORS

from ubenchai.core.config import get_config
from ubenchai.core.models import ServiceStatus, RunStatus
from ubenchai.servers.manager import get_server_manager
from ubenchai.clients.manager import get_client_manager
from ubenchai.monitors.manager import get_monitor_manager
from ubenchai.logs.manager import get_log_manager
from ubenchai.utils.logging import get_logger

logger = get_logger(__name__)


def create_app(config: Optional[dict] = None) -> Flask:
    """
    Create and configure the Flask application.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured Flask application
    """
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static"),
    )
    
    # Default configuration
    app.config.update(
        SECRET_KEY="ubenchai-secret-key-change-in-production",
        JSON_SORT_KEYS=False,
        JSONIFY_PRETTYPRINT_REGULAR=True,
    )
    
    # Apply custom config
    if config:
        app.config.update(config)
    
    # Enable CORS for API endpoints
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # Register routes
    register_routes(app)
    register_api_routes(app)
    
    logger.info("Flask application created")

    @app.route("/api/analysis/summary", methods=["GET"])
    def api_analysis_summary():
        """Get analysis summary of benchmark results."""
        try:
            # Load results from results directory
            results_dir = Path("results")
            all_results = []
            
            if results_dir.exists():
                for f in results_dir.glob("**/*.json"):
                    try:
                        with open(f) as fp:
                            data = json.load(fp)
                            if isinstance(data, list):
                                all_results.extend(data)
                            else:
                                all_results.append(data)
                    except:
                        pass
            
            # Calculate summary
            models = {}
            for r in all_results:
                model = r.get("model", "unknown")
                if model not in models:
                    models[model] = {"runs": 0, "total_tps": 0, "values": []}
                models[model]["runs"] += 1
                tps = r.get("tokens_per_second", r.get("tps", 0))
                models[model]["total_tps"] += tps
                models[model]["values"].append(tps)
            
            summary = {
                "total_runs": len(all_results),
                "models_tested": len(models),
                "models": {
                    m: {
                        "runs": d["runs"],
                        "avg_throughput": d["total_tps"] / d["runs"] if d["runs"] > 0 else 0,
                        "max_throughput": max(d["values"]) if d["values"] else 0,
                        "min_throughput": min(d["values"]) if d["values"] else 0,
                    }
                    for m, d in models.items()
                }
            }
            
            return jsonify(summary)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/models", methods=["GET"])
    def api_list_models():
        """List available models from Ollama."""
        try:
            import urllib.request
            
            # Try to get Ollama endpoint from running services
            ollama_url = None
            for service in service_registry.get_all():
                if "ollama" in service.recipe_name.lower():
                    ollama_url = service.endpoint
                    break
            
            if not ollama_url:
                return jsonify({"models": [], "error": "No Ollama service running"})
            
            url = f"{ollama_url}/api/tags"
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read())
            
            models = []
            for m in data.get("models", []):
                models.append({
                    "name": m.get("name"),
                    "size": m.get("size", 0),
                    "size_gb": round(m.get("size", 0) / 1e9, 2),
                    "family": m.get("details", {}).get("family", "unknown"),
                    "parameters": m.get("details", {}).get("parameter_size", "unknown"),
                    "quantization": m.get("details", {}).get("quantization_level", "unknown"),
                })
            
            return jsonify({"models": models, "count": len(models)})
        except Exception as e:
            return jsonify({"models": [], "error": str(e)})


    return app


def register_routes(app: Flask) -> None:
    """Register web page routes."""
    
    @app.route("/")
    def index():
        """Dashboard home page."""
        return render_template("index.html")
    
    @app.route("/services")
    def services_page():
        """Services management page."""
        return render_template("services.html")
    
    @app.route("/benchmarks")
    def benchmarks_page():
        """Benchmarks page."""
        return render_template("benchmarks.html")
    
    @app.route("/monitoring")
    def monitoring_page():
        """Monitoring page."""
        return render_template("monitoring.html")
    
    @app.route("/logs")
    def logs_page():
        """Logs viewer page."""
        return render_template("logs.html")
    
    @app.route("/health")
    def health_check():
        """Health check endpoint."""
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
        })


def register_api_routes(app: Flask) -> None:
    """Register API routes."""
    
    # =========================================================================
    # Dashboard API
    # =========================================================================
    
    @app.route("/api/dashboard/stats")
    def api_dashboard_stats():
        """Get dashboard statistics."""
        try:
            server_manager = get_server_manager()
            client_manager = get_client_manager()
            monitor_manager = get_monitor_manager()
            
            services = server_manager.list_services()
            runs = client_manager.list_runs()
            monitors = monitor_manager.list_monitors()
            
            running_services = [s for s in services if s.status == ServiceStatus.RUNNING]
            active_runs = [r for r in runs if r.status in [RunStatus.RUNNING, RunStatus.QUEUED]]
            running_monitors = [m for m in monitors if m.status == ServiceStatus.RUNNING]
            
            return jsonify({
                "services": {
                    "total": len(services),
                    "running": len(running_services),
                },
                "benchmarks": {
                    "total": len(runs),
                    "active": len(active_runs),
                    "completed": len([r for r in runs if r.status == RunStatus.COMPLETED]),
                },
                "monitors": {
                    "total": len(monitors),
                    "running": len(running_monitors),
                },
                "timestamp": datetime.now().isoformat(),
            })
        except Exception as e:
            logger.error(f"Dashboard stats error: {e}")
            return jsonify({"error": str(e)}), 500
    
    # =========================================================================
    # Services API
    # =========================================================================
    
    @app.route("/api/services")
    def api_list_services():
        """List all services."""
        try:
            manager = get_server_manager()
            services = manager.list_services()
            
            return jsonify({
                "services": [
                    {
                        "id": s.id,
                        "recipe_name": s.recipe_name,
                        "status": s.status.value,
                        "node": s.node,
                        "slurm_job_id": s.slurm_job_id,
                        "created_at": s.created_at.isoformat(),
                        "endpoints": s.endpoints,
                    }
                    for s in services
                ],
                "total": len(services),
            })
        except Exception as e:
            logger.error(f"List services error: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/services/<service_id>")
    def api_get_service(service_id: str):
        """Get service details."""
        try:
            manager = get_server_manager()
            service = manager.get_service_status(service_id)
            
            return jsonify({
                "id": service.id,
                "recipe_name": service.recipe_name,
                "status": service.status.value,
                "node": service.node,
                "slurm_job_id": service.slurm_job_id,
                "created_at": service.created_at.isoformat(),
                "started_at": service.started_at.isoformat() if service.started_at else None,
                "endpoints": service.endpoints,
                "error_message": service.error_message,
            })
        except Exception as e:
            logger.error(f"Get service error: {e}")
            return jsonify({"error": str(e)}), 404
    
    @app.route("/api/services", methods=["POST"])
    def api_start_service():
        """Start a new service."""
        try:
            data = request.get_json()
            recipe_name = data.get("recipe")
            config_overrides = data.get("config", {})
            
            if not recipe_name:
                return jsonify({"error": "Recipe name required"}), 400
            
            manager = get_server_manager()
            service = manager.start_service(
                recipe_name=recipe_name,
                config_overrides=config_overrides,
                wait_for_ready=False,
            )
            
            return jsonify({
                "message": "Service started",
                "service_id": service.id,
                "slurm_job_id": service.slurm_job_id,
            }), 201
        except Exception as e:
            logger.error(f"Start service error: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/services/<service_id>", methods=["DELETE"])
    def api_stop_service(service_id: str):
        """Stop a service."""
        try:
            manager = get_server_manager()
            success = manager.stop_service(service_id)
            
            if success:
                return jsonify({"message": "Service stopped", "service_id": service_id})
            else:
                return jsonify({"error": "Failed to stop service"}), 500
        except Exception as e:
            logger.error(f"Stop service error: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/services/<service_id>/health")
    def api_service_health(service_id: str):
        """Check service health."""
        try:
            manager = get_server_manager()
            is_healthy = manager.check_health(service_id)
            
            return jsonify({
                "service_id": service_id,
                "healthy": is_healthy,
                "timestamp": datetime.now().isoformat(),
            })
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/recipes/servers")
    def api_list_server_recipes():
        """List available server recipes."""
        try:
            manager = get_server_manager()
            recipes = manager.list_available_recipes()
            
            recipe_details = []
            for name in recipes:
                try:
                    recipe = manager.recipe_loader.load_server(name)
                    recipe_details.append({
                        "name": recipe.name,
                        "description": recipe.description,
                        "resources": {
                            "nodes": recipe.resources.nodes,
                            "gpus": recipe.resources.gpus,
                            "memory": recipe.resources.memory,
                        },
                    })
                except Exception:
                    recipe_details.append({"name": name, "error": True})
            
            return jsonify({"recipes": recipe_details})
        except Exception as e:
            logger.error(f"List recipes error: {e}")
            return jsonify({"error": str(e)}), 500
    
    # =========================================================================
    # Benchmarks API
    # =========================================================================
    
    @app.route("/api/benchmarks")
    def api_list_benchmarks():
        """List all benchmark runs."""
        try:
            manager = get_client_manager()
            runs = manager.list_runs()
            
            return jsonify({
                "runs": [
                    {
                        "id": r.id,
                        "recipe_name": r.recipe_name,
                        "status": r.status.value,
                        "slurm_job_id": r.slurm_job_id,
                        "target_service_id": r.target_service_id,
                        "created_at": r.created_at.isoformat(),
                        "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                        "results_path": r.results_path,
                    }
                    for r in runs
                ],
                "total": len(runs),
            })
        except Exception as e:
            logger.error(f"List benchmarks error: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/benchmarks/<run_id>")
    def api_get_benchmark(run_id: str):
        """Get benchmark details."""
        try:
            manager = get_client_manager()
            run = manager.get_run_status(run_id)
            
            return jsonify({
                "id": run.id,
                "recipe_name": run.recipe_name,
                "status": run.status.value,
                "slurm_job_id": run.slurm_job_id,
                "target_service_id": run.target_service_id,
                "created_at": run.created_at.isoformat(),
                "started_at": run.started_at.isoformat() if run.started_at else None,
                "completed_at": run.completed_at.isoformat() if run.completed_at else None,
                "results_path": run.results_path,
                "error_message": run.error_message,
            })
        except Exception as e:
            logger.error(f"Get benchmark error: {e}")
            return jsonify({"error": str(e)}), 404
    
    @app.route("/api/benchmarks/<run_id>/results")
    def api_get_benchmark_results(run_id: str):
        """Get benchmark results."""
        try:
            manager = get_client_manager()
            results = manager.get_run_results(run_id)
            
            if results:
                return jsonify(results)
            else:
                return jsonify({"error": "No results available"}), 404
        except Exception as e:
            logger.error(f"Get results error: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/benchmarks", methods=["POST"])
    def api_start_benchmark():
        """Start a new benchmark."""
        try:
            data = request.get_json()
            recipe_name = data.get("recipe")
            target_service_id = data.get("target_service_id")
            config_overrides = data.get("config", {})
            
            if not recipe_name:
                return jsonify({"error": "Recipe name required"}), 400
            
            manager = get_client_manager()
            run = manager.run_client(
                recipe_name=recipe_name,
                target_service_id=target_service_id,
                config_overrides=config_overrides,
                wait_for_completion=False,
            )
            
            return jsonify({
                "message": "Benchmark started",
                "run_id": run.id,
                "slurm_job_id": run.slurm_job_id,
            }), 201
        except Exception as e:
            logger.error(f"Start benchmark error: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/benchmarks/<run_id>", methods=["DELETE"])
    def api_stop_benchmark(run_id: str):
        """Stop a benchmark run."""
        try:
            manager = get_client_manager()
            success = manager.stop_run(run_id)
            
            if success:
                return jsonify({"message": "Benchmark stopped", "run_id": run_id})
            else:
                return jsonify({"error": "Failed to stop benchmark"}), 500
        except Exception as e:
            logger.error(f"Stop benchmark error: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/recipes/clients")
    def api_list_client_recipes():
        """List available client recipes."""
        try:
            manager = get_client_manager()
            recipes = manager.list_available_recipes()
            
            recipe_details = []
            for name in recipes:
                try:
                    recipe = manager.recipe_loader.load_client(name)
                    recipe_details.append({
                        "name": recipe.name,
                        "description": recipe.description,
                        "workload_type": recipe.workload.get("type", "unknown"),
                    })
                except Exception:
                    recipe_details.append({"name": name, "error": True})
            
            return jsonify({"recipes": recipe_details})
        except Exception as e:
            logger.error(f"List recipes error: {e}")
            return jsonify({"error": str(e)}), 500
    
    # =========================================================================
    # Monitors API
    # =========================================================================
    
    @app.route("/api/monitors")
    def api_list_monitors():
        """List all monitors."""
        try:
            manager = get_monitor_manager()
            monitors = manager.list_monitors()
            
            return jsonify({
                "monitors": [
                    {
                        "id": m.id,
                        "recipe_name": m.recipe_name,
                        "status": m.status.value,
                        "prometheus_url": m.prometheus_url,
                        "grafana_url": m.grafana_url,
                        "targets": m.targets,
                        "created_at": m.created_at.isoformat(),
                    }
                    for m in monitors
                ],
                "total": len(monitors),
            })
        except Exception as e:
            logger.error(f"List monitors error: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/monitors", methods=["POST"])
    def api_start_monitor():
        """Start monitoring stack."""
        try:
            data = request.get_json()
            recipe_name = data.get("recipe", "default-monitor")
            target_ids = data.get("targets", [])
            
            manager = get_monitor_manager()
            monitor = manager.start_monitor(
                recipe_name=recipe_name,
                target_ids=target_ids,
                wait_for_ready=False,
            )
            
            return jsonify({
                "message": "Monitor started",
                "monitor_id": monitor.id,
                "prometheus_job_id": monitor.prometheus_job_id,
            }), 201
        except Exception as e:
            logger.error(f"Start monitor error: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/monitors/<monitor_id>", methods=["DELETE"])
    def api_stop_monitor(monitor_id: str):
        """Stop a monitor."""
        try:
            manager = get_monitor_manager()
            success = manager.stop_monitor(monitor_id)
            
            if success:
                return jsonify({"message": "Monitor stopped", "monitor_id": monitor_id})
            else:
                return jsonify({"error": "Failed to stop monitor"}), 500
        except Exception as e:
            logger.error(f"Stop monitor error: {e}")
            return jsonify({"error": str(e)}), 500
    
    # =========================================================================
    # Logs API
    # =========================================================================
    
    @app.route("/api/logs/service/<service_id>")
    def api_service_logs(service_id: str):
        """Get service logs."""
        try:
            lines = request.args.get("lines", 100, type=int)
            log_type = request.args.get("type", "output")
            
            manager = get_log_manager()
            content = manager.get_service_logs(service_id, lines, log_type)
            
            return jsonify({
                "service_id": service_id,
                "log_type": log_type,
                "lines": lines,
                "content": content,
                "timestamp": datetime.now().isoformat(),
            })
        except Exception as e:
            logger.error(f"Get logs error: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/logs/client/<run_id>")
    def api_client_logs(run_id: str):
        """Get client run logs."""
        try:
            lines = request.args.get("lines", 100, type=int)
            log_type = request.args.get("type", "output")
            
            manager = get_log_manager()
            content = manager.get_client_logs(run_id, lines, log_type)
            
            return jsonify({
                "run_id": run_id,
                "log_type": log_type,
                "lines": lines,
                "content": content,
                "timestamp": datetime.now().isoformat(),
            })
        except Exception as e:
            logger.error(f"Get logs error: {e}")
            return jsonify({"error": str(e)}), 500


# CLI command to run the web server
def run_server(host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
    """Run the web server."""
    app = create_app()
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_server(debug=True)
