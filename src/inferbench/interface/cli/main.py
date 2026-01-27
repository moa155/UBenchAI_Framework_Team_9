"""
InferBench Framework CLI - Main entry point.

Provides the command-line interface for managing AI benchmarking
workflows on the MeluXina supercomputer.
"""

import sys
import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from inferbench import __version__
from inferbench.core.config import get_config
from inferbench.core.models import ServiceStatus, RecipeType
from inferbench.core.exceptions import (
    InferBenchError,
    RecipeNotFoundError,
    ServiceNotFoundError,
    ServiceStartError,
    ServiceStopError,
)
from inferbench.utils.logging import setup_logging, get_logger

# Rich console for pretty output
console = Console()
logger = get_logger(__name__)


def handle_error(func):
    """Decorator to handle errors gracefully."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except InferBenchError as e:
            console.print(f"[red]Error:[/red] {e.message}")
            if e.details:
                for key, value in e.details.items():
                    console.print(f"  [dim]{key}:[/dim] {value}")
            sys.exit(1)
        except Exception as e:
            console.print(f"[red]Unexpected error:[/red] {e}")
            logger.exception("Unexpected error")
            sys.exit(1)
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


@click.group()
@click.version_option(version=__version__, prog_name="inferbench")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose (DEBUG) logging")
@click.option("-q", "--quiet", is_flag=True, help="Suppress non-essential output")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, quiet: bool) -> None:
    """
    InferBench - Unified Benchmarking Framework for AI Factory Workloads
    
    A modular framework for benchmarking AI services on HPC infrastructure.
    
    \b
    Examples:
        inferbench server list              # List available server recipes
        inferbench server start --recipe vllm-inference
        inferbench client run --recipe stress-test
        inferbench monitor start --recipe default-monitor
    """
    if verbose:
        log_level = "DEBUG"
    elif quiet:
        log_level = "ERROR"
    else:
        # Default: only show warnings and errors (cleaner output)
        log_level = "WARNING"
    
    setup_logging(level=log_level)
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet
    ctx.obj["config"] = get_config()


# =============================================================================
# Server Commands
# =============================================================================

@cli.group()
@click.pass_context
def server(ctx: click.Context) -> None:
    """Manage AI service servers."""
    pass


@server.command("list")
@click.option("--running", is_flag=True, help="Show only running services")
@click.pass_context
@handle_error
def server_list(ctx: click.Context, running: bool) -> None:
    """List available or running server recipes."""
    from inferbench.servers.manager import get_server_manager
    
    manager = get_server_manager()
    
    if running:
        services = manager.list_services(running_only=True)
        
        if not services:
            console.print("[dim]No services currently running.[/dim]")
            return
        
        table = Table(title="Running Services")
        table.add_column("ID", style="cyan")
        table.add_column("Recipe", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Job ID", style="blue")
        table.add_column("Node", style="magenta")
        table.add_column("Endpoint", style="white")
        
        for svc in services:
            endpoint = svc.get_endpoint("api") or "-"
            table.add_row(
                svc.id, svc.recipe_name, svc.status.value,
                svc.slurm_job_id or "-", svc.node or "-", endpoint,
            )
        console.print(table)
    else:
        recipes = manager.list_available_recipes()
        
        if not recipes:
            console.print("[dim]No server recipes found. Add recipes to recipes/servers/[/dim]")
            return
        
        console.print("[bold cyan]Available Server Recipes:[/bold cyan]\n")
        for recipe_name in recipes:
            try:
                recipe = manager.recipe_loader.load_server(recipe_name)
                desc = recipe.description or "No description"
                console.print(f"  [green]•[/green] [bold]{recipe_name}[/bold]")
                console.print(f"    [dim]{desc}[/dim]")
                console.print(f"    [dim]Resources: {recipe.resources.nodes} node(s), {recipe.resources.gpus} GPU(s), {recipe.resources.memory}[/dim]\n")
            except Exception:
                console.print(f"  [red]•[/red] [bold]{recipe_name}[/bold] [red](error loading)[/red]")


@server.command("start")
@click.option("--recipe", "-r", required=True, help="Recipe name to start")
@click.option("--config", "-c", type=click.Path(exists=True), help="Config override file")
@click.option("--no-wait", is_flag=True, help="Don't wait for service to be ready")
@click.option("--timeout", "-t", default=300, help="Timeout for waiting (seconds)")
@click.pass_context
@handle_error
def server_start(ctx: click.Context, recipe: str, config: str | None, no_wait: bool, timeout: int) -> None:
    """Start a server from a recipe."""
    from inferbench.servers.manager import get_server_manager
    import yaml
    
    manager = get_server_manager()
    overrides = None
    if config:
        with open(config, "r") as f:
            overrides = yaml.safe_load(f)
    
    console.print(f"[green]Starting server with recipe:[/green] [bold]{recipe}[/bold]")
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Submitting job...", total=None)
        service = manager.start_service(
            recipe_name=recipe, config_overrides=overrides,
            wait_for_ready=not no_wait, timeout=timeout,
        )
        progress.update(task, description="Service started!")
    
    console.print()
    console.print(Panel.fit(
        f"[green]✓ Service started successfully![/green]\n\n"
        f"[cyan]Service ID:[/cyan] {service.id}\n"
        f"[cyan]Recipe:[/cyan] {service.recipe_name}\n"
        f"[cyan]Status:[/cyan] {service.status.value}\n"
        f"[cyan]SLURM Job ID:[/cyan] {service.slurm_job_id}\n"
        f"[cyan]Node:[/cyan] {service.node or 'pending'}\n"
        f"[cyan]Endpoint:[/cyan] {service.get_endpoint('api') or 'pending'}",
        title="Service Started"
    ))


@server.command("stop")
@click.argument("service_id")
@click.option("--force", "-f", is_flag=True, help="Force stop")
@click.pass_context
@handle_error
def server_stop(ctx: click.Context, service_id: str, force: bool) -> None:
    """Stop a running server by ID or SLURM job ID."""
    from inferbench.servers.manager import get_server_manager
    
    manager = get_server_manager()
    console.print(f"[yellow]Stopping service:[/yellow] {service_id}")
    success = manager.stop_service(service_id, force=force)
    
    if success:
        console.print(f"[green]✓ Service {service_id} stopped successfully[/green]")
    else:
        console.print(f"[red]✗ Failed to stop service {service_id}[/red]")


@server.command("status")
@click.argument("service_id")
@click.pass_context
@handle_error
def server_status(ctx: click.Context, service_id: str) -> None:
    """Get status of a running server."""
    from inferbench.servers.manager import get_server_manager
    
    manager = get_server_manager()
    service = manager.get_service_status(service_id)
    
    status_colors = {
        ServiceStatus.RUNNING: "green", ServiceStatus.PENDING: "yellow",
        ServiceStatus.STARTING: "yellow", ServiceStatus.STOPPING: "yellow",
        ServiceStatus.STOPPED: "dim", ServiceStatus.ERROR: "red",
    }
    color = status_colors.get(service.status, "white")
    
    endpoints_str = "\n".join(f"  • {n}: {u}" for n, u in service.endpoints.items()) if service.endpoints else "  [dim]None[/dim]"
    
    console.print(Panel.fit(
        f"[cyan]Service ID:[/cyan] {service.id}\n"
        f"[cyan]Recipe:[/cyan] {service.recipe_name}\n"
        f"[cyan]Status:[/cyan] [{color}]{service.status.value}[/{color}]\n"
        f"[cyan]SLURM Job ID:[/cyan] {service.slurm_job_id or '-'}\n"
        f"[cyan]Node:[/cyan] {service.node or '-'}\n"
        f"[cyan]Created:[/cyan] {service.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"[cyan]Started:[/cyan] {service.started_at.strftime('%Y-%m-%d %H:%M:%S') if service.started_at else '-'}\n"
        f"[cyan]Endpoints:[/cyan]\n{endpoints_str}",
        title=f"Service Status: {service.id}"
    ))
    
    if service.error_message:
        console.print(f"\n[red]Error:[/red] {service.error_message}")


@server.command("logs")
@click.argument("service_id")
@click.option("--lines", "-n", default=50, help="Number of lines to show")
@click.option("--error", "-e", is_flag=True, help="Show error logs")
@click.pass_context
@handle_error
def server_logs(ctx: click.Context, service_id: str, lines: int, error: bool) -> None:
    """Show logs for a server."""
    from inferbench.servers.manager import get_server_manager
    
    manager = get_server_manager()
    log_type = "error" if error else "output"
    logs_content = manager.get_service_logs(service_id, lines=lines, log_type=log_type)
    console.print(Panel(logs_content, title=f"{'Error' if error else 'Output'} Logs: {service_id}", border_style="dim"))


@server.command("health")
@click.argument("service_id")
@click.pass_context
@handle_error
def server_health(ctx: click.Context, service_id: str) -> None:
    """Check health of a running server."""
    from inferbench.servers.manager import get_server_manager
    
    manager = get_server_manager()
    result = manager.check_health(service_id)
    
    if result["healthy"]:
        console.print(f"[green]✓ Service {service_id} is healthy[/green]")
    else:
        console.print(f"[red]✗ Service {service_id} is unhealthy[/red]")
    
    for key, value in result.items():
        if key != "healthy":
            console.print(f"  [dim]{key}:[/dim] {value}")


# =============================================================================
# Client Commands
# =============================================================================

@cli.group()
@click.pass_context
def client(ctx: click.Context) -> None:
    """Manage benchmark clients."""
    pass


@client.command("list")
@click.option("--running", is_flag=True, help="Show only running clients")
@click.pass_context
@handle_error
def client_list(ctx: click.Context, running: bool) -> None:
    """List available or running client recipes."""
    from inferbench.clients.manager import get_client_manager
    
    manager = get_client_manager()
    
    if running:
        runs = manager.list_runs(active_only=True)
        
        if not runs:
            console.print("[dim]No clients currently running.[/dim]")
            return
        
        table = Table(title="Active Client Runs")
        table.add_column("ID", style="cyan")
        table.add_column("Recipe", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Job ID", style="blue")
        table.add_column("Target", style="magenta")
        
        for run in runs:
            table.add_row(
                run.id, run.recipe_name, run.status.value,
                run.slurm_job_id or "-", run.target_service_id or "-",
            )
        console.print(table)
    else:
        recipes = manager.list_available_recipes()
        
        if not recipes:
            console.print("[dim]No client recipes found.[/dim]")
            return
        
        console.print("[bold cyan]Available Client Recipes:[/bold cyan]\n")
        for name in recipes:
            try:
                recipe = manager.recipe_loader.load_client(name)
                desc = recipe.description or "No description"
                workload = recipe.workload.get("type", "unknown")
                console.print(f"  [green]•[/green] [bold]{name}[/bold]")
                console.print(f"    [dim]{desc}[/dim]")
                console.print(f"    [dim]Workload: {workload}, Resources: {recipe.resources.nodes} node(s), {recipe.resources.memory}[/dim]\n")
            except Exception:
                console.print(f"  [red]•[/red] [bold]{name}[/bold] [red](error)[/red]")


@client.command("run")
@click.option("--recipe", "-r", required=True, help="Client recipe name")
@click.option("--target", "-t", help="Target service ID to benchmark")
@click.option("--config", "-c", type=click.Path(exists=True), help="Config override file")
@click.option("--wait", is_flag=True, help="Wait for completion")
@click.option("--timeout", default=3600, help="Timeout in seconds")
@click.pass_context
@handle_error
def client_run(
    ctx: click.Context, 
    recipe: str, 
    target: str | None,
    config: str | None,
    wait: bool,
    timeout: int
) -> None:
    """Run a benchmark client."""
    from inferbench.clients.manager import get_client_manager
    import yaml
    
    manager = get_client_manager()
    
    overrides = None
    if config:
        with open(config, "r") as f:
            overrides = yaml.safe_load(f)
    
    console.print(f"[green]Running benchmark client:[/green] [bold]{recipe}[/bold]")
    if target:
        console.print(f"[cyan]Target service:[/cyan] {target}")
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Submitting benchmark job...", total=None)
        
        run = manager.run_client(
            recipe_name=recipe,
            target_service_id=target,
            config_overrides=overrides,
            wait_for_completion=wait,
            timeout=timeout,
        )
        
        progress.update(task, description="Benchmark submitted!")
    
    console.print()
    console.print(Panel.fit(
        f"[green]✓ Benchmark client started![/green]\n\n"
        f"[cyan]Run ID:[/cyan] {run.id}\n"
        f"[cyan]Recipe:[/cyan] {run.recipe_name}\n"
        f"[cyan]Status:[/cyan] {run.status.value}\n"
        f"[cyan]SLURM Job ID:[/cyan] {run.slurm_job_id}\n"
        f"[cyan]Target:[/cyan] {run.target_service_id or 'default'}\n"
        f"[cyan]Results Path:[/cyan] {run.results_path or 'pending'}",
        title="Benchmark Started"
    ))
    
    if not wait:
        console.print("\n[dim]Use 'inferbench client status {id}' to check progress.[/dim]")
        console.print("[dim]Use 'inferbench client results {id}' to view results when complete.[/dim]")


@client.command("stop")
@click.argument("run_id")
@click.pass_context
@handle_error
def client_stop(ctx: click.Context, run_id: str) -> None:
    """Stop a running client."""
    from inferbench.clients.manager import get_client_manager
    
    manager = get_client_manager()
    console.print(f"[yellow]Stopping client run:[/yellow] {run_id}")
    
    success = manager.stop_run(run_id)
    
    if success:
        console.print(f"[green]✓ Run {run_id} stopped successfully[/green]")
    else:
        console.print(f"[red]✗ Failed to stop run {run_id}[/red]")


@client.command("status")
@click.argument("run_id")
@click.pass_context
@handle_error
def client_status(ctx: click.Context, run_id: str) -> None:
    """Get status of a client run."""
    from inferbench.clients.manager import get_client_manager
    from inferbench.core.models import RunStatus
    
    manager = get_client_manager()
    run = manager.get_run_status(run_id)
    
    status_colors = {
        RunStatus.SUBMITTED: "yellow",
        RunStatus.QUEUED: "yellow",
        RunStatus.RUNNING: "blue",
        RunStatus.COMPLETED: "green",
        RunStatus.FAILED: "red",
        RunStatus.CANCELED: "dim",
    }
    color = status_colors.get(run.status, "white")
    
    console.print(Panel.fit(
        f"[cyan]Run ID:[/cyan] {run.id}\n"
        f"[cyan]Recipe:[/cyan] {run.recipe_name}\n"
        f"[cyan]Status:[/cyan] [{color}]{run.status.value}[/{color}]\n"
        f"[cyan]SLURM Job ID:[/cyan] {run.slurm_job_id or '-'}\n"
        f"[cyan]Target Service:[/cyan] {run.target_service_id or '-'}\n"
        f"[cyan]Created:[/cyan] {run.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"[cyan]Started:[/cyan] {run.started_at.strftime('%Y-%m-%d %H:%M:%S') if run.started_at else '-'}\n"
        f"[cyan]Completed:[/cyan] {run.completed_at.strftime('%Y-%m-%d %H:%M:%S') if run.completed_at else '-'}\n"
        f"[cyan]Results:[/cyan] {run.results_path or '-'}",
        title=f"Client Run: {run.id}"
    ))
    
    if run.error_message:
        console.print(f"\n[red]Error:[/red] {run.error_message}")


@client.command("results")
@click.argument("run_id")
@click.option("--raw", is_flag=True, help="Show raw JSON output")
@click.pass_context
@handle_error
def client_results(ctx: click.Context, run_id: str, raw: bool) -> None:
    """View results of a completed benchmark."""
    from inferbench.clients.manager import get_client_manager
    import json
    
    manager = get_client_manager()
    results = manager.get_run_results(run_id)
    
    if not results:
        console.print(f"[yellow]No results available for run {run_id}[/yellow]")
        console.print("[dim]The run may still be in progress or failed before producing results.[/dim]")
        return
    
    if raw:
        console.print(json.dumps(results, indent=2))
        return
    
    # Pretty print results
    summary = results.get("summary", {})
    latency = results.get("latency", {})
    
    console.print(Panel.fit(
        f"[bold cyan]Benchmark: {results.get('benchmark', 'unknown')}[/bold cyan]\n"
        f"[dim]Timestamp: {results.get('timestamp', '-')}[/dim]\n"
        f"[dim]Target: {results.get('target', '-')}[/dim]\n\n"
        f"[green]Summary:[/green]\n"
        f"  Total Requests: {summary.get('total_requests', 0)}\n"
        f"  Successful: {summary.get('successful_requests', 0)}\n"
        f"  Failed: {summary.get('failed_requests', 0)}\n"
        f"  Success Rate: {summary.get('success_rate', 0):.2f}%\n"
        f"  Throughput: {summary.get('actual_throughput', 0):.2f} req/s\n\n"
        f"[green]Latency (seconds):[/green]\n"
        f"  Min: {latency.get('min', 0):.4f}\n"
        f"  Max: {latency.get('max', 0):.4f}\n"
        f"  Mean: {latency.get('mean', 0):.4f}\n"
        f"  Median: {latency.get('median', 0):.4f}\n"
        f"  P95: {latency.get('p95', 0):.4f}\n"
        f"  P99: {latency.get('p99', 0):.4f}",
        title="Benchmark Results"
    ))


@client.command("logs")
@click.argument("run_id")
@click.option("--lines", "-n", default=50, help="Number of lines to show")
@click.option("--error", "-e", is_flag=True, help="Show error logs")
@click.pass_context
@handle_error
def client_logs(ctx: click.Context, run_id: str, lines: int, error: bool) -> None:
    """Show logs for a client run."""
    from inferbench.clients.manager import get_client_manager
    
    manager = get_client_manager()
    log_type = "error" if error else "output"
    logs_content = manager.get_run_logs(run_id, lines=lines, log_type=log_type)
    console.print(Panel(logs_content, title=f"{'Error' if error else 'Output'} Logs: {run_id}", border_style="dim"))


# =============================================================================
# Monitor Commands
# =============================================================================

@cli.group()
@click.pass_context
def monitor(ctx: click.Context) -> None:
    """Manage monitoring stack (Prometheus + Grafana)."""
    pass


@monitor.command("list")
@click.option("--running", is_flag=True, help="Show only running monitors")
@click.pass_context
@handle_error
def monitor_list(ctx: click.Context, running: bool) -> None:
    """List available or running monitors."""
    from inferbench.monitors.manager import get_monitor_manager
    
    manager = get_monitor_manager()
    
    if running:
        monitors = manager.list_monitors(running_only=True)
        
        if not monitors:
            console.print("[dim]No monitors currently running.[/dim]")
            return
        
        table = Table(title="Running Monitors")
        table.add_column("ID", style="cyan")
        table.add_column("Recipe", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Prometheus", style="blue")
        table.add_column("Targets", style="magenta")
        
        for mon in monitors:
            table.add_row(
                mon.id,
                mon.recipe_name,
                mon.status.value,
                mon.prometheus_url or "-",
                str(len(mon.targets)),
            )
        console.print(table)
    else:
        recipes = manager.list_available_recipes()
        
        if not recipes:
            console.print("[dim]No monitor recipes found. Add recipes to recipes/monitors/[/dim]")
            return
        
        console.print("[bold cyan]Available Monitor Recipes:[/bold cyan]\n")
        for name in recipes:
            try:
                recipe = manager.recipe_loader.load_monitor(name)
                desc = recipe.description or "No description"
                console.print(f"  [green]•[/green] [bold]{name}[/bold]")
                console.print(f"    [dim]{desc}[/dim]")
                console.print(f"    [dim]Scrape interval: {recipe.scrape_interval}s, Retention: {recipe.retention}[/dim]\n")
            except Exception:
                console.print(f"  [red]•[/red] [bold]{name}[/bold] [red](error)[/red]")


@monitor.command("start")
@click.option("--recipe", "-r", required=True, help="Monitor recipe name")
@click.option("--targets", "-t", help="Comma-separated service IDs to monitor")
@click.option("--no-wait", is_flag=True, help="Don't wait for services to be ready")
@click.pass_context
@handle_error
def monitor_start(ctx: click.Context, recipe: str, targets: str | None, no_wait: bool) -> None:
    """Start monitoring stack."""
    from inferbench.monitors.manager import get_monitor_manager
    
    manager = get_monitor_manager()
    
    target_ids = targets.split(",") if targets else []
    
    console.print(f"[green]Starting monitoring stack:[/green] [bold]{recipe}[/bold]")
    if target_ids:
        console.print(f"[cyan]Targets:[/cyan] {', '.join(target_ids)}")
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Starting Prometheus...", total=None)
        
        monitor = manager.start_monitor(
            recipe_name=recipe,
            target_ids=target_ids,
            wait_for_ready=not no_wait,
        )
        
        progress.update(task, description="Monitoring stack started!")
    
    console.print()
    console.print(Panel.fit(
        f"[green]✓ Monitoring stack started![/green]\n\n"
        f"[cyan]Monitor ID:[/cyan] {monitor.id}\n"
        f"[cyan]Recipe:[/cyan] {monitor.recipe_name}\n"
        f"[cyan]Status:[/cyan] {monitor.status.value}\n"
        f"[cyan]Prometheus Job:[/cyan] {monitor.prometheus_job_id or '-'}\n"
        f"[cyan]Prometheus URL:[/cyan] {monitor.prometheus_url or 'pending'}\n"
        f"[cyan]Targets:[/cyan] {len(monitor.targets)}",
        title="Monitoring Started"
    ))
    
    console.print("\n[bold]Access Instructions:[/bold]")
    console.print("[dim]1. Set up SSH tunnel from your local machine:[/dim]")
    if monitor.prometheus_url:
        node = monitor.prometheus_url.split("//")[1].split(":")[0]
        port = monitor.prometheus_url.split(":")[-1]
        console.print(f"   ssh -L {port}:{node}:{port} YOUR_USER@login.lxp.lu")
        console.print(f"\n[dim]2. Open in browser: http://localhost:{port}[/dim]")


@monitor.command("stop")
@click.argument("monitor_id")
@click.pass_context
@handle_error
def monitor_stop(ctx: click.Context, monitor_id: str) -> None:
    """Stop a monitoring instance."""
    from inferbench.monitors.manager import get_monitor_manager
    
    manager = get_monitor_manager()
    
    console.print(f"[yellow]Stopping monitor:[/yellow] {monitor_id}")
    
    success = manager.stop_monitor(monitor_id)
    
    if success:
        console.print(f"[green]✓ Monitor {monitor_id} stopped successfully[/green]")
    else:
        console.print(f"[red]✗ Failed to stop monitor {monitor_id}[/red]")


@monitor.command("status")
@click.argument("monitor_id")
@click.pass_context
@handle_error
def monitor_status(ctx: click.Context, monitor_id: str) -> None:
    """Get status of a monitoring instance."""
    from inferbench.monitors.manager import get_monitor_manager
    
    manager = get_monitor_manager()
    monitor = manager.get_monitor_status(monitor_id)
    
    status_colors = {
        ServiceStatus.RUNNING: "green",
        ServiceStatus.PENDING: "yellow",
        ServiceStatus.STOPPED: "dim",
        ServiceStatus.ERROR: "red",
    }
    color = status_colors.get(monitor.status, "white")
    
    targets_str = ", ".join(monitor.targets) if monitor.targets else "[dim]None[/dim]"
    
    console.print(Panel.fit(
        f"[cyan]Monitor ID:[/cyan] {monitor.id}\n"
        f"[cyan]Recipe:[/cyan] {monitor.recipe_name}\n"
        f"[cyan]Status:[/cyan] [{color}]{monitor.status.value}[/{color}]\n"
        f"[cyan]Prometheus Job:[/cyan] {monitor.prometheus_job_id or '-'}\n"
        f"[cyan]Prometheus URL:[/cyan] {monitor.prometheus_url or '-'}\n"
        f"[cyan]Grafana Job:[/cyan] {monitor.grafana_job_id or '-'}\n"
        f"[cyan]Grafana URL:[/cyan] {monitor.grafana_url or '-'}\n"
        f"[cyan]Targets:[/cyan] {targets_str}\n"
        f"[cyan]Created:[/cyan] {monitor.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
        title=f"Monitor Status: {monitor.id}"
    ))


@monitor.command("add-target")
@click.argument("monitor_id")
@click.argument("service_id")
@click.pass_context
@handle_error
def monitor_add_target(ctx: click.Context, monitor_id: str, service_id: str) -> None:
    """Add a service target to an existing monitor."""
    from inferbench.monitors.manager import get_monitor_manager
    
    manager = get_monitor_manager()
    
    success = manager.add_target(monitor_id, service_id)
    
    if success:
        console.print(f"[green]✓ Added target {service_id} to monitor {monitor_id}[/green]")
    else:
        console.print(f"[red]✗ Failed to add target {service_id}[/red]")


@monitor.command("remove-target")
@click.argument("monitor_id")
@click.argument("service_id")
@click.pass_context
@handle_error
def monitor_remove_target(ctx: click.Context, monitor_id: str, service_id: str) -> None:
    """Remove a service target from a monitor."""
    from inferbench.monitors.manager import get_monitor_manager
    
    manager = get_monitor_manager()
    
    success = manager.remove_target(monitor_id, service_id)
    
    if success:
        console.print(f"[green]✓ Removed target {service_id} from monitor {monitor_id}[/green]")
    else:
        console.print(f"[yellow]Target {service_id} not found in monitor {monitor_id}[/yellow]")


# =============================================================================
# Logs Commands
# =============================================================================

@cli.group()
@click.pass_context
def logs(ctx: click.Context) -> None:
    """Manage log collection and export."""
    pass


@logs.command("show")
@click.option("--service-id", "-s", help="Service ID")
@click.option("--client-id", "-c", help="Client run ID")
@click.option("--job-id", "-j", help="SLURM job ID")
@click.option("--lines", "-n", default=100, help="Number of lines")
@click.option("--error", "-e", is_flag=True, help="Show error logs")
@click.option("--follow", "-f", is_flag=True, help="Follow log output (not implemented)")
@click.pass_context
@handle_error
def logs_show(
    ctx: click.Context, 
    service_id: str | None, 
    client_id: str | None,
    job_id: str | None, 
    lines: int,
    error: bool,
    follow: bool
) -> None:
    """Show logs for a service, client, or job."""
    from inferbench.logs.manager import get_log_manager
    
    manager = get_log_manager()
    log_type = "error" if error else "output"
    
    if service_id:
        content = manager.get_service_logs(service_id, lines, log_type)
        title = f"{'Error' if error else 'Output'} Logs: {service_id}"
    elif client_id:
        content = manager.get_client_logs(client_id, lines, log_type)
        title = f"{'Error' if error else 'Output'} Logs: {client_id}"
    elif job_id:
        content = manager.get_job_logs(job_id, lines, log_type)
        title = f"{'Error' if error else 'Output'} Logs: Job {job_id}"
    else:
        console.print("[red]Please specify --service-id, --client-id, or --job-id[/red]")
        return
    
    if follow:
        console.print("[yellow]Follow mode not yet implemented. Showing current logs.[/yellow]\n")
    
    console.print(Panel(content, title=title, border_style="dim"))


@logs.command("tail")
@click.argument("service_id")
@click.option("--lines", "-n", default=50, help="Number of lines")
@click.option("--error", "-e", is_flag=True, help="Show error logs")
@click.pass_context
@handle_error
def logs_tail(ctx: click.Context, service_id: str, lines: int, error: bool) -> None:
    """Show the last N lines of logs for a service."""
    from inferbench.logs.manager import get_log_manager
    
    manager = get_log_manager()
    log_type = "error" if error else "output"
    content = manager.tail_logs(service_id, lines, log_type)
    
    console.print(Panel(content, title=f"Tail: {service_id} (last {lines} lines)", border_style="dim"))


@logs.command("search")
@click.argument("service_id")
@click.argument("pattern")
@click.option("--lines", "-n", default=1000, help="Number of lines to search")
@click.option("--context", "-C", default=2, help="Lines of context")
@click.pass_context
@handle_error
def logs_search(ctx: click.Context, service_id: str, pattern: str, lines: int, context: int) -> None:
    """Search logs for a pattern."""
    from inferbench.logs.manager import get_log_manager
    
    manager = get_log_manager()
    matches = manager.search_logs(service_id, pattern, lines, context)
    
    if not matches:
        console.print(f"[yellow]No matches found for pattern: {pattern}[/yellow]")
        return
    
    console.print(f"[green]Found {len(matches)} matches for:[/green] [bold]{pattern}[/bold]\n")
    
    for i, match in enumerate(matches, 1):
        console.print(f"[cyan]Match {i} (line {match['line']}):[/cyan]")
        
        # Context before
        for line in match["context_before"]:
            console.print(f"  [dim]{line}[/dim]")
        
        # Match line
        console.print(f"  [bold yellow]{match['match']}[/bold yellow]")
        
        # Context after
        for line in match["context_after"]:
            console.print(f"  [dim]{line}[/dim]")
        
        console.print()


@logs.command("stats")
@click.argument("service_id")
@click.option("--lines", "-n", default=1000, help="Number of lines to analyze")
@click.pass_context
@handle_error
def logs_stats(ctx: click.Context, service_id: str, lines: int) -> None:
    """Show statistics about logs."""
    from inferbench.logs.manager import get_log_manager
    
    manager = get_log_manager()
    stats = manager.get_log_stats(service_id, lines)
    
    level_str = "\n".join(f"    {k}: {v}" for k, v in stats["level_counts"].items())
    sources_str = ", ".join(stats["sources"][:5]) if stats["sources"] else "None"
    if len(stats["sources"]) > 5:
        sources_str += f" (+{len(stats['sources']) - 5} more)"
    
    console.print(Panel.fit(
        f"[cyan]Total Lines:[/cyan] {stats['total_lines']}\n"
        f"[cyan]Start Time:[/cyan] {stats['start_time'] or 'N/A'}\n"
        f"[cyan]End Time:[/cyan] {stats['end_time'] or 'N/A'}\n"
        f"[cyan]Log Levels:[/cyan]\n{level_str}\n"
        f"[cyan]Sources:[/cyan] {sources_str}",
        title=f"Log Statistics: {service_id}"
    ))


@logs.command("export")
@click.option("--service-id", "-s", help="Service ID to export logs for")
@click.option("--client-id", "-c", help="Client run ID to export logs for")
@click.option("--output", "-o", help="Output file path")
@click.option("--format", "-f", "fmt", type=click.Choice(["text", "json", "csv"]), default="text", help="Export format")
@click.option("--lines", "-n", default=1000, help="Number of lines to export")
@click.option("--include-error", is_flag=True, help="Include error logs")
@click.pass_context
@handle_error
def logs_export(
    ctx: click.Context, 
    service_id: str | None, 
    client_id: str | None,
    output: str | None, 
    fmt: str,
    lines: int,
    include_error: bool
) -> None:
    """Export logs to a file."""
    from inferbench.logs.manager import get_log_manager
    from pathlib import Path
    
    manager = get_log_manager()
    
    if service_id:
        output_path = Path(output) if output else None
        result_path = manager.export_service_logs(
            service_id, output_path, fmt, lines, include_error
        )
        console.print(f"[green]✓ Logs exported to:[/green] {result_path}")
    elif client_id:
        # Get client logs and export
        logs = manager.get_client_logs(client_id, lines, "output", parse=True)
        if include_error:
            error_logs = manager.get_client_logs(client_id, lines, "error", parse=True)
            logs.entries.extend(error_logs.entries)
            logs.total_lines = len(logs.entries)
        
        if output is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output = f"logs/exports/{client_id}_{timestamp}.{fmt}"
        
        result_path = manager.export_logs(logs, Path(output), fmt)
        console.print(f"[green]✓ Logs exported to:[/green] {result_path}")
    else:
        console.print("[red]Please specify --service-id or --client-id[/red]")


@logs.command("clean")
@click.option("--older-than", "-d", default=7, help="Delete logs older than N days")
@click.option("--dry-run", is_flag=True, help="Show what would be deleted")
@click.pass_context
@handle_error
def logs_clean(ctx: click.Context, older_than: int, dry_run: bool) -> None:
    """Clean old log files."""
    from pathlib import Path
    from datetime import datetime, timedelta
    
    config = ctx.obj["config"]
    cutoff = datetime.now() - timedelta(days=older_than)
    
    log_dirs = [
        config.logs_dir / "servers",
        config.logs_dir / "clients",
        config.logs_dir / "monitors",
        config.logs_dir / "exports",
    ]
    
    files_to_delete = []
    total_size = 0
    
    for log_dir in log_dirs:
        if not log_dir.exists():
            continue
        
        for file_path in log_dir.rglob("*"):
            if file_path.is_file():
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if mtime < cutoff:
                    files_to_delete.append(file_path)
                    total_size += file_path.stat().st_size
    
    if not files_to_delete:
        console.print(f"[green]No log files older than {older_than} days found.[/green]")
        return
    
    size_mb = total_size / (1024 * 1024)
    
    if dry_run:
        console.print(f"[yellow]Would delete {len(files_to_delete)} files ({size_mb:.2f} MB):[/yellow]")
        for f in files_to_delete[:10]:
            console.print(f"  [dim]{f}[/dim]")
        if len(files_to_delete) > 10:
            console.print(f"  [dim]... and {len(files_to_delete) - 10} more[/dim]")
    else:
        for f in files_to_delete:
            f.unlink()
        console.print(f"[green]✓ Deleted {len(files_to_delete)} files ({size_mb:.2f} MB)[/green]")


# =============================================================================
# Utility Commands
# =============================================================================

@cli.command("web")
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=5000, help="Port to listen on")
@click.pass_context
def web(ctx: click.Context, host: str, port: int) -> None:
    """Start the web dashboard interface."""
    console.print(Panel.fit(
        f"[green]Starting InferBench Web Dashboard[/green]\n"
        f"[cyan]URL: http://{host}:{port}[/cyan]",
        title="Web Interface"
    ))
    console.print("[yellow]Web interface implementation coming in Phase 7.[/yellow]")


@cli.command("info")
@click.pass_context
def info(ctx: click.Context) -> None:
    """Show framework information and configuration."""
    config = ctx.obj["config"]
    
    console.print(Panel.fit(
        f"[bold cyan]InferBench Framework v{__version__}[/bold cyan]\n\n"
        f"[green]Configuration:[/green]\n"
        f"  Config Dir: {config.config_dir}\n"
        f"  Recipes Dir: {config.recipes_dir}\n"
        f"  Results Dir: {config.results_dir}\n"
        f"  Logs Dir: {config.logs_dir}\n\n"
        f"[green]MeluXina:[/green]\n"
        f"  User: {config.meluxina_user or 'Not configured'}\n"
        f"  Project: {config.meluxina_project or 'Not configured'}\n"
        f"  Partition: {config.slurm.partition}\n\n"
        f"[green]Monitoring:[/green]\n"
        f"  Prometheus Port: {config.monitoring.prometheus_port}\n"
        f"  Grafana Port: {config.monitoring.grafana_port}",
        title="Framework Information"
    ))


# =============================================================================
# Web Interface Commands
# =============================================================================

@cli.command("web")
@click.option("--host", "-h", default="0.0.0.0", help="Host to bind to")
@click.option("--port", "-p", default=5000, help="Port to listen on")
@click.option("--debug", "-d", is_flag=True, help="Enable debug mode")
@click.pass_context
@handle_error
def web_server(ctx: click.Context, host: str, port: int, debug: bool) -> None:
    """Start the web dashboard server."""
    from inferbench.interface.web.app import create_app
    
    console.print(f"[green]Starting InferBench Web Dashboard[/green]")
    console.print(f"[cyan]URL:[/cyan] http://{host}:{port}")
    console.print(f"[cyan]Debug:[/cyan] {'Enabled' if debug else 'Disabled'}")
    console.print()
    console.print("[dim]Press Ctrl+C to stop the server[/dim]")
    console.print()
    
    app = create_app()
    app.run(host=host, port=port, debug=debug)


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
