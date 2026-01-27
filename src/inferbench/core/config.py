"""
Configuration management for InferBench Framework.

Handles loading configuration from environment variables, config files,
and provides defaults for all framework settings.
"""

import os
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field

from dotenv import load_dotenv


@dataclass
class SlurmConfig:
    """SLURM-specific configuration."""
    
    account: Optional[str] = None
    partition: str = "gpu"
    qos: str = "default"
    default_time: str = "01:00:00"
    default_nodes: int = 1
    default_gpus: int = 1
    default_memory: str = "32G"


@dataclass
class ContainerConfig:
    """Container runtime configuration."""
    
    runtime: str = "apptainer"  # apptainer or singularity
    cache_dir: Optional[str] = None
    images_dir: Optional[str] = None
    bind_paths: list[str] = field(default_factory=list)


@dataclass
class MonitoringConfig:
    """Monitoring stack configuration."""
    
    prometheus_port: int = 9090
    grafana_port: int = 3000
    grafana_admin_user: str = "admin"
    grafana_admin_password: str = "admin"
    collection_interval: int = 15  # seconds


@dataclass
class WebConfig:
    """Web interface configuration."""
    
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False


@dataclass
class Config:
    """
    Main configuration class for InferBench Framework.

    Loads configuration from environment variables and provides
    sensible defaults for all settings.
    """
    
    # Paths
    base_dir: Path = field(default_factory=lambda: Path.cwd())
    config_dir: Path = field(default_factory=lambda: Path.cwd() / "config")
    recipes_dir: Path = field(default_factory=lambda: Path.cwd() / "recipes")
    results_dir: Path = field(default_factory=lambda: Path.cwd() / "results")
    logs_dir: Path = field(default_factory=lambda: Path.cwd() / "logs")
    
    # MeluXina settings
    meluxina_user: Optional[str] = None
    meluxina_project: Optional[str] = None
    
    # Logging
    log_level: str = "INFO"
    
    # Sub-configurations
    slurm: SlurmConfig = field(default_factory=SlurmConfig)
    container: ContainerConfig = field(default_factory=ContainerConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    web: WebConfig = field(default_factory=WebConfig)
    
    @classmethod
    def from_env(cls, env_file: Optional[Path] = None) -> "Config":
        """
        Load configuration from environment variables.
        
        Args:
            env_file: Optional path to .env file
            
        Returns:
            Config instance with values from environment
        """
        # Load .env file if it exists
        if env_file and env_file.exists():
            load_dotenv(env_file)
        else:
            load_dotenv()  # Try default .env
        
        config = cls()
        
        # Load path settings
        if base := os.getenv("INFERBENCH_BASE_DIR"):
            config.base_dir = Path(base)
        if config_dir := os.getenv("INFERBENCH_CONFIG_DIR"):
            config.config_dir = Path(config_dir)
        if recipes_dir := os.getenv("INFERBENCH_RECIPES_DIR"):
            config.recipes_dir = Path(recipes_dir)
        if results_dir := os.getenv("INFERBENCH_RESULTS_DIR"):
            config.results_dir = Path(results_dir)
        if logs_dir := os.getenv("INFERBENCH_LOGS_DIR"):
            config.logs_dir = Path(logs_dir)
        
        # MeluXina settings
        config.meluxina_user = os.getenv("MELUXINA_USER")
        config.meluxina_project = os.getenv("MELUXINA_PROJECT")
        
        # Logging
        config.log_level = os.getenv("INFERBENCH_LOG_LEVEL", "INFO")
        
        # SLURM config
        config.slurm = SlurmConfig(
            account=os.getenv("SLURM_ACCOUNT"),
            partition=os.getenv("MELUXINA_PARTITION", "gpu"),
            qos=os.getenv("SLURM_QOS", "default"),
        )
        
        # Container config
        config.container = ContainerConfig(
            runtime=os.getenv("CONTAINER_RUNTIME", "apptainer"),
            cache_dir=os.getenv("CONTAINER_CACHE_DIR"),
            images_dir=os.getenv("SIF_IMAGES_DIR"),
        )
        
        # Monitoring config
        config.monitoring = MonitoringConfig(
            prometheus_port=int(os.getenv("PROMETHEUS_PORT", "9090")),
            grafana_port=int(os.getenv("GRAFANA_PORT", "3000")),
            grafana_admin_user=os.getenv("GRAFANA_ADMIN_USER", "admin"),
            grafana_admin_password=os.getenv("GRAFANA_ADMIN_PASSWORD", "admin"),
            collection_interval=int(os.getenv("METRICS_COLLECTION_INTERVAL", "15")),
        )
        
        # Web config
        config.web = WebConfig(
            host=os.getenv("WEB_HOST", "0.0.0.0"),
            port=int(os.getenv("WEB_PORT", "5000")),
            debug=os.getenv("WEB_DEBUG", "false").lower() == "true",
        )
        
        return config
    
    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        for dir_path in [
            self.config_dir,
            self.recipes_dir,
            self.results_dir,
            self.logs_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def get_recipe_path(self, recipe_type: str, recipe_name: str) -> Path:
        """
        Get the full path to a recipe file.
        
        Args:
            recipe_type: Type of recipe (servers, clients, monitors, benchmarks)
            recipe_name: Name of the recipe
            
        Returns:
            Path to the recipe YAML file
        """
        recipe_file = f"{recipe_name}.yaml"
        return self.recipes_dir / recipe_type / recipe_file


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config
