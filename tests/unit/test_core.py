"""
Tests for the core module of InferBench Framework.
"""

import pytest
from pathlib import Path

from inferbench import __version__
from inferbench.core.config import Config, get_config, SlurmConfig
from inferbench.core.exceptions import (
    InferBenchError,
    RecipeNotFoundError,
    ServiceNotFoundError,
    SlurmError,
)


class TestVersion:
    """Test version information."""
    
    def test_version_exists(self):
        """Version should be defined."""
        assert __version__ is not None
        assert isinstance(__version__, str)
    
    def test_version_format(self):
        """Version should follow semver format."""
        parts = __version__.split(".")
        assert len(parts) >= 2
        assert all(p.isdigit() for p in parts[:2])


class TestConfig:
    """Test configuration management."""
    
    def test_default_config(self):
        """Default config should have sensible defaults."""
        config = Config()
        assert config.log_level == "INFO"
        assert config.slurm.partition == "gpu"
        assert config.monitoring.prometheus_port == 9090
        assert config.monitoring.grafana_port == 3000
    
    def test_config_from_env(self, monkeypatch):
        """Config should load from environment variables."""
        monkeypatch.setenv("INFERBENCH_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("MELUXINA_USER", "testuser")
        monkeypatch.setenv("PROMETHEUS_PORT", "9999")
        
        config = Config.from_env()
        assert config.log_level == "DEBUG"
        assert config.meluxina_user == "testuser"
        assert config.monitoring.prometheus_port == 9999
    
    def test_slurm_config_defaults(self):
        """SLURM config should have proper defaults."""
        slurm = SlurmConfig()
        assert slurm.partition == "gpu"
        assert slurm.default_nodes == 1
        assert slurm.default_gpus == 1
        assert slurm.default_memory == "32G"
    
    def test_get_recipe_path(self):
        """Should return correct recipe path."""
        config = Config()
        path = config.get_recipe_path("servers", "vllm-inference")
        assert path.name == "vllm-inference.yaml"
        assert "servers" in str(path)


class TestExceptions:
    """Test custom exceptions."""
    
    def test_base_exception(self):
        """Base exception should work correctly."""
        err = InferBenchError("Test error", {"key": "value"})
        assert "Test error" in str(err)
        assert err.message == "Test error"
        assert err.details == {"key": "value"}
    
    def test_recipe_not_found(self):
        """RecipeNotFoundError should format correctly."""
        err = RecipeNotFoundError("test-recipe", "servers")
        assert "test-recipe" in str(err)
        assert "servers" in str(err)
        assert err.details["recipe_name"] == "test-recipe"
    
    def test_service_not_found(self):
        """ServiceNotFoundError should format correctly."""
        err = ServiceNotFoundError("abc123")
        assert "abc123" in str(err)
        assert err.details["service_id"] == "abc123"
    
    def test_slurm_error(self):
        """SlurmError should include operation and job_id."""
        err = SlurmError("submit", "quota exceeded", job_id="12345")
        assert "submit" in str(err)
        assert "quota exceeded" in str(err)
        assert err.details["job_id"] == "12345"


class TestConfigDirectories:
    """Test configuration directory management."""
    
    def test_ensure_directories(self, tmp_path):
        """ensure_directories should create all required dirs."""
        config = Config(
            config_dir=tmp_path / "config",
            recipes_dir=tmp_path / "recipes",
            results_dir=tmp_path / "results",
            logs_dir=tmp_path / "logs",
        )
        
        # Directories shouldn't exist yet
        assert not config.config_dir.exists()
        
        # Create them
        config.ensure_directories()
        
        # Now they should exist
        assert config.config_dir.exists()
        assert config.recipes_dir.exists()
        assert config.results_dir.exists()
        assert config.logs_dir.exists()
