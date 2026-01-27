"""
Apptainer/Singularity container runtime integration.

Provides utilities for building container commands, managing binds,
and handling GPU passthrough for AI workloads.
"""

import os
import shutil
from pathlib import Path
from typing import Optional

from inferbench.core.config import get_config
from inferbench.core.exceptions import ContainerError
from inferbench.core.models import ContainerSpec, ResourceSpec
from inferbench.utils.logging import get_logger

logger = get_logger(__name__)


class ApptainerRuntime:
    """
    Apptainer/Singularity container runtime manager.
    
    Handles container command generation, GPU passthrough, and bind mounts.
    """
    
    # Default bind mounts for MeluXina
    DEFAULT_BINDS = [
        "/tmp:/tmp",
        "/dev/shm:/dev/shm",
    ]
    
    # GPU-related environment variables
    GPU_ENV_VARS = [
        "NVIDIA_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES",
    ]
    
    def __init__(self, runtime: str = "apptainer"):
        """
        Initialize the container runtime.
        
        Args:
            runtime: Runtime to use (apptainer or singularity)
        """
        self.config = get_config()
        self.runtime = runtime or self.config.container.runtime
        self._check_runtime_available()
    
    def _check_runtime_available(self) -> None:
        """Check if the container runtime is available."""
        if not shutil.which(self.runtime):
            logger.debug(f"{self.runtime} not found in PATH")
    
    def validate_image(self, image_path: str) -> bool:
        """
        Validate that a container image exists and is readable.
        
        Args:
            image_path: Path to the .sif image
            
        Returns:
            True if image is valid
        """
        path = Path(image_path)
        
        if not path.exists():
            logger.error(f"Container image not found: {image_path}")
            return False
        
        if not path.suffix == ".sif":
            logger.warning(f"Image does not have .sif extension: {image_path}")
        
        if not os.access(path, os.R_OK):
            logger.error(f"Container image not readable: {image_path}")
            return False
        
        return True
    
    def get_bind_args(
        self, 
        container_spec: ContainerSpec,
        extra_binds: Optional[list[str]] = None
    ) -> list[str]:
        """
        Generate bind mount arguments for the container.
        
        Args:
            container_spec: Container specification
            extra_binds: Additional bind mounts
            
        Returns:
            List of bind arguments
        """
        binds = set(self.DEFAULT_BINDS)
        
        # Add container spec binds
        for bind in container_spec.binds:
            binds.add(bind)
        
        # Add extra binds
        if extra_binds:
            for bind in extra_binds:
                binds.add(bind)
        
        # Add config-specified bind paths
        for bind in self.config.container.bind_paths:
            binds.add(bind)
        
        # Generate arguments
        args = []
        for bind in sorted(binds):
            args.extend(["--bind", bind])
        
        return args
    
    def get_gpu_args(self, resources: ResourceSpec) -> list[str]:
        """
        Generate GPU passthrough arguments.
        
        Args:
            resources: Resource specification
            
        Returns:
            List of GPU arguments
        """
        if resources.gpus <= 0:
            return []
        
        return ["--nv"]  # Enable NVIDIA GPU support
    
    def get_env_args(self, environment: dict[str, str]) -> list[str]:
        """
        Generate environment variable arguments.
        
        Args:
            environment: Environment variables
            
        Returns:
            List of environment arguments
        """
        args = []
        
        for key, value in environment.items():
            args.extend(["--env", f"{key}={value}"])
        
        return args
    
    def build_exec_command(
        self,
        container_spec: ContainerSpec,
        resources: ResourceSpec,
        command: str,
        environment: Optional[dict[str, str]] = None,
        extra_binds: Optional[list[str]] = None,
        work_dir: Optional[str] = None,
    ) -> list[str]:
        """
        Build a complete container exec command.
        
        Args:
            container_spec: Container specification
            resources: Resource requirements
            command: Command to run inside container
            environment: Environment variables
            extra_binds: Additional bind mounts
            work_dir: Working directory inside container
            
        Returns:
            Complete command as list of arguments
        """
        cmd = [self.runtime, "exec"]
        
        # Add bind mounts
        cmd.extend(self.get_bind_args(container_spec, extra_binds))
        
        # Add GPU support
        cmd.extend(self.get_gpu_args(resources))
        
        # Add environment variables
        if environment:
            cmd.extend(self.get_env_args(environment))
        
        # Add working directory
        if work_dir:
            cmd.extend(["--pwd", work_dir])
        
        # Add cleanenv to start with clean environment
        cmd.append("--cleanenv")
        
        # Add the image path
        cmd.append(container_spec.image)
        
        # Add the command (as shell command)
        cmd.extend(["bash", "-c", command])
        
        return cmd
    
    def build_run_command(
        self,
        container_spec: ContainerSpec,
        resources: ResourceSpec,
        environment: Optional[dict[str, str]] = None,
        extra_binds: Optional[list[str]] = None,
    ) -> list[str]:
        """
        Build a container run command (uses container's default entrypoint).
        
        Args:
            container_spec: Container specification
            resources: Resource requirements
            environment: Environment variables
            extra_binds: Additional bind mounts
            
        Returns:
            Complete command as list of arguments
        """
        cmd = [self.runtime, "run"]
        
        # Add bind mounts
        cmd.extend(self.get_bind_args(container_spec, extra_binds))
        
        # Add GPU support
        cmd.extend(self.get_gpu_args(resources))
        
        # Add environment variables
        if environment:
            cmd.extend(self.get_env_args(environment))
        
        # Add cleanenv
        cmd.append("--cleanenv")
        
        # Add the image path
        cmd.append(container_spec.image)
        
        return cmd
    
    def build_shell_command(
        self,
        container_spec: ContainerSpec,
        resources: ResourceSpec,
        extra_binds: Optional[list[str]] = None,
    ) -> list[str]:
        """
        Build a command to get an interactive shell in the container.
        
        Args:
            container_spec: Container specification
            resources: Resource requirements
            extra_binds: Additional bind mounts
            
        Returns:
            Shell command as list of arguments
        """
        cmd = [self.runtime, "shell"]
        
        # Add bind mounts
        cmd.extend(self.get_bind_args(container_spec, extra_binds))
        
        # Add GPU support
        cmd.extend(self.get_gpu_args(resources))
        
        # Add the image path
        cmd.append(container_spec.image)
        
        return cmd
    
    def generate_exec_script(
        self,
        container_spec: ContainerSpec,
        resources: ResourceSpec,
        command: str,
        environment: Optional[dict[str, str]] = None,
        extra_binds: Optional[list[str]] = None,
        work_dir: Optional[str] = None,
    ) -> str:
        """
        Generate a shell script command string for container execution.
        
        This is useful for embedding in SLURM batch scripts.
        
        Args:
            container_spec: Container specification
            resources: Resource requirements
            command: Command to run inside container
            environment: Environment variables
            extra_binds: Additional bind mounts
            work_dir: Working directory
            
        Returns:
            Shell command string
        """
        cmd_parts = [self.runtime, "exec"]
        
        # Add bind mounts
        for bind in self._get_unique_binds(container_spec, extra_binds):
            cmd_parts.append(f"--bind {bind}")
        
        # Add GPU support
        if resources.gpus > 0:
            cmd_parts.append("--nv")
        
        # Add environment variables
        if environment:
            for key, value in environment.items():
                # Escape quotes in value
                escaped_value = value.replace('"', '\\"')
                cmd_parts.append(f'--env {key}="{escaped_value}"')
        
        # Add working directory
        if work_dir:
            cmd_parts.append(f"--pwd {work_dir}")
        
        # Add cleanenv
        cmd_parts.append("--cleanenv")
        
        # Add image
        cmd_parts.append(container_spec.image)
        
        # Add command
        # Escape the command for embedding in bash
        escaped_command = command.replace("'", "'\"'\"'")
        cmd_parts.append(f"bash -c '{escaped_command}'")
        
        return " \\\n    ".join(cmd_parts)
    
    def _get_unique_binds(
        self, 
        container_spec: ContainerSpec, 
        extra_binds: Optional[list[str]]
    ) -> list[str]:
        """Get unique bind mounts."""
        binds = set(self.DEFAULT_BINDS)
        binds.update(container_spec.binds)
        if extra_binds:
            binds.update(extra_binds)
        binds.update(self.config.container.bind_paths)
        return sorted(binds)
    
    def pull_image(self, docker_image: str, output_path: Path) -> bool:
        """
        Pull a Docker image and convert to SIF format.
        
        Args:
            docker_image: Docker image reference (e.g., "vllm/vllm-openai:latest")
            output_path: Path for the output .sif file
            
        Returns:
            True if successful
        """
        import subprocess
        
        try:
            cmd = [
                self.runtime, "pull",
                str(output_path),
                f"docker://{docker_image}"
            ]
            
            logger.info(f"Pulling image: {docker_image}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout for large images
            )
            
            if result.returncode != 0:
                raise ContainerError(
                    operation="pull",
                    image=docker_image,
                    reason=result.stderr
                )
            
            logger.info(f"Image pulled successfully: {output_path}")
            return True
            
        except subprocess.TimeoutExpired:
            raise ContainerError(
                operation="pull",
                image=docker_image,
                reason="Image pull timed out"
            )
        except Exception as e:
            raise ContainerError(
                operation="pull",
                image=docker_image,
                reason=str(e)
            )


# Global runtime instance
_runtime: Optional[ApptainerRuntime] = None


def get_apptainer_runtime() -> ApptainerRuntime:
    """Get the global Apptainer runtime instance."""
    global _runtime
    if _runtime is None:
        _runtime = ApptainerRuntime()
    return _runtime
