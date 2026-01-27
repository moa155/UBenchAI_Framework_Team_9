"""
SLURM Orchestrator for InferBench Framework.

Handles job submission, monitoring, and management on SLURM-based
HPC clusters like MeluXina.
"""

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from inferbench.core.config import get_config
from inferbench.core.exceptions import SlurmError
from inferbench.core.models import ResourceSpec, ServiceStatus, RunStatus
from inferbench.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SlurmJobInfo:
    """Information about a SLURM job."""
    job_id: str
    name: str
    state: str
    node: Optional[str]
    partition: str
    time_used: str
    reason: Optional[str] = None
    
    @property
    def is_running(self) -> bool:
        """Check if job is running."""
        return self.state.upper() == "RUNNING"
    
    @property
    def is_pending(self) -> bool:
        """Check if job is pending."""
        return self.state.upper() in ["PENDING", "PD"]
    
    @property
    def is_completed(self) -> bool:
        """Check if job completed (successfully or not)."""
        return self.state.upper() in ["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "CD", "F", "CA", "TO"]


class SlurmOrchestrator:
    """
    Orchestrator for managing SLURM jobs.
    
    Provides methods for submitting, monitoring, and canceling SLURM jobs.
    """
    
    def __init__(self):
        """Initialize the SLURM orchestrator."""
        self.config = get_config()
        self._check_slurm_available()
    
    def _check_slurm_available(self) -> None:
        """Check if SLURM commands are available."""
        try:
            result = subprocess.run(
                ["sinfo", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                logger.debug("SLURM not available or not configured")
        except FileNotFoundError:
            logger.debug("SLURM commands not found - running in local mode")
        except Exception as e:
            logger.debug(f"SLURM check failed: {e}")
    
    def _run_command(
        self, 
        cmd: list[str], 
        timeout: int = 30,
        check: bool = True
    ) -> subprocess.CompletedProcess:
        """
        Run a shell command and return the result.
        
        Args:
            cmd: Command and arguments
            timeout: Timeout in seconds
            check: Whether to raise on non-zero exit
            
        Returns:
            CompletedProcess result
        """
        logger.debug(f"Running command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if check and result.returncode != 0:
                raise SlurmError(
                    operation=cmd[0],
                    reason=result.stderr or f"Exit code {result.returncode}"
                )
            
            return result
            
        except subprocess.TimeoutExpired:
            raise SlurmError(operation=cmd[0], reason=f"Command timed out after {timeout}s")
        except FileNotFoundError:
            raise SlurmError(operation=cmd[0], reason="Command not found")
    
    def generate_batch_script(
        self,
        job_name: str,
        command: str,
        resources: ResourceSpec,
        environment: dict[str, str],
        output_dir: Path,
        setup_commands: Optional[list[str]] = None,
    ) -> str:
        """
        Generate a SLURM batch script.
        
        Args:
            job_name: Name for the SLURM job
            command: Main command to execute
            resources: Resource requirements
            environment: Environment variables
            output_dir: Directory for output files
            setup_commands: Optional setup commands to run first
            
        Returns:
            Batch script content
        """
        # Build SBATCH directives
        directives = [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --time={resources.time}",
            f"#SBATCH --partition={resources.partition}",
            f"#SBATCH --nodes={resources.nodes}",
            f"#SBATCH --cpus-per-task={resources.cpus_per_task}",
            f"#SBATCH --mem={resources.memory}",
            f"#SBATCH --output={output_dir}/{job_name}_%j.out",
            f"#SBATCH --error={output_dir}/{job_name}_%j.err",
        ]
        
        # Add GPU resources if needed (MeluXina uses --gres format)
        if resources.gpus > 0:
            if resources.gpu_type:
                directives.append(f"#SBATCH --gres=gpu:{resources.gpus}")
            else:
                directives.append(f"#SBATCH --gres=gpu:{resources.gpus}")
        
        # Add account if configured
        if self.config.slurm.account:
            directives.append(f"#SBATCH --account={self.config.slurm.account}")
        
        # Add QOS if configured
        if hasattr(self.config.slurm, 'qos') and self.config.slurm.qos:
            directives.append(f"#SBATCH --qos={self.config.slurm.qos}")
        
        # Build script
        script_lines = directives + [
            "",
            "# Print job information",
            'echo "Job ID: $SLURM_JOB_ID"',
            'echo "Node: $SLURM_NODELIST"',
            'echo "Start time: $(date)"',
            "",
            "# Load modules (MeluXina 2024.1)",
            "module load env/release/2024.1",
            "module load Python/3.11.10-GCCcore-13.3.0",
            "module load Apptainer",
            "",
            "# Set environment variables",
        ]
        
        # Add environment variables
        for key, value in environment.items():
            script_lines.append(f'export {key}="{value}"')
        
        script_lines.append("")
        
        # Add setup commands
        if setup_commands:
            script_lines.append("# Setup commands")
            script_lines.extend(setup_commands)
            script_lines.append("")
        
        # Add main command
        script_lines.extend([
            "# Main command",
            command,
            "",
            "# Print completion",
            'echo "End time: $(date)"',
            'echo "Exit code: $?"',
        ])
        
        return "\n".join(script_lines)
    
    def submit_job(
        self,
        script_content: str,
        script_name: str = "job.sh",
        work_dir: Optional[Path] = None,
    ) -> str:
        """
        Submit a SLURM job.
        
        Args:
            script_content: Batch script content
            script_name: Name for the script file
            work_dir: Working directory for the job
            
        Returns:
            SLURM job ID
        """
        # Create temporary script file
        work_dir = work_dir or Path(tempfile.gettempdir())
        work_dir.mkdir(parents=True, exist_ok=True)
        
        script_path = work_dir / script_name
        script_path.write_text(script_content)
        script_path.chmod(0o755)
        
        logger.debug(f"Created job script: {script_path}")
        
        # Submit job
        result = self._run_command(["sbatch", str(script_path)])
        
        # Parse job ID from output
        # Expected format: "Submitted batch job 12345678"
        match = re.search(r"Submitted batch job (\d+)", result.stdout)
        if not match:
            raise SlurmError(
                operation="submit",
                reason=f"Could not parse job ID from: {result.stdout}"
            )
        
        job_id = match.group(1)
        logger.info(f"Submitted SLURM job: {job_id}")
        
        return job_id
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a SLURM job.
        
        Args:
            job_id: Job ID to cancel
            
        Returns:
            True if cancellation was successful
        """
        try:
            self._run_command(["scancel", job_id])
            logger.info(f"Cancelled SLURM job: {job_id}")
            return True
        except SlurmError as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            return False
    
    def get_job_info(self, job_id: str) -> Optional[SlurmJobInfo]:
        """
        Get information about a SLURM job.
        
        Args:
            job_id: Job ID to query
            
        Returns:
            Job info or None if not found
        """
        try:
            # Use squeue to get job info
            result = self._run_command([
                "squeue",
                "--job", job_id,
                "--noheader",
                "--format=%i|%j|%T|%N|%P|%M|%r"
            ], check=False)
            
            if result.returncode != 0 or not result.stdout.strip():
                # Job not in queue, check sacct for completed jobs
                return self._get_completed_job_info(job_id)
            
            # Parse squeue output
            parts = result.stdout.strip().split("|")
            if len(parts) >= 6:
                return SlurmJobInfo(
                    job_id=parts[0],
                    name=parts[1],
                    state=parts[2],
                    node=parts[3] if parts[3] else None,
                    partition=parts[4],
                    time_used=parts[5],
                    reason=parts[6] if len(parts) > 6 and parts[6] else None
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get job info for {job_id}: {e}")
            return None
    
    def _get_completed_job_info(self, job_id: str) -> Optional[SlurmJobInfo]:
        """Get info for a completed job from sacct."""
        try:
            result = self._run_command([
                "sacct",
                "--job", job_id,
                "--noheader",
                "--parsable2",
                "--format=JobID,JobName,State,NodeList,Partition,Elapsed"
            ], check=False)
            
            if result.returncode != 0 or not result.stdout.strip():
                return None
            
            # Parse first line (main job, not steps)
            for line in result.stdout.strip().split("\n"):
                parts = line.split("|")
                if len(parts) >= 6 and not "." in parts[0]:  # Skip job steps
                    return SlurmJobInfo(
                        job_id=parts[0],
                        name=parts[1],
                        state=parts[2],
                        node=parts[3] if parts[3] else None,
                        partition=parts[4],
                        time_used=parts[5]
                    )
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to get completed job info: {e}")
            return None
    
    def get_job_status(self, job_id: str) -> ServiceStatus:
        """
        Get the status of a job as a ServiceStatus enum.
        
        Args:
            job_id: Job ID to query
            
        Returns:
            ServiceStatus enum value
        """
        job_info = self.get_job_info(job_id)
        
        if job_info is None:
            return ServiceStatus.UNKNOWN
        
        state = job_info.state.upper()
        
        # Map SLURM states to ServiceStatus
        if state in ["RUNNING", "R"]:
            return ServiceStatus.RUNNING
        elif state in ["PENDING", "PD", "CONFIGURING", "CF"]:
            return ServiceStatus.PENDING
        elif state in ["COMPLETING", "CG"]:
            return ServiceStatus.STOPPING
        elif state in ["COMPLETED", "CD"]:
            return ServiceStatus.STOPPED
        elif state in ["FAILED", "F", "TIMEOUT", "TO", "CANCELLED", "CA", "NODE_FAIL", "NF"]:
            return ServiceStatus.ERROR
        else:
            return ServiceStatus.UNKNOWN
    
    def get_job_node(self, job_id: str) -> Optional[str]:
        """Get the node where a job is running."""
        job_info = self.get_job_info(job_id)
        return job_info.node if job_info else None
    
    def get_job_output(self, job_id: str, output_dir: Path, lines: int = 100) -> str:
        """
        Get the output of a job from its output file.
        
        Args:
            job_id: Job ID
            output_dir: Directory containing output files
            lines: Number of lines to return
            
        Returns:
            Job output content
        """
        # Find output file
        pattern = f"*_{job_id}.out"
        output_files = list(output_dir.glob(pattern))
        
        if not output_files:
            return f"No output file found for job {job_id}"
        
        output_file = output_files[0]
        
        try:
            with open(output_file, "r") as f:
                all_lines = f.readlines()
                return "".join(all_lines[-lines:])
        except Exception as e:
            return f"Error reading output: {e}"
    
    def get_job_error(self, job_id: str, output_dir: Path, lines: int = 100) -> str:
        """Get the error output of a job."""
        pattern = f"*_{job_id}.err"
        error_files = list(output_dir.glob(pattern))
        
        if not error_files:
            return f"No error file found for job {job_id}"
        
        error_file = error_files[0]
        
        try:
            with open(error_file, "r") as f:
                all_lines = f.readlines()
                return "".join(all_lines[-lines:])
        except Exception as e:
            return f"Error reading error file: {e}"
    
    def list_user_jobs(self, state: Optional[str] = None) -> list[SlurmJobInfo]:
        """
        List all jobs for the current user.
        
        Args:
            state: Optional state filter (RUNNING, PENDING, etc.)
            
        Returns:
            List of job info objects
        """
        cmd = ["squeue", "--me", "--noheader", "--format=%i|%j|%T|%N|%P|%M|%r"]
        
        if state:
            cmd.extend(["--state", state])
        
        try:
            result = self._run_command(cmd, check=False)
            
            if not result.stdout.strip():
                return []
            
            jobs = []
            for line in result.stdout.strip().split("\n"):
                parts = line.split("|")
                if len(parts) >= 6:
                    jobs.append(SlurmJobInfo(
                        job_id=parts[0],
                        name=parts[1],
                        state=parts[2],
                        node=parts[3] if parts[3] else None,
                        partition=parts[4],
                        time_used=parts[5],
                        reason=parts[6] if len(parts) > 6 and parts[6] else None
                    ))
            
            return jobs
            
        except Exception as e:
            logger.error(f"Failed to list jobs: {e}")
            return []


# Global orchestrator instance
_orchestrator: Optional[SlurmOrchestrator] = None


def get_slurm_orchestrator() -> SlurmOrchestrator:
    """Get the global SLURM orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = SlurmOrchestrator()
    return _orchestrator
