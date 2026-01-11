"""
Tests for SLURM and Apptainer modules.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from ubenchai.core.slurm import SlurmOrchestrator, SlurmJobInfo
from ubenchai.core.apptainer import ApptainerRuntime
from ubenchai.core.models import ResourceSpec, ContainerSpec, ServiceStatus


class TestSlurmOrchestrator:
    """Tests for SlurmOrchestrator class."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create a SLURM orchestrator."""
        with patch.object(SlurmOrchestrator, '_check_slurm_available'):
            return SlurmOrchestrator()
    
    def test_generate_batch_script(self, orchestrator, tmp_path):
        """Should generate a valid batch script."""
        resources = ResourceSpec(
            nodes=1, gpus=2, memory="64G",
            time="04:00:00", partition="gpu"
        )
        
        script = orchestrator.generate_batch_script(
            job_name="test-job",
            command="python train.py",
            resources=resources,
            environment={"MODEL": "llama"},
            output_dir=tmp_path
        )
        
        assert "#!/bin/bash" in script
        assert "#SBATCH --job-name=test-job" in script
        assert "#SBATCH --gres=gpu:2" in script
        assert 'export MODEL="llama"' in script
    
    @patch('subprocess.run')
    def test_submit_job(self, mock_run, orchestrator, tmp_path):
        """Should submit job and return job ID."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout="Submitted batch job 12345678", stderr=""
        )
        
        job_id = orchestrator.submit_job(
            script_content="#!/bin/bash\necho hello",
            work_dir=tmp_path
        )
        
        assert job_id == "12345678"
    
    @patch('subprocess.run')
    def test_cancel_job(self, mock_run, orchestrator):
        """Should cancel a job."""
        mock_run.return_value = MagicMock(returncode=0)
        result = orchestrator.cancel_job("12345678")
        assert result is True
    
    @patch('subprocess.run')
    def test_get_job_info(self, mock_run, orchestrator):
        """Should parse job info from squeue."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="12345|test-job|RUNNING|mel2091|gpu|01:30:00|"
        )
        
        job_info = orchestrator.get_job_info("12345")
        
        assert job_info.job_id == "12345"
        assert job_info.state == "RUNNING"
        assert job_info.is_running is True


class TestApptainerRuntime:
    """Tests for ApptainerRuntime class."""
    
    @pytest.fixture
    def runtime(self):
        """Create an Apptainer runtime."""
        with patch('shutil.which', return_value='/usr/bin/apptainer'):
            return ApptainerRuntime()
    
    @pytest.fixture
    def container_spec(self):
        """Create a sample container spec."""
        return ContainerSpec(
            image="/path/to/image.sif",
            binds=["/data:/data:ro"]
        )
    
    def test_get_bind_args(self, runtime, container_spec):
        """Should generate bind mount arguments."""
        args = runtime.get_bind_args(container_spec)
        assert "--bind" in args
        assert "/data:/data:ro" in args
    
    def test_get_gpu_args(self, runtime):
        """Should generate GPU arguments."""
        resources = ResourceSpec(gpus=1)
        assert runtime.get_gpu_args(resources) == ["--nv"]
        
        resources = ResourceSpec(gpus=0)
        assert runtime.get_gpu_args(resources) == []
    
    def test_build_exec_command(self, runtime, container_spec):
        """Should build complete exec command."""
        resources = ResourceSpec(gpus=1)
        cmd = runtime.build_exec_command(
            container_spec=container_spec,
            resources=resources,
            command="python train.py",
            environment={"MODEL": "llama"}
        )
        
        assert cmd[0] == "apptainer"
        assert cmd[1] == "exec"
        assert "--nv" in cmd
        assert container_spec.image in cmd
