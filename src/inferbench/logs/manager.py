"""
Log Manager for InferBench Framework.

Handles log collection, aggregation, filtering, and export
from SLURM jobs and services.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Iterator
from dataclasses import dataclass, field

from inferbench.core.config import get_config
from inferbench.core.exceptions import ServiceNotFoundError, ClientNotFoundError
from inferbench.core.registry import get_service_registry, get_run_registry
from inferbench.core.slurm import get_slurm_orchestrator
from inferbench.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LogEntry:
    """Represents a single log entry."""
    timestamp: Optional[datetime] = None
    level: str = "INFO"
    source: str = ""
    message: str = ""
    raw: str = ""
    line_number: int = 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "level": self.level,
            "source": self.source,
            "message": self.message,
            "line_number": self.line_number,
        }


@dataclass
class LogCollection:
    """Collection of log entries with metadata."""
    source_id: str
    source_type: str  # "service", "client", "monitor", "job"
    entries: list[LogEntry] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_lines: int = 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "source_id": self.source_id,
            "source_type": self.source_type,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_lines": self.total_lines,
            "entries": [e.to_dict() for e in self.entries],
        }


class LogManager:
    """
    Manages log collection and export for InferBench services.
    
    Collects logs from SLURM jobs, parses them, and provides
    filtering and export capabilities.
    """
    
    # Common log patterns
    LOG_PATTERNS = [
        # Standard Python logging: 2024-01-04 12:30:45 | INFO | module:func:123 - message
        re.compile(
            r'^(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s*\|\s*'
            r'(?P<level>\w+)\s*\|\s*(?P<source>[^|]+)\s*-\s*(?P<message>.*)$'
        ),
        # ISO timestamp with level: 2024-01-04T12:30:45 INFO message
        re.compile(
            r'^(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+'
            r'(?P<level>\w+)\s+(?P<message>.*)$'
        ),
        # Simple timestamp: [2024-01-04 12:30:45] message
        re.compile(
            r'^\[(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\]\s*(?P<message>.*)$'
        ),
        # SLURM style: slurmstepd: error: message
        re.compile(
            r'^(?P<source>slurmstepd|slurmd):\s*(?P<level>\w+):\s*(?P<message>.*)$'
        ),
    ]
    
    TIMESTAMP_FORMATS = [
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
    ]
    
    def __init__(self):
        """Initialize the log manager."""
        self.config = get_config()
        self.service_registry = get_service_registry()
        self.run_registry = get_run_registry()
        self.orchestrator = get_slurm_orchestrator()
        
        # Ensure log directories exist
        self._setup_directories()
        
        logger.info("LogManager initialized")
    
    def _setup_directories(self) -> None:
        """Create required directories."""
        dirs = [
            self.config.logs_dir,
            self.config.logs_dir / "servers",
            self.config.logs_dir / "clients",
            self.config.logs_dir / "monitors",
            self.config.logs_dir / "exports",
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _parse_timestamp(self, ts_str: str) -> Optional[datetime]:
        """Parse a timestamp string."""
        for fmt in self.TIMESTAMP_FORMATS:
            try:
                return datetime.strptime(ts_str.strip(), fmt)
            except ValueError:
                continue
        return None
    
    def _parse_log_line(self, line: str, line_number: int) -> LogEntry:
        """Parse a single log line."""
        line = line.rstrip()
        
        for pattern in self.LOG_PATTERNS:
            match = pattern.match(line)
            if match:
                groups = match.groupdict()
                
                timestamp = None
                if "timestamp" in groups and groups["timestamp"]:
                    timestamp = self._parse_timestamp(groups["timestamp"])
                
                return LogEntry(
                    timestamp=timestamp,
                    level=groups.get("level", "INFO").upper(),
                    source=groups.get("source", "").strip(),
                    message=groups.get("message", line).strip(),
                    raw=line,
                    line_number=line_number,
                )
        
        # No pattern matched - return raw entry
        return LogEntry(
            level="INFO",
            message=line,
            raw=line,
            line_number=line_number,
        )
    
    def _read_log_file(
        self,
        file_path: Path,
        lines: Optional[int] = None,
        tail: bool = True,
    ) -> Iterator[str]:
        """Read lines from a log file."""
        if not file_path.exists():
            return
        
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()
                
                if lines is not None:
                    if tail:
                        all_lines = all_lines[-lines:]
                    else:
                        all_lines = all_lines[:lines]
                
                for line in all_lines:
                    yield line
        except Exception as e:
            logger.error(f"Error reading log file {file_path}: {e}")
    
    def get_service_logs(
        self,
        service_id: str,
        lines: int = 100,
        log_type: str = "output",
        parse: bool = False,
    ) -> LogCollection | str:
        """
        Get logs for a service.
        
        Args:
            service_id: Service ID
            lines: Number of lines to return
            log_type: "output" or "error"
            parse: If True, return parsed LogCollection; else raw string
            
        Returns:
            LogCollection if parse=True, else raw log string
        """
        # Get service from registry
        try:
            service = self.service_registry.get(service_id)
        except ServiceNotFoundError:
            raise
        
        # Determine log file path
        work_dir = self.config.logs_dir / "servers" / service.id
        
        if log_type == "error":
            log_content = self.orchestrator.get_job_error(
                service.slurm_job_id, work_dir, lines
            )
        else:
            log_content = self.orchestrator.get_job_output(
                service.slurm_job_id, work_dir, lines
            )
        
        if not parse:
            return log_content
        
        # Parse logs
        collection = LogCollection(
            source_id=service_id,
            source_type="service",
        )
        
        for i, line in enumerate(log_content.split("\n"), 1):
            if line.strip():
                entry = self._parse_log_line(line, i)
                collection.entries.append(entry)
                
                if entry.timestamp:
                    if not collection.start_time or entry.timestamp < collection.start_time:
                        collection.start_time = entry.timestamp
                    if not collection.end_time or entry.timestamp > collection.end_time:
                        collection.end_time = entry.timestamp
        
        collection.total_lines = len(collection.entries)
        return collection
    
    def get_client_logs(
        self,
        run_id: str,
        lines: int = 100,
        log_type: str = "output",
        parse: bool = False,
    ) -> LogCollection | str:
        """Get logs for a client run."""
        try:
            run = self.run_registry.get(run_id)
        except ClientNotFoundError:
            raise
        
        work_dir = self.config.logs_dir / "clients" / run.id
        
        if log_type == "error":
            log_content = self.orchestrator.get_job_error(
                run.slurm_job_id, work_dir, lines
            )
        else:
            log_content = self.orchestrator.get_job_output(
                run.slurm_job_id, work_dir, lines
            )
        
        if not parse:
            return log_content
        
        collection = LogCollection(
            source_id=run_id,
            source_type="client",
        )
        
        for i, line in enumerate(log_content.split("\n"), 1):
            if line.strip():
                entry = self._parse_log_line(line, i)
                collection.entries.append(entry)
        
        collection.total_lines = len(collection.entries)
        return collection
    
    def get_job_logs(
        self,
        job_id: str,
        lines: int = 100,
        log_type: str = "output",
    ) -> str:
        """
        Get logs directly from a SLURM job.
        
        Args:
            job_id: SLURM job ID
            lines: Number of lines to return
            log_type: "output" or "error"
            
        Returns:
            Raw log content
        """
        # Try to find the log file in common locations
        possible_dirs = [
            self.config.logs_dir / "servers",
            self.config.logs_dir / "clients",
            self.config.logs_dir / "monitors",
            self.config.logs_dir,
            Path.cwd(),
        ]
        
        for base_dir in possible_dirs:
            if base_dir.exists():
                # Look for job output files
                patterns = [
                    f"**/slurm-{job_id}.out",
                    f"**/*_{job_id}.out",
                    f"**/job_{job_id}.out",
                ]
                
                for pattern in patterns:
                    matches = list(base_dir.glob(pattern))
                    if matches:
                        log_file = matches[0]
                        if log_type == "error":
                            error_file = log_file.with_suffix(".err")
                            if error_file.exists():
                                log_file = error_file
                        
                        content = list(self._read_log_file(log_file, lines))
                        return "".join(content)
        
        return f"No log files found for job {job_id}"
    
    def filter_logs(
        self,
        collection: LogCollection,
        level: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        pattern: Optional[str] = None,
        source: Optional[str] = None,
    ) -> LogCollection:
        """
        Filter a log collection.
        
        Args:
            collection: LogCollection to filter
            level: Filter by log level (INFO, WARNING, ERROR, etc.)
            start_time: Filter entries after this time
            end_time: Filter entries before this time
            pattern: Regex pattern to match in message
            source: Filter by source
            
        Returns:
            Filtered LogCollection
        """
        filtered = LogCollection(
            source_id=collection.source_id,
            source_type=collection.source_type,
        )
        
        regex = re.compile(pattern) if pattern else None
        
        for entry in collection.entries:
            # Level filter
            if level and entry.level != level.upper():
                continue
            
            # Time filters
            if start_time and entry.timestamp and entry.timestamp < start_time:
                continue
            if end_time and entry.timestamp and entry.timestamp > end_time:
                continue
            
            # Pattern filter
            if regex and not regex.search(entry.message):
                continue
            
            # Source filter
            if source and source.lower() not in entry.source.lower():
                continue
            
            filtered.entries.append(entry)
        
        filtered.total_lines = len(filtered.entries)
        
        if filtered.entries:
            timestamps = [e.timestamp for e in filtered.entries if e.timestamp]
            if timestamps:
                filtered.start_time = min(timestamps)
                filtered.end_time = max(timestamps)
        
        return filtered
    
    def export_logs(
        self,
        collection: LogCollection,
        output_path: Path,
        format: str = "text",
    ) -> Path:
        """
        Export logs to a file.
        
        Args:
            collection: LogCollection to export
            output_path: Output file path
            format: Export format ("text", "json", "csv")
            
        Returns:
            Path to exported file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            with open(output_path, "w") as f:
                json.dump(collection.to_dict(), f, indent=2, default=str)
        
        elif format == "csv":
            import csv
            with open(output_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["line", "timestamp", "level", "source", "message"])
                for entry in collection.entries:
                    writer.writerow([
                        entry.line_number,
                        entry.timestamp.isoformat() if entry.timestamp else "",
                        entry.level,
                        entry.source,
                        entry.message,
                    ])
        
        else:  # text
            with open(output_path, "w") as f:
                f.write(f"# Log Export: {collection.source_type}/{collection.source_id}\n")
                f.write(f"# Exported: {datetime.now().isoformat()}\n")
                f.write(f"# Total lines: {collection.total_lines}\n")
                f.write("#" + "=" * 70 + "\n\n")
                
                for entry in collection.entries:
                    f.write(entry.raw + "\n")
        
        logger.info(f"Exported {collection.total_lines} log entries to {output_path}")
        return output_path
    
    def export_service_logs(
        self,
        service_id: str,
        output_path: Optional[Path] = None,
        format: str = "text",
        lines: int = 1000,
        include_error: bool = True,
    ) -> Path:
        """
        Export all logs for a service.
        
        Args:
            service_id: Service ID
            output_path: Output file path (auto-generated if None)
            format: Export format
            lines: Number of lines to export
            include_error: Include error logs
            
        Returns:
            Path to exported file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.config.logs_dir / "exports" / f"{service_id}_{timestamp}.{format}"
        
        # Get output logs
        output_logs = self.get_service_logs(service_id, lines, "output", parse=True)
        
        if include_error:
            error_logs = self.get_service_logs(service_id, lines, "error", parse=True)
            # Merge entries
            output_logs.entries.extend(error_logs.entries)
            # Sort by line number or timestamp
            output_logs.entries.sort(key=lambda e: e.line_number)
            output_logs.total_lines = len(output_logs.entries)
        
        return self.export_logs(output_logs, output_path, format)
    
    def tail_logs(
        self,
        service_id: str,
        lines: int = 50,
        log_type: str = "output",
    ) -> str:
        """
        Get the tail of logs (most recent lines).
        
        Args:
            service_id: Service ID
            lines: Number of lines
            log_type: "output" or "error"
            
        Returns:
            Raw log content
        """
        return self.get_service_logs(service_id, lines, log_type, parse=False)
    
    def search_logs(
        self,
        service_id: str,
        pattern: str,
        lines: int = 1000,
        context: int = 2,
    ) -> list[dict]:
        """
        Search logs for a pattern.
        
        Args:
            service_id: Service ID
            pattern: Regex pattern to search for
            lines: Number of lines to search
            context: Number of context lines before/after match
            
        Returns:
            List of matches with context
        """
        logs = self.get_service_logs(service_id, lines, "output", parse=True)
        regex = re.compile(pattern, re.IGNORECASE)
        
        matches = []
        entries = logs.entries
        
        for i, entry in enumerate(entries):
            if regex.search(entry.message) or regex.search(entry.raw):
                # Get context
                start = max(0, i - context)
                end = min(len(entries), i + context + 1)
                
                match_info = {
                    "line": entry.line_number,
                    "match": entry.message,
                    "context_before": [e.raw for e in entries[start:i]],
                    "context_after": [e.raw for e in entries[i+1:end]],
                }
                matches.append(match_info)
        
        return matches
    
    def get_log_stats(self, service_id: str, lines: int = 1000) -> dict:
        """
        Get statistics about logs.
        
        Args:
            service_id: Service ID
            lines: Number of lines to analyze
            
        Returns:
            Statistics dictionary
        """
        logs = self.get_service_logs(service_id, lines, "output", parse=True)
        
        level_counts = {}
        for entry in logs.entries:
            level_counts[entry.level] = level_counts.get(entry.level, 0) + 1
        
        return {
            "total_lines": logs.total_lines,
            "start_time": logs.start_time.isoformat() if logs.start_time else None,
            "end_time": logs.end_time.isoformat() if logs.end_time else None,
            "level_counts": level_counts,
            "sources": list(set(e.source for e in logs.entries if e.source)),
        }


# Global log manager instance
_manager: Optional[LogManager] = None


def get_log_manager() -> LogManager:
    """Get the global log manager instance."""
    global _manager
    if _manager is None:
        _manager = LogManager()
    return _manager
