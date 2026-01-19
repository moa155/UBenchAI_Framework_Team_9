"""
Tests for the Log Manager module.
"""

import pytest
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, patch

from inferbench.logs.manager import (
    LogManager,
    LogEntry,
    LogCollection,
)


class TestLogEntry:
    """Tests for LogEntry dataclass."""
    
    def test_log_entry_creation(self):
        """Should create a log entry."""
        entry = LogEntry(
            timestamp=datetime(2024, 1, 4, 12, 30, 45),
            level="INFO",
            source="test.module",
            message="Test message",
            raw="2024-01-04 12:30:45 | INFO | test.module - Test message",
            line_number=1,
        )
        
        assert entry.level == "INFO"
        assert entry.message == "Test message"
        assert entry.line_number == 1
    
    def test_log_entry_to_dict(self):
        """Should convert to dictionary."""
        entry = LogEntry(
            timestamp=datetime(2024, 1, 4, 12, 30, 45),
            level="ERROR",
            message="Error occurred",
            line_number=42,
        )
        
        d = entry.to_dict()
        
        assert d["level"] == "ERROR"
        assert d["message"] == "Error occurred"
        assert d["line_number"] == 42
        assert "2024-01-04" in d["timestamp"]


class TestLogCollection:
    """Tests for LogCollection dataclass."""
    
    def test_log_collection_creation(self):
        """Should create a log collection."""
        collection = LogCollection(
            source_id="svc-001",
            source_type="service",
        )
        
        assert collection.source_id == "svc-001"
        assert collection.source_type == "service"
        assert collection.entries == []
        assert collection.total_lines == 0
    
    def test_log_collection_with_entries(self):
        """Should handle entries."""
        entry1 = LogEntry(level="INFO", message="First", line_number=1)
        entry2 = LogEntry(level="ERROR", message="Second", line_number=2)
        
        collection = LogCollection(
            source_id="svc-001",
            source_type="service",
            entries=[entry1, entry2],
            total_lines=2,
        )
        
        assert len(collection.entries) == 2
        assert collection.total_lines == 2
    
    def test_log_collection_to_dict(self):
        """Should convert to dictionary."""
        collection = LogCollection(
            source_id="svc-001",
            source_type="service",
            entries=[LogEntry(level="INFO", message="Test", line_number=1)],
            total_lines=1,
        )
        
        d = collection.to_dict()
        
        assert d["source_id"] == "svc-001"
        assert d["source_type"] == "service"
        assert len(d["entries"]) == 1


class TestLogManager:
    """Tests for LogManager class."""
    
    @pytest.fixture
    def manager(self, tmp_path):
        """Create a log manager with temp directories."""
        with patch('inferbench.logs.manager.get_config') as mock_config:
            config = MagicMock()
            config.logs_dir = tmp_path / "logs"
            mock_config.return_value = config
            
            with patch('inferbench.logs.manager.get_service_registry'), \
                 patch('inferbench.logs.manager.get_run_registry'), \
                 patch('inferbench.logs.manager.get_slurm_orchestrator'):
                return LogManager()
    
    def test_parse_timestamp(self, manager):
        """Should parse various timestamp formats."""
        # Format 1: with microseconds
        ts = manager._parse_timestamp("2024-01-04 12:30:45.123456")
        assert ts is not None
        assert ts.year == 2024
        assert ts.month == 1
        assert ts.day == 4
        
        # Format 2: without microseconds
        ts = manager._parse_timestamp("2024-01-04 12:30:45")
        assert ts is not None
        
        # Format 3: ISO format
        ts = manager._parse_timestamp("2024-01-04T12:30:45")
        assert ts is not None
        
        # Invalid format
        ts = manager._parse_timestamp("invalid")
        assert ts is None
    
    def test_parse_log_line_python_format(self, manager):
        """Should parse Python logging format."""
        line = "2024-01-04 12:30:45.123 | INFO | inferbench.core:func:42 - Test message"
        
        entry = manager._parse_log_line(line, 1)
        
        assert entry.level == "INFO"
        assert entry.timestamp is not None
        assert "Test message" in entry.message
        assert entry.line_number == 1
    
    def test_parse_log_line_iso_format(self, manager):
        """Should parse ISO timestamp format."""
        line = "2024-01-04T12:30:45 ERROR Something went wrong"
        
        entry = manager._parse_log_line(line, 5)
        
        assert entry.level == "ERROR"
        assert entry.timestamp is not None
        assert "went wrong" in entry.message
    
    def test_parse_log_line_simple_format(self, manager):
        """Should parse simple bracket format."""
        line = "[2024-01-04 12:30:45] Simple log message"
        
        entry = manager._parse_log_line(line, 10)
        
        assert entry.timestamp is not None
        assert "Simple log message" in entry.message
    
    def test_parse_log_line_slurm_format(self, manager):
        """Should parse SLURM format."""
        line = "slurmstepd: error: Task launch failed"
        
        entry = manager._parse_log_line(line, 1)
        
        assert entry.level == "ERROR"
        assert "slurmstepd" in entry.source
        assert "Task launch" in entry.message
    
    def test_parse_log_line_unknown_format(self, manager):
        """Should handle unknown format."""
        line = "Just some random text"
        
        entry = manager._parse_log_line(line, 1)
        
        assert entry.level == "INFO"
        assert entry.message == "Just some random text"
        assert entry.raw == line
    
    def test_filter_logs_by_level(self, manager):
        """Should filter logs by level."""
        collection = LogCollection(
            source_id="test",
            source_type="service",
            entries=[
                LogEntry(level="INFO", message="Info 1", line_number=1),
                LogEntry(level="ERROR", message="Error 1", line_number=2),
                LogEntry(level="INFO", message="Info 2", line_number=3),
                LogEntry(level="WARNING", message="Warn 1", line_number=4),
            ],
            total_lines=4,
        )
        
        filtered = manager.filter_logs(collection, level="ERROR")
        
        assert filtered.total_lines == 1
        assert filtered.entries[0].message == "Error 1"
    
    def test_filter_logs_by_pattern(self, manager):
        """Should filter logs by pattern."""
        collection = LogCollection(
            source_id="test",
            source_type="service",
            entries=[
                LogEntry(level="INFO", message="Starting server", line_number=1),
                LogEntry(level="INFO", message="Loading model", line_number=2),
                LogEntry(level="INFO", message="Server ready", line_number=3),
            ],
            total_lines=3,
        )
        
        filtered = manager.filter_logs(collection, pattern="[Ss]erver")
        
        assert filtered.total_lines == 2
    
    def test_filter_logs_by_time(self, manager):
        """Should filter logs by time range."""
        collection = LogCollection(
            source_id="test",
            source_type="service",
            entries=[
                LogEntry(
                    timestamp=datetime(2024, 1, 4, 10, 0, 0),
                    level="INFO", message="Early", line_number=1
                ),
                LogEntry(
                    timestamp=datetime(2024, 1, 4, 12, 0, 0),
                    level="INFO", message="Middle", line_number=2
                ),
                LogEntry(
                    timestamp=datetime(2024, 1, 4, 14, 0, 0),
                    level="INFO", message="Late", line_number=3
                ),
            ],
            total_lines=3,
        )
        
        filtered = manager.filter_logs(
            collection,
            start_time=datetime(2024, 1, 4, 11, 0, 0),
            end_time=datetime(2024, 1, 4, 13, 0, 0),
        )
        
        assert filtered.total_lines == 1
        assert filtered.entries[0].message == "Middle"
    
    def test_export_logs_text(self, manager, tmp_path):
        """Should export logs as text."""
        collection = LogCollection(
            source_id="test",
            source_type="service",
            entries=[
                LogEntry(level="INFO", message="Test 1", raw="Test 1", line_number=1),
                LogEntry(level="INFO", message="Test 2", raw="Test 2", line_number=2),
            ],
            total_lines=2,
        )
        
        output_path = tmp_path / "export.txt"
        result = manager.export_logs(collection, output_path, "text")
        
        assert result.exists()
        content = result.read_text()
        assert "Test 1" in content
        assert "Test 2" in content
    
    def test_export_logs_json(self, manager, tmp_path):
        """Should export logs as JSON."""
        collection = LogCollection(
            source_id="test",
            source_type="service",
            entries=[
                LogEntry(level="INFO", message="Test", line_number=1),
            ],
            total_lines=1,
        )
        
        output_path = tmp_path / "export.json"
        result = manager.export_logs(collection, output_path, "json")
        
        assert result.exists()
        data = json.loads(result.read_text())
        assert data["source_id"] == "test"
        assert len(data["entries"]) == 1
    
    def test_export_logs_csv(self, manager, tmp_path):
        """Should export logs as CSV."""
        collection = LogCollection(
            source_id="test",
            source_type="service",
            entries=[
                LogEntry(level="INFO", message="Test", line_number=1),
            ],
            total_lines=1,
        )
        
        output_path = tmp_path / "export.csv"
        result = manager.export_logs(collection, output_path, "csv")
        
        assert result.exists()
        content = result.read_text()
        assert "line,timestamp,level,source,message" in content
        assert "INFO" in content


class TestLogManagerIntegration:
    """Integration-style tests for LogManager."""
    
    @pytest.fixture
    def manager_with_logs(self, tmp_path):
        """Create manager with sample log files."""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        
        # Create sample log file
        (logs_dir / "servers").mkdir()
        service_dir = logs_dir / "servers" / "svc-001"
        service_dir.mkdir()
        
        log_file = service_dir / "slurm-12345678.out"
        log_file.write_text("""2024-01-04 12:30:45 | INFO | startup - Server starting
2024-01-04 12:30:46 | INFO | model - Loading model
2024-01-04 12:30:50 | WARNING | memory - High memory usage
2024-01-04 12:31:00 | INFO | ready - Server ready
2024-01-04 12:31:15 | ERROR | handler - Request failed
""")
        
        with patch('inferbench.logs.manager.get_config') as mock_config:
            config = MagicMock()
            config.logs_dir = logs_dir
            mock_config.return_value = config
            
            with patch('inferbench.logs.manager.get_service_registry'), \
                 patch('inferbench.logs.manager.get_run_registry'), \
                 patch('inferbench.logs.manager.get_slurm_orchestrator'):
                return LogManager()
    
    def test_read_log_file(self, manager_with_logs, tmp_path):
        """Should read log file."""
        log_file = tmp_path / "logs" / "servers" / "svc-001" / "slurm-12345678.out"
        
        lines = list(manager_with_logs._read_log_file(log_file, lines=3))
        
        assert len(lines) == 3
    
    def test_read_log_file_tail(self, manager_with_logs, tmp_path):
        """Should read tail of log file."""
        log_file = tmp_path / "logs" / "servers" / "svc-001" / "slurm-12345678.out"
        
        lines = list(manager_with_logs._read_log_file(log_file, lines=2, tail=True))
        
        assert len(lines) == 2
        assert "ready" in lines[0] or "ERROR" in lines[1]
