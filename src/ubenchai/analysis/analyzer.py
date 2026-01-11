"""Benchmark result analyzer with statistical analysis."""

import json
import statistics
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

@dataclass
class AnalysisResult:
    """Statistical analysis result."""
    metric: str
    count: int
    mean: float
    median: float
    std_dev: float
    min_val: float
    max_val: float
    p50: float
    p95: float
    p99: float

class BenchmarkAnalyzer:
    """Analyzes benchmark results and generates statistics."""
    
    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = data_path
        self.data = []
    
    def load_data(self, path: Path) -> None:
        """Load benchmark data from JSON file."""
        with open(path) as f:
            self.data = json.load(f)
    
    def analyze_metric(self, values: list[float], metric_name: str) -> AnalysisResult:
        """Perform statistical analysis on a metric."""
        if not values:
            raise ValueError("No values to analyze")
        
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        
        return AnalysisResult(
            metric=metric_name,
            count=n,
            mean=statistics.mean(values),
            median=statistics.median(values),
            std_dev=statistics.stdev(values) if n > 1 else 0,
            min_val=min(values),
            max_val=max(values),
            p50=sorted_vals[int(n * 0.50)],
            p95=sorted_vals[int(n * 0.95)] if n >= 20 else sorted_vals[-1],
            p99=sorted_vals[int(n * 0.99)] if n >= 100 else sorted_vals[-1],
        )
    
    def analyze_throughput(self, results: list[dict]) -> AnalysisResult:
        """Analyze throughput metrics."""
        values = [r.get("tokens_per_second", 0) for r in results if r.get("tokens_per_second")]
        return self.analyze_metric(values, "throughput_tokens_per_sec")
    
    def analyze_latency(self, results: list[dict]) -> AnalysisResult:
        """Analyze latency metrics."""
        values = [r.get("latency_ms", 0) for r in results if r.get("latency_ms")]
        return self.analyze_metric(values, "latency_ms")
    
    def generate_summary(self, results: list[dict]) -> dict:
        """Generate a complete analysis summary."""
        summary = {
            "total_runs": len(results),
            "successful_runs": len([r for r in results if r.get("success", True)]),
            "failed_runs": len([r for r in results if not r.get("success", True)]),
        }
        
        # Analyze by model
        models = {}
        for r in results:
            model = r.get("model", "unknown")
            if model not in models:
                models[model] = []
            models[model].append(r)
        
        summary["models"] = {}
        for model, model_results in models.items():
            tps_values = [r.get("tokens_per_second", 0) for r in model_results if r.get("tokens_per_second")]
            if tps_values:
                summary["models"][model] = {
                    "runs": len(model_results),
                    "avg_throughput": statistics.mean(tps_values),
                    "max_throughput": max(tps_values),
                    "min_throughput": min(tps_values),
                }
        
        return summary
    
    def to_json(self, summary: dict) -> str:
        """Convert summary to JSON string."""
        return json.dumps(summary, indent=2)
