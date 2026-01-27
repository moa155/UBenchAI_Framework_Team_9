"""Compare benchmark results between runs."""

import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

@dataclass  
class ComparisonResult:
    """Result of comparing two benchmark runs."""
    metric: str
    baseline_value: float
    new_value: float
    absolute_diff: float
    percent_diff: float
    improved: bool

class BenchmarkComparator:
    """Compare benchmark results between baseline and new runs."""
    
    def __init__(self):
        self.baseline = None
        self.new_run = None
    
    def load_baseline(self, path: Path) -> None:
        """Load baseline benchmark results."""
        with open(path) as f:
            self.baseline = json.load(f)
    
    def load_new_run(self, path: Path) -> None:
        """Load new benchmark results."""
        with open(path) as f:
            self.new_run = json.load(f)
    
    def compare_metric(self, metric: str, higher_is_better: bool = True) -> ComparisonResult:
        """Compare a specific metric between baseline and new run."""
        baseline_val = self._extract_metric(self.baseline, metric)
        new_val = self._extract_metric(self.new_run, metric)
        
        abs_diff = new_val - baseline_val
        pct_diff = (abs_diff / baseline_val * 100) if baseline_val != 0 else 0
        
        improved = abs_diff > 0 if higher_is_better else abs_diff < 0
        
        return ComparisonResult(
            metric=metric,
            baseline_value=baseline_val,
            new_value=new_val,
            absolute_diff=abs_diff,
            percent_diff=pct_diff,
            improved=improved,
        )
    
    def _extract_metric(self, data: dict, metric: str) -> float:
        """Extract a metric value from benchmark data."""
        if isinstance(data, list):
            values = [d.get(metric, 0) for d in data if d.get(metric)]
            return sum(values) / len(values) if values else 0
        return data.get(metric, 0)
    
    def generate_comparison_report(self) -> dict:
        """Generate a full comparison report."""
        metrics = [
            ("throughput_tokens_per_sec", True),
            ("latency_ms", False),
            ("error_rate", False),
        ]
        
        comparisons = []
        for metric, higher_better in metrics:
            try:
                comp = self.compare_metric(metric, higher_better)
                comparisons.append({
                    "metric": comp.metric,
                    "baseline": comp.baseline_value,
                    "new": comp.new_value,
                    "diff": comp.absolute_diff,
                    "diff_percent": comp.percent_diff,
                    "improved": comp.improved,
                })
            except Exception:
                pass
        
        return {
            "comparisons": comparisons,
            "overall_improved": sum(1 for c in comparisons if c["improved"]) > len(comparisons) / 2,
        }
