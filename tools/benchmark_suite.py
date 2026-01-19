#!/usr/bin/env python3
"""
InferBench Framework - Comprehensive Benchmarking & Visualization Suite
EUMaster4HPC Student Challenge 2025 
This script provides:
1. Automated benchmarking of LLM models
2. Publication-ready graphs and plots
3. Performance comparison analysis
4. Export to multiple formats (PNG, PDF, CSV, JSON)

Usage:
    python3 benchmark_suite.py --node mel2044 --output results/
"""

import json
import csv
import subprocess
import time
import datetime
import argparse
import os
from pathlib import Path

# Check for optional dependencies
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Warning: numpy not installed. Install with: pip install numpy")


class BenchmarkSuite:
    """Complete benchmarking suite for InferBench Framework."""
    
    def __init__(self, node: str, output_dir: str = "results"):
        self.node = node
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {
            "metadata": {
                "node": node,
                "timestamp": self.timestamp,
                "framework": "InferBench",
                "team": "EUMaster4HPC"
            },
            "benchmarks": {}
        }
        
    def run_ollama_benchmark(self, model: str, prompt: str, num_predict: int = 100) -> dict:
        """Run a single benchmark against Ollama API."""
        import urllib.request
        
        url = f"http://{self.node}:11434/api/generate"
        data = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": num_predict}
        }).encode()
        
        req = urllib.request.Request(url, data=data)
        req.add_header('Content-Type', 'application/json')
        
        start_time = time.time()
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read())
                end_time = time.time()
                
                return {
                    "model": model,
                    "success": True,
                    "tokens": result.get("eval_count", 0),
                    "prompt_tokens": result.get("prompt_eval_count", 0),
                    "eval_duration_ns": result.get("eval_duration", 0),
                    "total_duration_ns": result.get("total_duration", 0),
                    "load_duration_ns": result.get("load_duration", 0),
                    "tokens_per_second": result.get("eval_count", 0) / (result.get("eval_duration", 1) / 1e9),
                    "wall_time": end_time - start_time,
                    "response_preview": result.get("response", "")[:100]
                }
        except Exception as e:
            return {
                "model": model,
                "success": False,
                "error": str(e)
            }
    
    def run_model_comparison(self, models: list, prompts: list) -> dict:
        """Run benchmarks across multiple models and prompts."""
        print(f"\n{'='*60}")
        print(f"  MODEL COMPARISON BENCHMARK")
        print(f"  Node: {self.node}")
        print(f"{'='*60}\n")
        
        results = []
        
        for model in models:
            print(f"Testing {model}...")
            model_results = []
            
            for i, prompt in enumerate(prompts):
                print(f"  Prompt {i+1}/{len(prompts)}...", end=" ")
                result = self.run_ollama_benchmark(model, prompt["text"], prompt.get("max_tokens", 100))
                
                if result["success"]:
                    print(f"✓ {result['tokens']} tokens @ {result['tokens_per_second']:.1f} tok/s")
                    model_results.append(result)
                else:
                    print(f"✗ {result.get('error', 'Unknown error')}")
            
            if model_results:
                avg_tps = sum(r["tokens_per_second"] for r in model_results) / len(model_results)
                avg_tokens = sum(r["tokens"] for r in model_results) / len(model_results)
                
                results.append({
                    "model": model,
                    "runs": model_results,
                    "avg_tokens_per_second": avg_tps,
                    "avg_tokens": avg_tokens,
                    "total_runs": len(model_results)
                })
        
        self.results["benchmarks"]["model_comparison"] = results
        return results
    
    def run_load_test(self, model: str, concurrent_levels: list = [1, 5, 10, 20, 50]) -> dict:
        """Run load tests with varying concurrency levels."""
        print(f"\n{'='*60}")
        print(f"  LOAD TEST - {model}")
        print(f"{'='*60}\n")
        
        results = []
        
        for n in concurrent_levels:
            print(f"Testing {n} concurrent requests...", end=" ")
            
            # Use subprocess to run concurrent curl commands
            start = time.time()
            processes = []
            
            for i in range(n):
                cmd = [
                    "curl", "-s", f"http://{self.node}:11434/api/generate",
                    "-d", json.dumps({"model": model, "prompt": f"Count to {i+1}", "stream": False})
                ]
                p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                processes.append(p)
            
            # Wait for all to complete
            for p in processes:
                p.wait()
            
            end = time.time()
            total_time = end - start
            throughput = n / total_time
            
            print(f"✓ {total_time:.2f}s | {throughput:.2f} req/s")
            
            results.append({
                "concurrent_requests": n,
                "total_time": total_time,
                "throughput_rps": throughput,
                "avg_latency": total_time / n
            })
        
        self.results["benchmarks"]["load_test"] = {
            "model": model,
            "results": results
        }
        return results
    
    def run_token_length_benchmark(self, model: str, token_lengths: list = [32, 64, 128, 256, 512]) -> dict:
        """Benchmark performance at different output token lengths."""
        print(f"\n{'='*60}")
        print(f"  TOKEN LENGTH BENCHMARK - {model}")
        print(f"{'='*60}\n")
        
        results = []
        prompt = "Write a detailed explanation of artificial intelligence."
        
        for length in token_lengths:
            print(f"Testing {length} tokens...", end=" ")
            result = self.run_ollama_benchmark(model, prompt, num_predict=length)
            
            if result["success"]:
                print(f"✓ {result['tokens']} tokens @ {result['tokens_per_second']:.1f} tok/s")
                results.append({
                    "target_tokens": length,
                    "actual_tokens": result["tokens"],
                    "tokens_per_second": result["tokens_per_second"],
                    "total_time": result["wall_time"]
                })
            else:
                print(f"✗ {result.get('error', 'Unknown error')}")
        
        self.results["benchmarks"]["token_length"] = {
            "model": model,
            "results": results
        }
        return results
    
    def save_results(self):
        """Save all results to files."""
        # JSON
        json_path = self.output_dir / f"benchmark_results_{self.timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Results saved to: {json_path}")
        
        # CSV summary
        csv_path = self.output_dir / f"benchmark_summary_{self.timestamp}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Model", "Avg Tokens/s", "Avg Tokens", "Runs"])
            
            if "model_comparison" in self.results["benchmarks"]:
                for r in self.results["benchmarks"]["model_comparison"]:
                    writer.writerow([
                        r["model"],
                        f"{r['avg_tokens_per_second']:.1f}",
                        f"{r['avg_tokens']:.0f}",
                        r["total_runs"]
                    ])
        print(f"✓ Summary saved to: {csv_path}")
        
        return json_path, csv_path
    
    def create_visualizations(self):
        """Create publication-ready visualizations."""
        if not HAS_MATPLOTLIB or not HAS_NUMPY:
            print("Skipping visualizations (matplotlib/numpy not installed)")
            return
        
        # Set style for publication
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 11,
            'figure.figsize': (10, 6),
            'figure.dpi': 150
        })
        
        # 1. Model Comparison Bar Chart
        if "model_comparison" in self.results["benchmarks"]:
            self._plot_model_comparison()
        
        # 2. Load Test Line Chart
        if "load_test" in self.results["benchmarks"]:
            self._plot_load_test()
        
        # 3. Token Length Performance
        if "token_length" in self.results["benchmarks"]:
            self._plot_token_length()
        
        print(f"\n✓ Visualizations saved to: {self.output_dir}")
    
    def _plot_model_comparison(self):
        """Create model comparison bar chart."""
        data = self.results["benchmarks"]["model_comparison"]
        
        models = [r["model"] for r in data]
        speeds = [r["avg_tokens_per_second"] for r in data]
        
        # Color palette
        colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']
        
        fig, ax = plt.subplots(figsize=(12, 7))
        bars = ax.bar(models, speeds, color=colors[:len(models)], edgecolor='black', linewidth=1.2)
        
        # Add value labels on bars
        for bar, speed in zip(bars, speeds):
            height = bar.get_height()
            ax.annotate(f'{speed:.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=12, fontweight='bold')
        
        ax.set_xlabel('Model', fontweight='bold')
        ax.set_ylabel('Tokens per Second', fontweight='bold')
        ax.set_title('InferBench Framework - Model Performance Comparison\nMeluXina Supercomputer (4x NVIDIA A100-40GB)', 
                     fontweight='bold', pad=20)
        
        # Add grid
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        # Add footer
        fig.text(0.99, 0.01, f'EUMaster4HPC Challenge 2025 | Node: {self.node}',
                ha='right', fontsize=9, style='italic', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'model_comparison_{self.timestamp}.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / f'model_comparison_{self.timestamp}.pdf', bbox_inches='tight')
        plt.close()
    
    def _plot_load_test(self):
        """Create load test visualization."""
        data = self.results["benchmarks"]["load_test"]["results"]
        model = self.results["benchmarks"]["load_test"]["model"]
        
        concurrent = [r["concurrent_requests"] for r in data]
        throughput = [r["throughput_rps"] for r in data]
        latency = [r["avg_latency"] * 1000 for r in data]  # Convert to ms
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Throughput plot
        ax1.plot(concurrent, throughput, 'o-', color='#2ecc71', linewidth=2.5, markersize=10, label='Throughput')
        ax1.fill_between(concurrent, throughput, alpha=0.3, color='#2ecc71')
        ax1.set_xlabel('Concurrent Requests', fontweight='bold')
        ax1.set_ylabel('Throughput (req/s)', fontweight='bold')
        ax1.set_title(f'Load Test - Throughput\n{model}', fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Latency plot
        ax2.plot(concurrent, latency, 's-', color='#e74c3c', linewidth=2.5, markersize=10, label='Latency')
        ax2.fill_between(concurrent, latency, alpha=0.3, color='#e74c3c')
        ax2.set_xlabel('Concurrent Requests', fontweight='bold')
        ax2.set_ylabel('Average Latency (ms)', fontweight='bold')
        ax2.set_title(f'Load Test - Latency\n{model}', fontweight='bold')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        fig.suptitle('InferBench Framework - Load Testing Results', fontsize=16, fontweight='bold', y=1.02)
        
        fig.text(0.99, 0.01, f'EUMaster4HPC Challenge 2025 | Node: {self.node}',
                ha='right', fontsize=9, style='italic', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'load_test_{self.timestamp}.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / f'load_test_{self.timestamp}.pdf', bbox_inches='tight')
        plt.close()
    
    def _plot_token_length(self):
        """Create token length performance plot."""
        data = self.results["benchmarks"]["token_length"]["results"]
        model = self.results["benchmarks"]["token_length"]["model"]
        
        tokens = [r["actual_tokens"] for r in data]
        speeds = [r["tokens_per_second"] for r in data]
        times = [r["total_time"] for r in data]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Speed vs tokens
        ax1.plot(tokens, speeds, 'o-', color='#3498db', linewidth=2.5, markersize=10)
        ax1.fill_between(tokens, speeds, alpha=0.3, color='#3498db')
        ax1.set_xlabel('Output Tokens', fontweight='bold')
        ax1.set_ylabel('Tokens per Second', fontweight='bold')
        ax1.set_title(f'Generation Speed vs Output Length\n{model}', fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Time vs tokens
        ax2.plot(tokens, times, 's-', color='#9b59b6', linewidth=2.5, markersize=10)
        ax2.fill_between(tokens, times, alpha=0.3, color='#9b59b6')
        ax2.set_xlabel('Output Tokens', fontweight='bold')
        ax2.set_ylabel('Total Time (s)', fontweight='bold')
        ax2.set_title(f'Generation Time vs Output Length\n{model}', fontweight='bold')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        fig.suptitle('InferBench Framework - Token Length Analysis', fontsize=16, fontweight='bold', y=1.02)
        
        fig.text(0.99, 0.01, f'EUMaster4HPC Challenge 2025 | Node: {self.node}',
                ha='right', fontsize=9, style='italic', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'token_length_{self.timestamp}.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / f'token_length_{self.timestamp}.pdf', bbox_inches='tight')
        plt.close()


def create_architecture_diagram():
    """Create framework architecture diagram."""
    if not HAS_MATPLOTLIB:
        print("Skipping architecture diagram (matplotlib not installed)")
        return
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Colors
    colors = {
        'header': '#2c3e50',
        'server': '#27ae60',
        'client': '#3498db',
        'monitor': '#9b59b6',
        'logs': '#e67e22',
        'interface': '#e74c3c',
        'core': '#34495e',
        'hpc': '#1abc9c'
    }
    
    # Title
    ax.text(8, 9.5, 'InferBench Framework Architecture', fontsize=20, fontweight='bold', 
            ha='center', va='center', color=colors['header'])
    ax.text(8, 9.0, 'EUMaster4HPC Student Challenge 2025', fontsize=12, 
            ha='center', va='center', style='italic', color='gray')
    
    # Boxes
    boxes = [
        # (x, y, width, height, label, color)
        (0.5, 6.5, 3, 2, 'Server Module\n\n• Service Management\n• Recipe Loading\n• Health Checks', colors['server']),
        (4, 6.5, 3, 2, 'Client Module\n\n• Benchmark Execution\n• Load Testing\n• Results Collection', colors['client']),
        (7.5, 6.5, 3, 2, 'Monitor Module\n\n• Prometheus\n• Grafana\n• Real-time Metrics', colors['monitor']),
        (11, 6.5, 3, 2, 'Logs Module\n\n• Log Collection\n• Search & Filter\n• Export', colors['logs']),
        (0.5, 3.5, 6.5, 2, 'Core Infrastructure\n\n• SLURM Orchestrator  • Apptainer Runtime  • Recipe System  • Configuration', colors['core']),
        (7.5, 3.5, 6.5, 2, 'Web Interface\n\n• Dashboard  • REST API  • Real-time Updates  • Service Control', colors['interface']),
        (0.5, 0.5, 13.5, 2, 'MeluXina HPC Infrastructure\n\n• 4x NVIDIA A100-40GB GPUs  • SLURM Scheduler  • InfiniBand HDR 200Gbps  • Apptainer Containers', colors['hpc']),
    ]
    
    for x, y, w, h, label, color in boxes:
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", 
                                        facecolor=color, edgecolor='white', linewidth=2, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, fontsize=10, ha='center', va='center', 
                color='white', fontweight='bold', wrap=True)
    
    # Arrows
    arrow_style = dict(arrowstyle='->', color='gray', lw=2)
    
    # Vertical arrows to core
    for x in [2, 5.5, 9, 12.5]:
        ax.annotate('', xy=(x, 5.5), xytext=(x, 6.5), arrowprops=arrow_style)
    
    # Core to HPC
    ax.annotate('', xy=(7, 2.5), xytext=(7, 3.5), arrowprops=arrow_style)
    
    plt.tight_layout()
    plt.savefig('architecture_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('architecture_diagram.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Architecture diagram saved")


def main():
    parser = argparse.ArgumentParser(description='InferBench Benchmark Suite')
    parser.add_argument('--node', type=str, required=True, help='Compute node name (e.g., mel2044)')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--models', type=str, nargs='+', 
                        default=['tinyllama', 'phi', 'mistral', 'llama2:13b', 'llama2:70b', 'codellama:70b', 'deepseek-coder:33b', 'qwen:72b'],
                        help='Models to benchmark')
    parser.add_argument('--skip-load-test', action='store_true', help='Skip load testing')
    parser.add_argument('--skip-visualizations', action='store_true', help='Skip creating visualizations')
    parser.add_argument('--architecture-only', action='store_true', help='Only create architecture diagram')
    
    args = parser.parse_args()
    
    if args.architecture_only:
        create_architecture_diagram()
        return
    
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║        InferBench Framework - Comprehensive Benchmark Suite        ║
║        EUMaster4HPC Student Challenge 2025              ║
╚══════════════════════════════════════════════════════════════════╝

Node: {args.node}
Output: {args.output}
Models: {', '.join(args.models)}
""")
    
    suite = BenchmarkSuite(args.node, args.output)
    
    # Define test prompts
    prompts = [
        {"text": "Explain quantum computing in 3 sentences.", "max_tokens": 100},
        {"text": "Write a Python function to sort a list.", "max_tokens": 150},
        {"text": "What are the benefits of renewable energy?", "max_tokens": 120},
    ]
    
    # Run benchmarks
    suite.run_model_comparison(args.models, prompts)
    
    if not args.skip_load_test:
        suite.run_load_test("tinyllama", [1, 5, 10, 20])
    
    suite.run_token_length_benchmark("mistral", [32, 64, 128, 256])
    
    # Save results
    suite.save_results()
    
    # Create visualizations
    if not args.skip_visualizations:
        suite.create_visualizations()
    
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                    BENCHMARK COMPLETE! ✅                         ║
╚══════════════════════════════════════════════════════════════════╝

Results saved to: {args.output}/
- benchmark_results_*.json (raw data)
- benchmark_summary_*.csv (summary table)
- model_comparison_*.png/pdf (bar chart)
- load_test_*.png/pdf (throughput/latency)
- token_length_*.png/pdf (scaling analysis)
""")


if __name__ == "__main__":
    main()
