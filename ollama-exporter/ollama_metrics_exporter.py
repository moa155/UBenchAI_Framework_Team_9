#!/usr/bin/env python3
"""
Ollama Metrics Exporter for Prometheus
Exposes Ollama server metrics in Prometheus format
"""

import http.server
import json
import time
import urllib.request
from urllib.error import URLError

OLLAMA_URL = "http://mel2004:11434"
EXPORTER_PORT = 8000

class MetricsHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/metrics":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            metrics = self.collect_metrics()
            self.wfile.write(metrics.encode())
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "healthy"}')
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        pass  # Suppress logging
    
    def collect_metrics(self):
        metrics = []
        
        # Check Ollama health
        try:
            with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as resp:
                data = json.loads(resp.read())
                models = data.get("models", [])
                
                metrics.append("# HELP ollama_up Ollama server health (1=up, 0=down)")
                metrics.append("# TYPE ollama_up gauge")
                metrics.append("ollama_up 1")
                
                metrics.append("# HELP ollama_models_total Number of available models")
                metrics.append("# TYPE ollama_models_total gauge")
                metrics.append(f"ollama_models_total {len(models)}")
                
                metrics.append("# HELP ollama_model_size_bytes Model size in bytes")
                metrics.append("# TYPE ollama_model_size_bytes gauge")
                for model in models:
                    name = model.get("name", "unknown").replace(":", "_")
                    size = model.get("size", 0)
                    metrics.append(f'ollama_model_size_bytes{{model="{name}"}} {size}')
                
                metrics.append("# HELP ollama_model_info Model information")
                metrics.append("# TYPE ollama_model_info gauge")
                for model in models:
                    name = model.get("name", "unknown")
                    family = model.get("details", {}).get("family", "unknown")
                    params = model.get("details", {}).get("parameter_size", "unknown")
                    quant = model.get("details", {}).get("quantization_level", "unknown")
                    metrics.append(f'ollama_model_info{{model="{name}",family="{family}",parameters="{params}",quantization="{quant}"}} 1')
                    
        except URLError:
            metrics.append("# HELP ollama_up Ollama server health (1=up, 0=down)")
            metrics.append("# TYPE ollama_up gauge")
            metrics.append("ollama_up 0")
        
        metrics.append("")
        metrics.append(f"# Scraped at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return "\n".join(metrics)

if __name__ == "__main__":
    print(f"Starting Ollama Metrics Exporter on port {EXPORTER_PORT}")
    print(f"Metrics available at http://localhost:{EXPORTER_PORT}/metrics")
    server = http.server.HTTPServer(("", EXPORTER_PORT), MetricsHandler)
    server.serve_forever()
