# Ollama Metrics Exporter

Exposes Ollama server metrics in Prometheus format.

## Usage
```bash
# Start the exporter (run on the same node as Ollama)
python3 ollama_metrics_exporter.py

# Test metrics
curl http://localhost:8000/metrics
```

## Metrics Exposed

- `ollama_up` - Server health (1=up, 0=down)
- `ollama_models_total` - Number of available models
- `ollama_model_size_bytes` - Size of each model
- `ollama_model_info` - Model metadata (family, parameters, quantization)
