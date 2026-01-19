# InferBench-Framework

**Unified Benchmarking Framework for AI Factory Workloads**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## 🎯 Project Overview

InferBench-Framework is a modular benchmarking framework designed to evaluate the performance of AI Factory components on the **MeluXina supercomputer**. This project is part of the **EUMaster4HPC Student Challenge 2025-2026**.

The framework provides tools for:
- 🖥️ **Server Management**: Deploy and manage AI services (vLLM, Ollama, Vector DBs)
- 📊 **Client Workloads**: Generate benchmark loads against services
- 📈 **Monitoring**: Real-time metrics collection with Prometheus + Grafana
- 📝 **Logging**: Centralized log collection and analysis
- 🎛️ **Interface**: CLI and Web UI for orchestration

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Interface (CLI/Web)                      │
├─────────────┬─────────────┬─────────────┬──────────────────┤
│   Servers   │   Clients   │  Monitors   │       Logs       │
├─────────────┴─────────────┴─────────────┴──────────────────┤
│                    Core Infrastructure                       │
│              (Recipes, Registry, Orchestrator)              │
├─────────────────────────────────────────────────────────────┤
│                  SLURM / Apptainer Runtime                   │
└─────────────────────────────────────────────────────────────┘
```

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- [Poetry](https://python-poetry.org/docs/#installation) 2.0+ for dependency management
- Access to MeluXina supercomputer (for HPC features)

> **Note for Poetry 2.0+**: The `poetry shell` command is no longer available by default. Use `eval $(poetry env activate)` or prefix commands with `poetry run`.

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/moa155/InferBench-Framework.git
   cd InferBench-Framework
   ```

2. **Install dependencies**:
   ```bash
   poetry install
   ```

3. **Activate the virtual environment**:
   ```bash
   # Poetry 2.0+ (recommended)
   eval $(poetry env activate)
   
   # Or use poetry run for individual commands
   poetry run inferbench --help
   ```

4. **Verify installation**:
   ```bash
   inferbench --version
   inferbench --help
   ```

## 🚀 Quick Start

### Start a Server
```bash
# List available server recipes
inferbench server list

# Start a vLLM inference server
inferbench server start --recipe vllm-inference
```

### Run a Benchmark Client
```bash
# Run a stress test against the server
inferbench client run --recipe llm-stress-test
```

### Monitor Services
```bash
# Start monitoring stack (Prometheus + Grafana)
inferbench monitor start --recipe default-monitor --targets <job-id>
```

### View Logs
```bash
# Get logs for a specific service
inferbench logs show --service-id <service-id>
```

## 📁 Project Structure

```
InferBench-Framework/
├── src/inferbench/           # Main source code
│   ├── servers/            # Server module
│   ├── clients/            # Client module
│   ├── monitors/           # Monitor module
│   ├── logs/               # Logs module
│   ├── interface/          # CLI and Web interface
│   ├── core/               # Core infrastructure
│   └── utils/              # Utility functions
├── recipes/                # YAML recipe configurations
│   ├── servers/            # Server recipes
│   ├── clients/            # Client recipes
│   ├── monitors/           # Monitor recipes
│   └── benchmarks/         # Full benchmark recipes
├── config/                 # Configuration files
├── tests/                  # Test suite
├── docs/                   # Documentation
├── templates/              # Report and dashboard templates
├── scripts/                # Utility scripts
├── results/                # Benchmark results
└── examples/               # Example configurations
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# MeluXina Configuration
MELUXINA_USER=your_username
MELUXINA_PROJECT=your_project

# Framework Settings
INFERBENCH_LOG_LEVEL=INFO
INFERBENCH_CONFIG_DIR=/path/to/configs
INFERBENCH_RESULTS_DIR=/path/to/results

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
```

### Recipe Configuration

Recipes are YAML files that define services, clients, and monitoring:

```yaml
# recipes/servers/vllm-inference.yaml
name: vllm-inference
type: server
image: /path/to/vllm.sif
resources:
  nodes: 1
  gpus: 1
  memory: 32G
  time: "02:00:00"
ports:
  - 8000
environment:
  MODEL_NAME: "meta-llama/Llama-2-7b-hf"
healthcheck:
  endpoint: /health
  interval: 30s
```

## 📊 Monitoring

The framework integrates with Prometheus and Grafana for real-time monitoring:

- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization dashboards
- **Custom Exporters**: Service-specific metrics

## 🧪 Testing

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src/inferbench

# Run specific test file
poetry run pytest tests/unit/test_servers.py
```

## 📝 Development

### Code Formatting
```bash
poetry run black src/
poetry run ruff check src/
```

### Type Checking
```bash
poetry run mypy src/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- EUMaster4HPC Program
- LuxProvide and MeluXina Supercomputer
- Dr. Farouk Mansouri for supervision and mentoring

---

**Built with ❤️ for the EUMaster4HPC Student Challenge 2025-2026**
