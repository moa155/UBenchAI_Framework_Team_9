#!/bin/bash
# InferBench-Framework Directory Structure Setup

# Create main source directories
mkdir -p src/inferbench/{servers,clients,monitors,logs,interface,core,utils}
mkdir -p src/inferbench/interface/{cli,web}

# Create supporting directories  
mkdir -p config
mkdir -p recipes/{servers,clients,monitors,benchmarks}
mkdir -p tests/{unit,integration}
mkdir -p docs/{design,api,guides}
mkdir -p scripts
mkdir -p templates/{reports,dashboards}
mkdir -p results
mkdir -p examples

# Create __init__.py files for Python packages
touch src/inferbench/__init__.py
touch src/inferbench/servers/__init__.py
touch src/inferbench/clients/__init__.py
touch src/inferbench/monitors/__init__.py
touch src/inferbench/logs/__init__.py
touch src/inferbench/interface/__init__.py
touch src/inferbench/interface/cli/__init__.py
touch src/inferbench/interface/web/__init__.py
touch src/inferbench/core/__init__.py
touch src/inferbench/utils/__init__.py
touch tests/__init__.py
touch tests/unit/__init__.py
touch tests/integration/__init__.py

echo "✅ Directory structure created successfully!"
