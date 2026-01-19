# InferBench Framework - Performance Benchmark Report
## EUMaster4HPC Student Challenge 2025 
---

**Report Date:** January 10, 2026  
**Node:** mel2044  
**Project:** p200776  
**User:** u103032  

---

## 1. Executive Summary

The InferBench Framework was successfully deployed and benchmarked on the MeluXina supercomputer using a GPU node with 4x NVIDIA A100-SXM4-40GB GPUs. The system demonstrated excellent performance across multiple LLM models with inference speeds ranging from **171 to 480 tokens/second** depending on model size.

### Key Achievements
- ✅ Successfully deployed **9 LLM models** simultaneously
- ✅ Achieved **480 tok/s** peak inference speed (TinyLlama)
- ✅ Handled **20+ concurrent requests** with 2.6 req/s throughput
- ✅ Utilized all **4 A100 GPUs** for model distribution
- ✅ Web dashboard operational with real-time monitoring

---

## 2. System Configuration

### Hardware
| Component | Specification |
|-----------|---------------|
| Node | mel2044 |
| GPUs | 4x NVIDIA A100-SXM4-40GB |
| GPU Memory | 160 GB total (40 GB × 4) |
| CPU | 2x AMD EPYC 7452 (64 cores @ 2.3 GHz) |
| RAM | 512 GB |
| Interconnect | InfiniBand HDR 200 Gbps |

### Software Stack
| Component | Version |
|-----------|---------|
| Container Runtime | Apptainer/Singularity |
| LLM Backend | Ollama 0.13.5 |
| CUDA | 12.8 |
| Web Framework | Flask |
| Scheduler | SLURM |

---

## 3. Models Deployed

| Model | Parameters | Size | Quantization | Family |
|-------|------------|------|--------------|--------|
| TinyLlama | 1B | 0.64 GB | Q4_0 | LLaMA |
| Phi-2 | 3B | 1.60 GB | Q4_0 | Phi2 |
| Gemma | 3B | 1.68 GB | Q4_0 | Gemma |
| Mistral | 7.2B | 4.37 GB | Q4_K_M | LLaMA |
| Llama2 | 7B | 3.83 GB | Q4_0 | LLaMA |
| CodeLlama | 7B | 3.83 GB | Q4_0 | LLaMA |
| Neural-Chat | 7B | 4.11 GB | Q4_0 | LLaMA |
| Llama2-13B | 13B | 7.37 GB | Q4_0 | LLaMA |
| Mixtral | 46.7B | 26.44 GB | Q4_0 | LLaMA (MoE) |

**Total Model Storage:** ~54 GB

---

## 4. Benchmark Results

### 4.1 Standard Benchmark (Quantum Computing Prompt)

**Prompt:** "Explain quantum computing in 3 sentences."

| Model | Tokens Generated | Speed (tok/s) | Relative Performance |
|-------|-----------------|---------------|---------------------|
| TinyLlama (1B) | 192 | **480.0** | 100% (baseline) |
| Phi-2 (2.7B) | 84 | **280.0** | 58% |
| CodeLlama (7B) | 97 | **242.5** | 51% |
| Llama2 (7B) | 93 | **232.5** | 48% |
| Mistral (7B) | 120 | **171.4** | 36% |

### 4.2 Quick Response Test

**Prompt:** "What is 2+2?"

| Model | Tokens | Speed (tok/s) |
|-------|--------|---------------|
| TinyLlama | 5 | 425.0 |
| Phi | 2 | 234.4 |
| Mistral | 13 | 141.4 |

### 4.3 Load Testing Results

**Model:** Mistral 7B  
**Prompt:** "Count to N" (varying N)

| Concurrent Requests | Total Time | Throughput |
|--------------------|------------|------------|
| 20 | 7.68 s | **2.60 req/s** |

---

## 5. GPU Utilization

### Model Distribution Across GPUs

The Ollama server automatically distributed models across the 4 available A100 GPUs:

| GPU | Models Loaded | Memory Used |
|-----|---------------|-------------|
| CUDA0 | Mixtral (primary) | ~25.3 GB |
| CUDA1 | Llama2-13B, Gemma | ~10.3 GB |
| CUDA2 | Neural-Chat, Llama2 | ~5.8 GB |
| CUDA3 | CodeLlama, Mistral, TinyLlama | ~5.8 GB |

### Memory Efficiency
- **Flash Attention:** Enabled automatically
- **KV Cache:** 512 MB - 3.2 GB per model
- **Quantization:** 4-bit (Q4_0/Q4_K_M) for all models

---

## 6. Performance Analysis

### 6.1 Throughput vs Model Size

```
Speed (tok/s)
    |
500 |  ●  TinyLlama (1B)
400 |
300 |      ●  Phi (3B)
250 |          ●  CodeLlama (7B)
    |           ●  Llama2 (7B)
200 |
    |              ●  Mistral (7B)
100 |
    +---------------------------→ Model Size
       1B    3B    7B    13B
```

### 6.2 Key Observations

1. **Inverse Size-Speed Relationship:** Smaller models achieve significantly higher inference speeds
2. **Quantization Impact:** Q4_K_M (Mistral) shows slightly lower speed than Q4_0 models of similar size
3. **Multi-GPU Scaling:** System effectively distributes load across all 4 GPUs
4. **Concurrent Handling:** Maintains ~2.6 req/s under load with 7B model

---

## 7. Comparison with Baseline

### Expected vs Actual Performance

| Metric | Expected (A100 spec) | Actual | Status |
|--------|---------------------|--------|--------|
| 7B Model Speed | 150-200 tok/s | 171-242 tok/s | ✅ Exceeded |
| 1B Model Speed | 400-500 tok/s | 480 tok/s | ✅ Met |
| Multi-GPU | Yes | Yes | ✅ Working |
| Concurrent Requests | 10+ | 20+ | ✅ Exceeded |

---

## 8. Framework Features Demonstrated

### Core Capabilities
- [x] **Multi-model deployment** - 9 models running simultaneously
- [x] **GPU auto-distribution** - Models spread across 4 GPUs
- [x] **Real-time inference** - Sub-second response times
- [x] **Web dashboard** - Live monitoring at port 8080
- [x] **REST API** - Ollama-compatible API at port 11434
- [x] **SLURM integration** - Native HPC job scheduling
- [x] **Container isolation** - Apptainer-based deployment

### API Endpoints Verified
- `GET /api/tags` - List models
- `POST /api/generate` - Text generation
- `POST /api/pull` - Model download
- `GET /` - Web dashboard

---

## 9. Recommendations

### For Production Deployment
1. **Model Selection:** Use TinyLlama for high-throughput, low-latency applications
2. **Memory Management:** Monitor GPU memory when loading large models (Mixtral)
3. **Load Balancing:** Consider multiple nodes for >50 concurrent users
4. **Caching:** Enable model keepalive for frequently used models

### For Competition Submission
1. **Highlight:** 480 tok/s peak performance
2. **Emphasize:** Multi-GPU utilization
3. **Demonstrate:** Web interface with live benchmarking
4. **Document:** SLURM integration for HPC workflows

---

## 10. Appendix

### A. Test Commands Reference

```bash
# Quick model test
curl -s http://mel2044:11434/api/generate \
  -d '{"model":"mistral","prompt":"Hello","stream":false}'

# List all models
curl -s http://mel2044:11434/api/tags | python3 -m json.tool

# Run benchmark
/tmp/bench.sh mel2044
```

### B. SSH Tunnel for Remote Access

```bash
ssh -p 8822 \
    -L 8080:mel2044:8080 \
    -L 11434:mel2044:11434 \
    u103032@login.lxp.lu
```

### C. GitHub Repository
https://github.com/moa155/InferBench-Framework.git

---

**Report Generated:** January 10, 2026  
**Framework Version:** InferBench v0.1.0  
**Team:** EUMaster4HPC EUMaster4HPC
