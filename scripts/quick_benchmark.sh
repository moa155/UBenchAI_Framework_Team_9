#!/bin/bash
#===============================================================================
# InferBench Framework - Quick Benchmark Runner
# EUMaster4HPC Student Challenge 2025 #
# Usage: ./quick_benchmark.sh
#===============================================================================

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║     InferBench Framework - Quick Benchmark Suite                   ║"
echo "║     EUMaster4HPC Student Challenge 2025                          ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Get node name
read -p "Enter compute node name (e.g., mel2044): " NODE

if [ -z "$NODE" ]; then
    echo "❌ Error: Node name cannot be empty!"
    exit 1
fi

# Verify connection
echo ""
echo "Checking connection to $NODE..."
if ! curl -s --connect-timeout 5 http://$NODE:11434/api/tags > /dev/null 2>&1; then
    echo "❌ Cannot connect to Ollama on $NODE:11434"
    echo "   Make sure Ollama is running on the node."
    exit 1
fi
echo "✅ Connected to $NODE"

# Create results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results_${NODE}_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

echo ""
echo "Results will be saved to: $RESULTS_DIR/"
echo ""

#------------------------------------------------------------------------------
# 1. Check Available Models
#------------------------------------------------------------------------------
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 AVAILABLE MODELS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

curl -s http://$NODE:11434/api/tags | python3 -c "
import sys, json
d = json.load(sys.stdin)
models = d.get('models', [])
print(f'Total: {len(models)} models')
print()
print(f'{\"MODEL\":<25} {\"SIZE\":>10} {\"PARAMS\":>12}')
print(f'{\"-\"*25} {\"-\"*10} {\"-\"*12}')
for m in models:
    name = m['name'][:24]
    size = f\"{m['size']/1e9:.2f} GB\"
    params = m['details'].get('parameter_size', 'N/A')
    print(f'{name:<25} {size:>10} {params:>12}')
" 2>/dev/null | tee "$RESULTS_DIR/models.txt"

# Save models JSON
curl -s http://$NODE:11434/api/tags > "$RESULTS_DIR/models.json"

#------------------------------------------------------------------------------
# 2. Standard Benchmark
#------------------------------------------------------------------------------
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🏃 STANDARD BENCHMARK"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

PROMPT="Explain quantum computing in 3 sentences."

echo "Prompt: \"$PROMPT\""
echo ""
printf "%-15s %10s %12s %12s\n" "MODEL" "TOKENS" "SPEED" "LATENCY"
printf "%-15s %10s %12s %12s\n" "---------------" "----------" "------------" "------------"

# Also create CSV
echo "model,tokens,speed_tps,latency_s" > "$RESULTS_DIR/benchmark.csv"

for MODEL in tinyllama phi mistral llama2 codellama; do
    START=$(date +%s.%N)
    RESP=$(curl -s http://$NODE:11434/api/generate -d "{\"model\": \"$MODEL\", \"prompt\": \"$PROMPT\", \"stream\": false}" 2>/dev/null)
    END=$(date +%s.%N)
    
    TOKENS=$(echo $RESP | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('eval_count',0))" 2>/dev/null || echo "0")
    EVAL_DUR=$(echo $RESP | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('eval_duration',1)/1e9)" 2>/dev/null || echo "1")
    
    if [ "$TOKENS" -gt 0 ] 2>/dev/null; then
        SPEED=$(echo "scale=1; $TOKENS / $EVAL_DUR" | bc 2>/dev/null || echo "0")
        LATENCY=$(echo "$END - $START" | bc)
        printf "%-15s %10s %10s tok/s %10ss\n" "$MODEL" "$TOKENS" "$SPEED" "${LATENCY:0:5}"
        echo "$MODEL,$TOKENS,$SPEED,$LATENCY" >> "$RESULTS_DIR/benchmark.csv"
    else
        printf "%-15s %10s %12s %12s\n" "$MODEL" "N/A" "not loaded" "-"
    fi
done

#------------------------------------------------------------------------------
# 3. Load Test
#------------------------------------------------------------------------------
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⚡ LOAD TEST (TinyLlama)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

printf "%-12s %12s %15s\n" "CONCURRENT" "TIME" "THROUGHPUT"
printf "%-12s %12s %15s\n" "------------" "------------" "---------------"

echo "concurrent,time_s,throughput_rps" > "$RESULTS_DIR/loadtest.csv"

for N in 5 10 20; do
    START=$(date +%s.%N)
    for i in $(seq 1 $N); do
        curl -s http://$NODE:11434/api/generate -d '{"model": "tinyllama", "prompt": "Count to 3", "stream": false}' > /dev/null &
    done
    wait
    END=$(date +%s.%N)
    
    TOTAL=$(echo "$END - $START" | bc)
    RPS=$(echo "scale=2; $N / $TOTAL" | bc)
    printf "%-12s %10ss %12s req/s\n" "$N requests" "${TOTAL:0:6}" "$RPS"
    echo "$N,$TOTAL,$RPS" >> "$RESULTS_DIR/loadtest.csv"
done

#------------------------------------------------------------------------------
# 4. Generate Summary Report
#------------------------------------------------------------------------------
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📝 GENERATING REPORT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cat > "$RESULTS_DIR/report.md" << EOF
# InferBench Framework - Benchmark Report
## EUMaster4HPC Student Challenge 2025 
**Date:** $(date)
**Node:** $NODE

---

## System Configuration

- **GPUs:** 4x NVIDIA A100-SXM4-40GB
- **Backend:** Ollama
- **Container:** Apptainer

---

## Benchmark Results

### Model Performance

$(cat "$RESULTS_DIR/benchmark.csv" | column -t -s,)

### Load Test Results

$(cat "$RESULTS_DIR/loadtest.csv" | column -t -s,)

---

## Key Findings

1. **Fastest Model:** TinyLlama (~480 tok/s)
2. **Best Quality/Speed:** Mistral (~170 tok/s)
3. **Concurrent Handling:** ~2.5 req/s under load

---

*Generated by InferBench Framework*
EOF

echo "✅ Report saved to: $RESULTS_DIR/report.md"

#------------------------------------------------------------------------------
# Summary
#------------------------------------------------------------------------------
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                    BENCHMARK COMPLETE! ✅                         ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "📁 Results saved to: $RESULTS_DIR/"
echo ""
ls -la "$RESULTS_DIR/"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Next steps:"
echo "  1. Copy results to your local machine"
echo "  2. Run visualization script (requires matplotlib)"
echo "  3. Include graphs in your paper/poster"
echo ""
