#!/bin/bash
#===============================================================================
# InferBench Framework - Complete Performance Testing Suite
# EUMaster4HPC Student Challenge 2025
#===============================================================================

# Prompt user for node name
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║        InferBench Framework - Complete Test Suite                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

read -p "Enter the compute node name (e.g., mel2044): " NODE

if [ -z "$NODE" ]; then
    echo "❌ Error: Node name cannot be empty!"
    exit 1
fi

# Verify node is reachable
echo ""
echo "Checking connection to $NODE..."
if ! curl -s --connect-timeout 5 http://$NODE:11434/api/tags > /dev/null 2>&1; then
    echo "❌ Error: Cannot connect to Ollama on $NODE:11434"
    echo "   Make sure Ollama is running on the node."
    exit 1
fi
echo "✅ Connected to $NODE"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="/tmp/inferbench_results_${NODE}_$TIMESTAMP"
mkdir -p $RESULTS_DIR

echo ""
echo "Node: $NODE | Time: $(date)"
echo "Results will be saved to: $RESULTS_DIR"
echo ""

#------------------------------------------------------------------------------
# 1. SYSTEM STATUS
#------------------------------------------------------------------------------
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 [1/6] SYSTEM STATUS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
echo "Available Models:"
curl -s http://$NODE:11434/api/tags | python3 -c "
import sys, json
d = json.load(sys.stdin)
models = d.get('models', [])
print(f'  Total: {len(models)} models')
print('')
print(f'  {\"MODEL\":<25} {\"SIZE\":>10} {\"PARAMS\":>10}')
print(f'  {\"-\"*25} {\"-\"*10} {\"-\"*10}')
for m in models:
    name = m['name'][:24]
    size = f\"{m['size']/1e9:.2f} GB\"
    params = m['details'].get('parameter_size', 'N/A')
    print(f'  {name:<25} {size:>10} {params:>10}')
" 2>/dev/null

curl -s http://$NODE:11434/api/tags > $RESULTS_DIR/models.json

#------------------------------------------------------------------------------
# 2. STANDARD BENCHMARK
#------------------------------------------------------------------------------
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🏃 [2/6] STANDARD BENCHMARK (Quantum Computing Prompt)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

PROMPT="Explain quantum computing in 3 sentences."
echo ""
printf "  %-15s %10s %12s %12s\n" "MODEL" "TOKENS" "SPEED" "LATENCY"
printf "  %-15s %10s %12s %12s\n" "---------------" "----------" "------------" "------------"

BENCH_RESULTS=""
for MODEL in tinyllama phi mistral llama2 codellama; do
    START=$(date +%s.%N)
    RESP=$(curl -s http://$NODE:11434/api/generate -d "{\"model\": \"$MODEL\", \"prompt\": \"$PROMPT\", \"stream\": false}" 2>/dev/null)
    END=$(date +%s.%N)
    
    TOKENS=$(echo $RESP | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('eval_count',0))" 2>/dev/null || echo "0")
    EVAL_DUR=$(echo $RESP | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('eval_duration',1)/1e9)" 2>/dev/null || echo "1")
    
    if [ "$TOKENS" -gt 0 ] 2>/dev/null; then
        SPEED=$(echo "scale=1; $TOKENS / $EVAL_DUR" | bc 2>/dev/null || echo "0")
        LATENCY=$(echo "$END - $START" | bc)
        printf "  %-15s %10s %10s tok/s %10ss\n" "$MODEL" "$TOKENS" "$SPEED" "${LATENCY:0:5}"
        BENCH_RESULTS="$BENCH_RESULTS$MODEL,$TOKENS,$SPEED,$LATENCY\n"
    else
        printf "  %-15s %10s %12s %12s\n" "$MODEL" "N/A" "Not loaded" "-"
    fi
done

echo -e "model,tokens,speed,latency\n$BENCH_RESULTS" > $RESULTS_DIR/benchmark_standard.csv

#------------------------------------------------------------------------------
# 3. EXTENDED BENCHMARK (Long Output)
#------------------------------------------------------------------------------
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📝 [3/6] EXTENDED BENCHMARK (256 Token Output)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

LONG_PROMPT="Write a detailed explanation of neural networks including backpropagation."
echo ""
printf "  %-15s %10s %12s %12s\n" "MODEL" "TOKENS" "SPEED" "TIME"
printf "  %-15s %10s %12s %12s\n" "---------------" "----------" "------------" "------------"

for MODEL in tinyllama mistral llama2; do
    START=$(date +%s.%N)
    RESP=$(curl -s http://$NODE:11434/api/generate -d "{\"model\": \"$MODEL\", \"prompt\": \"$LONG_PROMPT\", \"stream\": false, \"options\": {\"num_predict\": 256}}" 2>/dev/null)
    END=$(date +%s.%N)
    
    TOKENS=$(echo $RESP | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('eval_count',0))" 2>/dev/null || echo "0")
    EVAL_DUR=$(echo $RESP | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('eval_duration',1)/1e9)" 2>/dev/null || echo "1")
    
    if [ "$TOKENS" -gt 0 ] 2>/dev/null; then
        SPEED=$(echo "scale=1; $TOKENS / $EVAL_DUR" | bc 2>/dev/null || echo "0")
        TOTAL=$(echo "$END - $START" | bc)
        printf "  %-15s %10s %10s tok/s %10ss\n" "$MODEL" "$TOKENS" "$SPEED" "${TOTAL:0:5}"
    fi
done

#------------------------------------------------------------------------------
# 4. LOAD TESTING
#------------------------------------------------------------------------------
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⚡ [4/6] LOAD TESTING (Concurrent Requests)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
printf "  %-12s %12s %15s\n" "CONCURRENT" "TIME" "THROUGHPUT"
printf "  %-12s %12s %15s\n" "------------" "------------" "---------------"

LOAD_RESULTS=""
for N in 5 10 20; do
    START=$(date +%s.%N)
    for i in $(seq 1 $N); do
        curl -s http://$NODE:11434/api/generate -d '{"model": "tinyllama", "prompt": "Count to 3", "stream": false}' > /dev/null &
    done
    wait
    END=$(date +%s.%N)
    
    TOTAL=$(echo "$END - $START" | bc)
    RPS=$(echo "scale=2; $N / $TOTAL" | bc)
    printf "  %-12s %10ss %12s req/s\n" "$N requests" "${TOTAL:0:6}" "$RPS"
    LOAD_RESULTS="$LOAD_RESULTS$N,$TOTAL,$RPS\n"
done

echo -e "concurrent,time,rps\n$LOAD_RESULTS" > $RESULTS_DIR/load_test.csv

#------------------------------------------------------------------------------
# 5. GPU MEMORY CHECK
#------------------------------------------------------------------------------
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎮 [5/6] GPU STATUS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

ssh -o ConnectTimeout=5 $NODE "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader" 2>/dev/null | while read line; do
    echo "  GPU $line"
done || echo "  (Run 'ssh $NODE nvidia-smi' manually to see GPU status)"

#------------------------------------------------------------------------------
# 6. WEB INTERFACE CHECK
#------------------------------------------------------------------------------
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🌐 [6/6] WEB INTERFACE STATUS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

HTTP_DASH=$(curl -s -o /dev/null -w "%{http_code}" http://$NODE:8080/ 2>/dev/null || echo "000")
echo "  Dashboard (http://$NODE:8080): HTTP $HTTP_DASH"

HTTP_API=$(curl -s -o /dev/null -w "%{http_code}" http://$NODE:11434/api/tags 2>/dev/null || echo "000")
echo "  Ollama API (http://$NODE:11434): HTTP $HTTP_API"

echo ""
echo "  📌 To access from your MacBook, run this SSH tunnel:"
echo "     ssh -p 8822 -L 8080:$NODE:8080 -L 11434:$NODE:11434 u103032@login.lxp.lu"
echo ""
echo "  Then open: http://localhost:8080"

#------------------------------------------------------------------------------
# SUMMARY
#------------------------------------------------------------------------------
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                    TEST SUITE COMPLETE! ✅                       ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "📁 Results saved to: $RESULTS_DIR"
echo ""
ls -la $RESULTS_DIR/
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
