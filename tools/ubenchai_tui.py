#!/usr/bin/env python3
"""
UBenchAI Framework - Interactive Terminal User Interface (TUI)
EUMaster4HPC Student Challenge 2025 - Team 9

Features:
- Interactive menus with colors
- Real-time progress bars
- Live model testing
- Chat playground
- Benchmark visualization

Usage:
    python3 ubenchai_tui.py
"""

import os
import sys
import json
import time
import urllib.request
import subprocess
from datetime import datetime

# ANSI Color codes
class Colors:
    HEADER = '\033[35m'
    BLUE = '\033[34m'
    CYAN = '\033[36m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    RED = '\033[31m'
    WHITE = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[3m'
    END = '\033[0m'

def clear_screen():
    os.system('clear' if os.name != 'nt' else 'cls')

def print_header():
    clear_screen()
    print(f"""
{Colors.CYAN}{Colors.BOLD}
╔═════════════════════════════════════════════════════════════════════════════════════╗
║           ██╗   ██╗██████╗ ███████╗███╗   ██╗ ██████╗██╗  ██╗ █████╗ ██╗            ║
║           ██║   ██║██╔══██╗██╔════╝████╗  ██║██╔════╝██║  ██║██╔══██╗██║            ║
║           ██║   ██║██████╔╝█████╗  ██╔██╗ ██║██║     ███████║███████║██║            ║
║           ██║   ██║██╔══██╗██╔══╝  ██║╚██╗██║██║     ██╔══██║██╔══██║██║            ║
║           ╚██████╔╝██████╔╝███████╗██║ ╚████║╚██████╗██║  ██║██║  ██║██║            ║
║            ╚═════╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝            ║
║                          {Colors.WHITE}AI Benchmarking Framework for HPC{Colors.CYAN}                          ║
║                        {Colors.DIM}EUMaster4HPC Challenge 2025 - Team 9{Colors.CYAN}{Colors.BOLD}                         ║
╚═════════════════════════════════════════════════════════════════════════════════════╝
""")

def progress_bar(current, total, width=50, label="Progress"):
    percent = current / total if total > 0 else 0
    filled = int(width * percent)
    bar = '█' * filled + '░' * (width - filled)
    color = Colors.GREEN if percent >= 1 else Colors.YELLOW
    print(f"\r{color}{label}: [{bar}] {percent*100:.1f}%{Colors.END}", end='', flush=True)

class UBenchAITUI:
    def __init__(self):
        self.node = None
        self.models = []
        self.results = []
        
    def get_node(self):
        print(f"\n{Colors.YELLOW}Enter compute node name (e.g., mel2044): {Colors.END}", end='')
        self.node = input().strip()
        
        if not self.node:
            print(f"{Colors.RED}❌ Node name cannot be empty!{Colors.END}")
            return False
        
        print(f"{Colors.CYAN}Connecting to {self.node}...{Colors.END}")
        
        try:
            url = f"http://{self.node}:11434/api/tags"
            with urllib.request.urlopen(url, timeout=5) as resp:
                data = json.loads(resp.read())
                self.models = [m['name'] for m in data.get('models', [])]
            
            print(f"{Colors.GREEN}✓ Connected! ({len(self.models)} models available){Colors.END}")
            return True
        except Exception as e:
            print(f"{Colors.RED}❌ Cannot connect: {e}{Colors.END}")
            return False
    
    def show_main_menu(self):
        menu = [
            ("1", "📊 View Models", self.show_models),
            ("2", "🏃 Quick Benchmark", self.run_benchmark),
            ("3", "💬 Chat Playground", self.chat_playground),
            ("4", "⚡ Load Test", self.load_test),
            ("5", "📈 Model Comparison", self.model_comparison),
            ("6", "🏆 Leaderboard", self.show_leaderboard),
            ("7", "📄 Generate Report", self.generate_report),
            ("q", "🚪 Quit", None),
        ]
        
        print(f"\n{Colors.BOLD}  MAIN MENU{Colors.END}  {Colors.DIM}[Node: {self.node}]{Colors.END}\n")
        
        for key, label, _ in menu:
            print(f"  {Colors.CYAN}[{key}]{Colors.END} {label}")
        
        print(f"\n{Colors.YELLOW}Select: {Colors.END}", end='')
        choice = input().strip().lower()
        
        for key, _, action in menu:
            if choice == key:
                if action:
                    action()
                return choice != 'q'
        
        print(f"{Colors.RED}Invalid option!{Colors.END}")
        time.sleep(1)
        return True
    
    def show_models(self):
        print_header()
        print(f"\n{Colors.BOLD}📊 AVAILABLE MODELS{Colors.END}\n")
        
        try:
            url = f"http://{self.node}:11434/api/tags"
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read())
            
            models = data.get('models', [])
            print(f"  {Colors.DIM}{'MODEL':<30} {'SIZE':>12} {'PARAMS':>12}{Colors.END}")
            print(f"  {'─' * 56}")
            
            for m in models:
                name = m['name'][:29]
                size = f"{m['size']/1e9:.2f} GB"
                params = m['details'].get('parameter_size', 'N/A')
                color = Colors.GREEN if m['size'] < 2e9 else Colors.YELLOW if m['size'] < 5e9 else Colors.RED
                print(f"  {color}{name:<30} {size:>12} {params:>12}{Colors.END}")
            
        except Exception as e:
            print(f"{Colors.RED}Error: {e}{Colors.END}")
        
        input(f"\n{Colors.DIM}Press Enter...{Colors.END}")
    
    def run_benchmark(self):
        print_header()
        print(f"\n{Colors.BOLD}🏃 QUICK BENCHMARK{Colors.END}\n")
        
        print(f"{Colors.YELLOW}Model (default: mistral): {Colors.END}", end='')
        model = input().strip() or 'mistral'
        
        print(f"{Colors.YELLOW}Prompt (Enter for default): {Colors.END}", end='')
        prompt = input().strip() or "Explain quantum computing in 3 sentences."
        
        print(f"\n{Colors.CYAN}Running 3 iterations...{Colors.END}\n")
        
        results = []
        for i in range(3):
            progress_bar(i, 3, label=f"Run {i+1}/3")
            
            try:
                url = f"http://{self.node}:11434/api/generate"
                data = json.dumps({"model": model, "prompt": prompt, "stream": False, "options": {"num_predict": 100}}).encode()
                req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
                
                start = time.time()
                with urllib.request.urlopen(req, timeout=120) as resp:
                    result = json.loads(resp.read())
                
                tokens = result.get('eval_count', 0)
                tps = tokens / (result.get('eval_duration', 1) / 1e9)
                results.append({'tokens': tokens, 'tps': tps, 'latency': time.time() - start})
            except Exception as e:
                print(f"\n{Colors.RED}Error: {e}{Colors.END}")
        
        progress_bar(3, 3, label="Complete")
        
        if results:
            avg_tps = sum(r['tps'] for r in results) / len(results)
            self.results.append({'model': model, 'tps': avg_tps, 'tokens': sum(r['tokens'] for r in results)/len(results), 'timestamp': datetime.now().isoformat()})
            
            print(f"\n\n{Colors.GREEN}{'─' * 50}")
            print(f"  Model: {model}")
            print(f"  Speed: {avg_tps:.1f} tokens/sec")
            print(f"  Runs:  {len(results)}/3 successful")
            print(f"{'─' * 50}{Colors.END}")
            
            # Visual bar
            bar = '█' * int(avg_tps / 10) + '░' * (50 - int(avg_tps / 10))
            print(f"\n  [{Colors.GREEN}{bar[:50]}{Colors.END}] {avg_tps:.0f} tok/s")
        
        input(f"\n{Colors.DIM}Press Enter...{Colors.END}")
    
    def chat_playground(self):
        print_header()
        print(f"\n{Colors.BOLD}💬 CHAT PLAYGROUND{Colors.END}")
        print(f"{Colors.DIM}Type 'exit' to return, 'switch' to change model{Colors.END}\n")
        
        print(f"{Colors.YELLOW}Model (default: tinyllama): {Colors.END}", end='')
        model = input().strip() or 'tinyllama'
        
        print(f"\n{Colors.GREEN}Chatting with {model}{Colors.END}\n")
        
        while True:
            print(f"{Colors.CYAN}You: {Colors.END}", end='')
            user_input = input().strip()
            
            if user_input.lower() == 'exit':
                break
            elif user_input.lower() == 'switch':
                print(f"{Colors.YELLOW}New model: {Colors.END}", end='')
                model = input().strip()
                continue
            elif not user_input:
                continue
            
            try:
                url = f"http://{self.node}:11434/api/generate"
                data = json.dumps({"model": model, "prompt": user_input, "stream": False}).encode()
                req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
                
                print(f"{Colors.DIM}Thinking...{Colors.END}", end='\r')
                
                with urllib.request.urlopen(req, timeout=60) as resp:
                    result = json.loads(resp.read())
                
                response = result.get('response', '').strip()
                tps = result.get('eval_count', 0) / (result.get('eval_duration', 1) / 1e9)
                
                print(f"{Colors.GREEN}{model}: {Colors.END}{response}")
                print(f"{Colors.DIM}[{tps:.1f} tok/s]{Colors.END}\n")
                
            except Exception as e:
                print(f"{Colors.RED}Error: {e}{Colors.END}\n")
    
    def load_test(self):
        print_header()
        print(f"\n{Colors.BOLD}⚡ LOAD TEST{Colors.END}\n")
        
        print(f"{Colors.YELLOW}Model (default: tinyllama): {Colors.END}", end='')
        model = input().strip() or 'tinyllama'
        
        print(f"\n{Colors.CYAN}Testing concurrent requests...{Colors.END}\n")
        
        results = []
        for n in [1, 5, 10, 20]:
            progress_bar(n, 20, label=f"{n} concurrent")
            
            start = time.time()
            procs = [subprocess.Popen(["curl", "-s", f"http://{self.node}:11434/api/generate", "-d", json.dumps({"model": model, "prompt": f"Hi {i}", "stream": False})], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for i in range(n)]
            for p in procs:
                p.wait()
            
            elapsed = time.time() - start
            rps = n / elapsed
            results.append((n, elapsed, rps))
            print(f" → {rps:.2f} req/s")
        
        print(f"\n{Colors.BOLD}Results:{Colors.END}")
        for n, t, rps in results:
            bar = '█' * int(rps * 5)
            print(f"  {n:>3} req: {t:.2f}s = {Colors.GREEN}{rps:.2f} req/s {bar}{Colors.END}")
        
        input(f"\n{Colors.DIM}Press Enter...{Colors.END}")
    
    def model_comparison(self):
        print_header()
        print(f"\n{Colors.BOLD}📈 MODEL COMPARISON{Colors.END}\n")
        
        models = ['tinyllama', 'phi', 'mistral', 'llama2', 'codellama']
        prompt = "Explain AI briefly."
        
        print(f"Testing: {', '.join(models)}\n")
        
        results = []
        for i, model in enumerate(models):
            progress_bar(i, len(models), label=f"Testing {model}")
            
            try:
                url = f"http://{self.node}:11434/api/generate"
                data = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
                req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
                
                with urllib.request.urlopen(req, timeout=60) as resp:
                    result = json.loads(resp.read())
                
                tps = result.get('eval_count', 0) / (result.get('eval_duration', 1) / 1e9)
                results.append((model, tps))
                print(f" → {tps:.1f} tok/s")
            except:
                print(f" → {Colors.RED}Failed{Colors.END}")
        
        results.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n{Colors.BOLD}🏆 RANKING{Colors.END}\n")
        medals = ['🥇', '🥈', '🥉']
        max_tps = results[0][1] if results else 1
        
        for i, (model, tps) in enumerate(results):
            medal = medals[i] if i < 3 else '  '
            bar = '█' * int((tps / max_tps) * 30)
            color = [Colors.GREEN, Colors.YELLOW, Colors.WHITE][min(i, 2)]
            print(f"  {medal} {color}{model:<15} {bar:<30} {tps:.1f} tok/s{Colors.END}")
        
        input(f"\n{Colors.DIM}Press Enter...{Colors.END}")
    
    def show_leaderboard(self):
        print_header()
        print(f"\n{Colors.BOLD}🏆 LEADERBOARD{Colors.END}\n")
        
        if not self.results:
            print(f"  {Colors.DIM}No benchmarks yet!{Colors.END}")
        else:
            sorted_results = sorted(self.results, key=lambda x: x['tps'], reverse=True)
            medals = ['🥇', '🥈', '🥉']
            
            print(f"  {'RANK':<6} {'MODEL':<20} {'SPEED':>15}")
            print(f"  {'─' * 45}")
            
            for i, r in enumerate(sorted_results[:10]):
                medal = medals[i] if i < 3 else f'{i+1}.'
                print(f"  {medal:<6} {r['model']:<20} {r['tps']:>12.1f} tok/s")
        
        input(f"\n{Colors.DIM}Press Enter...{Colors.END}")
    
    def generate_report(self):
        print_header()
        print(f"\n{Colors.BOLD}📄 GENERATE REPORT{Colors.END}\n")
        
        if not self.results:
            print(f"{Colors.YELLOW}No data yet!{Colors.END}")
            input(f"\n{Colors.DIM}Press Enter...{Colors.END}")
            return
        
        filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(filename, 'w') as f:
            f.write(f"# UBenchAI Report\n\n**Node:** {self.node}\n**Date:** {datetime.now()}\n\n")
            f.write("| Model | Speed (tok/s) |\n|-------|---------------|\n")
            for r in self.results:
                f.write(f"| {r['model']} | {r['tps']:.1f} |\n")
        
        print(f"{Colors.GREEN}✓ Saved: {filename}{Colors.END}")
        input(f"\n{Colors.DIM}Press Enter...{Colors.END}")
    
    def run(self):
        print_header()
        if not self.get_node():
            return
        
        time.sleep(1)
        while True:
            print_header()
            if not self.show_main_menu():
                break
        
        print(f"\n{Colors.CYAN}Goodbye! 👋{Colors.END}\n")

if __name__ == "__main__":
    try:
        UBenchAITUI().run()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Interrupted!{Colors.END}\n")
