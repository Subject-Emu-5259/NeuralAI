#!/usr/bin/env python3
"""
NeuralAI Colab Training Monitor
==============================
Run this in a background thread or separate cell to monitor training.
Prints GPU stats, loss trends, and ETA every N seconds.
Can also start a tiny HTTP server for remote monitoring.
"""
import os, time, json, subprocess, threading
from pathlib import Path
from datetime import datetime

class TrainingMonitor:
    def __init__(self, checkpoint_dir, log_interval=30, http_port=None):
        self.ckpt_dir = Path(checkpoint_dir)
        self.log_interval = log_interval
        self.http_port = http_port
        self.running = False
        self.stats = {"status": "starting", "step": 0, "loss": None, "eta_sec": 0}
        
    def get_gpu_stats(self):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                return {
                    "gpu_util": float(parts[0]),
                    "vram_used_mb": float(parts[1]),
                    "vram_total_mb": float(parts[2]),
                    "temp_c": float(parts[3]),
                }
        except Exception as e:
            pass
        return {"gpu_util": 0, "vram_used_mb": 0, "vram_total_mb": 0, "temp_c": 0}
    
    def get_training_state(self):
        """Infer training state from checkpoints and logs."""
        # Find latest checkpoint
        ckpts = sorted(self.ckpt_dir.glob("checkpoint-*"))
        latest_step = 0
        if ckpts:
            latest = ckpts[-1]
            try:
                latest_step = int(latest.name.split("-")[-1])
            except:
                pass
        
        # Try to read loss from trainer_state.json
        loss = None
        if ckpts:
            state_file = ckpts[-1] / "trainer_state.json"
            if state_file.exists():
                try:
                    state = json.loads(state_file.read_text())
                    log_history = state.get("log_history", [])
                    if log_history:
                        latest_log = log_history[-1]
                        loss = latest_log.get("loss", None)
                except:
                    pass
        
        return {"step": latest_step, "loss": loss}
    
    def print_status(self):
        gpu = self.get_gpu_stats()
        train = self.get_training_state()
        now = datetime.now().strftime("%H:%M:%S")
        
        vram_pct = (gpu["vram_used_mb"] / gpu["vram_total_mb"] * 100) if gpu["vram_total_mb"] > 0 else 0
        
        line = (
            f"[{now}] Step: {train['step']:>6} | "
            f"Loss: {train['loss']:.4f}" if train['loss'] else "Loss: N/A",
            f" | GPU: {gpu['gpu_util']:>3.0f}% | "
            f"VRAM: {gpu['vram_used_mb']/1024:.1f}/{gpu['vram_total_mb']/1024:.1f}GB ({vram_pct:.0f}%) | "
            f"Temp: {gpu['temp_c']:.0f}°C"
        )
        print(" ".join(line))
        
        self.stats = {
            "timestamp": now,
            "step": train["step"],
            "loss": train["loss"],
            "gpu_util": gpu["gpu_util"],
            "vram_used_gb": gpu["vram_used_mb"] / 1024,
            "vram_total_gb": gpu["vram_total_mb"] / 1024,
            "temp_c": gpu["temp_c"],
        }
    
    def monitor_loop(self):
        self.running = True
        print(f"[Monitor] Started. Checking every {self.log_interval}s")
        print(f"[Monitor] Checkpoint dir: {self.ckpt_dir}")
        print("-" * 80)
        while self.running:
            try:
                self.print_status()
            except Exception as e:
                print(f"[Monitor] Error: {e}")
            time.sleep(self.log_interval)
    
    def start(self):
        """Start monitoring in a background thread."""
        t = threading.Thread(target=self.monitor_loop, daemon=True)
        t.start()
        print(f"[Monitor] Background thread started (tid={t.ident})")
        return t
    
    def start_http(self):
        """Start a tiny HTTP server for remote monitoring."""
        if not self.http_port:
            return None
        from http.server import HTTPServer, BaseHTTPRequestHandler
        
        class Handler(BaseHTTPRequestHandler):
            def do_GET(inner_self):
                inner_self.send_response(200)
                inner_self.send_header("Content-Type", "application/json")
                inner_self.end_headers()
                inner_self.wfile.write(json.dumps(self.stats, indent=2).encode())
            
            def log_message(self, format, *args):
                pass  # Suppress HTTP logs
        
        server = HTTPServer(("0.0.0.0", self.http_port), Handler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        print(f"[Monitor] HTTP server on port {self.http_port}")
        print(f"[Monitor] Metrics: http://localhost:{self.http_port}/")
        return server
    
    def stop(self):
        self.running = False
        print("[Monitor] Stopped.")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="/content/neuralai/checkpoints")
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--http-port", type=int, default=None)
    args = parser.parse_args()
    
    mon = TrainingMonitor(args.checkpoint_dir, args.interval, args.http_port)
    mon.start()
    if args.http_port:
        mon.start_http()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        mon.stop()

if __name__ == "__main__":
    main()
