#!/usr/bin/env python3
"""
Monitor training progress of the comprehensive v4.1 comparison
"""

import os
import time
import json
import subprocess
from datetime import datetime

def check_training_status():
    print(f"🕐 Training Status Check - {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)
    
    # Check if process is running
    try:
        result = subprocess.run(['pgrep', '-f', 'comprehensive_v4_1_comparison'], 
                              capture_output=True, text=True)
        if result.stdout.strip():
            print("✅ Training script is still running")
            pids = result.stdout.strip().split('\n')
            print(f"   Process IDs: {', '.join(pids)}")
        else:
            print("❌ Training script is not running")
    except:
        print("⚠️  Could not check process status")
    
    # Check directories and files
    results_dir = "results/v4_1_comparison"
    models_dir = "models/v4_1_comparison"
    
    print(f"\n📂 Directory Status:")
    
    # Results directory
    if os.path.exists(results_dir):
        files = os.listdir(results_dir)
        print(f"   Results ({len(files)} files): {results_dir}")
        for f in files:
            size_kb = os.path.getsize(os.path.join(results_dir, f)) // 1024
            print(f"      📄 {f} ({size_kb} KB)")
    else:
        print(f"   ❌ Results directory not found: {results_dir}")
    
    # Models directory  
    if os.path.exists(models_dir):
        files = os.listdir(models_dir)
        print(f"   Models ({len(files)} files): {models_dir}")
        for f in files:
            size_mb = os.path.getsize(os.path.join(models_dir, f)) // (1024*1024)
            print(f"      🎯 {f} ({size_mb} MB)")
    else:
        print(f"   ❌ Models directory not found: {models_dir}")
    
    # Check for specific result files
    summary_file = os.path.join(results_dir, "model_comparison_summary.csv")
    if os.path.exists(summary_file):
        print(f"\n📊 Training Complete! Summary available:")
        with open(summary_file, 'r') as f:
            print(f.read())
    
    return os.path.exists(summary_file)

def main():
    print("🔍 V4.1 TRAINING MONITOR")
    print("Press Ctrl+C to stop monitoring")
    
    completed = False
    while not completed:
        try:
            completed = check_training_status()
            if not completed:
                print(f"\n⏳ Waiting 30 seconds for next check...")
                time.sleep(30)
            else:
                print(f"\n🎉 Training completed!")
                break
                
        except KeyboardInterrupt:
            print(f"\n👋 Monitoring stopped by user")
            break
        except Exception as e:
            print(f"\n❌ Error during monitoring: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main() 