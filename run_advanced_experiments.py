#!/usr/bin/env python3
"""
Advanced TEIA Architecture Experiments
Testing different architectures and loss functions to improve embedding similarity
"""

import subprocess
import sys
from datetime import datetime

def run_experiment(config_name, args):
    """Run a single experiment with given configuration"""
    print(f"\n{'='*60}")
    print(f"🚀 Starting Experiment: {config_name}")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, "main.py",
        "--project_name", "TEIA_ADVANCED_ARCH",
        "--exp_name", config_name,
        "--model_dir", "microsoft/DialoGPT-medium",
        "--dataset", "personachat", 
        "--blackbox_encoder", "sbert",
        "--surrogate_encoder", "sbert",
        "--training_size", "1000",
        "--batch_size", "32",
        "--mapping_lambda", "0.1",
        "--pivot_lambda", "0.1",
        "--surrogate_epoch", "20",
        "--embedding_consistency_weight", "0.5"
    ]
    
    # Add specific configuration arguments
    cmd.extend(args)
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {config_name} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {config_name} failed!")
        print(f"Error: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return False

def main():
    """Run systematic experiments with different architectures and loss functions"""
    
    experiments = [
        # Test 1: Baseline (simple architectures, standard loss)
        {
            "name": "Test1_Baseline_Simple",
            "args": [
                "--projection_architecture", "simple",
                "--mapping_architecture", "simple", 
                "--loss_type", "standard"
            ]
        },
        
        # Test 2: Deep architectures with standard loss
        {
            "name": "Test2_Deep_Architectures",
            "args": [
                "--projection_architecture", "deep",
                "--mapping_architecture", "deep",
                "--loss_type", "standard"
            ]
        },
        
        # Test 3: Residual architectures with standard loss
        {
            "name": "Test3_Residual_Architectures", 
            "args": [
                "--projection_architecture", "residual",
                "--mapping_architecture", "residual",
                "--loss_type", "standard"
            ]
        },
        
        # Test 4: Transformer projection with deep mapping
        {
            "name": "Test4_Transformer_Projection",
            "args": [
                "--projection_architecture", "transformer",
                "--mapping_architecture", "deep",
                "--loss_type", "standard"
            ]
        },
        
        # Test 5: Simple architectures with contrastive loss
        {
            "name": "Test5_Contrastive_Loss",
            "args": [
                "--projection_architecture", "simple",
                "--mapping_architecture", "simple",
                "--loss_type", "contrastive"
            ]
        },
        
        # Test 6: Deep architectures with triplet loss
        {
            "name": "Test6_Triplet_Loss",
            "args": [
                "--projection_architecture", "deep",
                "--mapping_architecture", "deep", 
                "--loss_type", "triplet"
            ]
        },
        
        # Test 7: Residual architectures with InfoNCE loss
        {
            "name": "Test7_InfoNCE_Loss",
            "args": [
                "--projection_architecture", "residual",
                "--mapping_architecture", "residual",
                "--loss_type", "infonce"
            ]
        },
        
        # Test 8: Best architecture combination (transformer + residual + contrastive)
        {
            "name": "Test8_Best_Combination",
            "args": [
                "--projection_architecture", "transformer",
                "--mapping_architecture", "residual",
                "--loss_type", "contrastive"
            ]
        }
    ]
    
    print("🔬 TEIA Advanced Architecture Experiments")
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🧪 Total experiments: {len(experiments)}")
    print("\n📋 Experiment Plan:")
    for i, exp in enumerate(experiments, 1):
        print(f"  {i}. {exp['name']}")
        print(f"     Args: {' '.join(exp['args'])}")
    
    # Run experiments
    results = {}
    successful = 0
    failed = 0
    
    for i, experiment in enumerate(experiments, 1):
        print(f"\n🔄 Progress: {i}/{len(experiments)}")
        
        success = run_experiment(experiment["name"], experiment["args"])
        results[experiment["name"]] = success
        
        if success:
            successful += 1
        else:
            failed += 1
            
        print(f"📊 Current status: {successful} successful, {failed} failed")
    
    # Final summary
    print(f"\n{'='*60}")
    print("📈 EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successful: {successful}/{len(experiments)}")
    print(f"❌ Failed: {failed}/{len(experiments)}")
    print(f"🕐 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n📋 Detailed Results:")
    for exp_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {exp_name}: {status}")
    
    print(f"\n💡 Check Weights & Biases (project: TEIA_ADVANCED_ARCH) for detailed results!")
    print("🎯 Focus on experiments with highest embedding similarity scores.")

if __name__ == "__main__":
    main()