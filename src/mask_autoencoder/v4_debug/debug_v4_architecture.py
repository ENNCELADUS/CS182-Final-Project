#!/usr/bin/env python3
"""
Systematic debugging of v4.py architecture components to identify learning issues.
This script tests each component individually to find the root cause.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader

# Import from existing modules
from v4 import (load_data, ProteinPairDataset, collate_fn, 
                RoPEPositionalEncoding, EnhancedTransformerLayer, 
                ProteinEncoder, CrossAttentionInteraction, 
                EnhancedMLPDecoder, ProteinInteractionClassifier)
from quick_fix_v4 import SimplifiedProteinClassifier

def test_gradient_magnitudes(model, data_loader, device, model_name):
    """Test gradient magnitudes across the model"""
    print(f"\n🔬 GRADIENT ANALYSIS: {model_name}")
    print("=" * 50)
    
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    
    # Get one batch
    batch = next(iter(data_loader))
    emb_a, emb_b, lengths_a, lengths_b, interactions = batch
    emb_a = emb_a.to(device).float()
    emb_b = emb_b.to(device).float()
    lengths_a = lengths_a.to(device)
    lengths_b = lengths_b.to(device)
    interactions = interactions.to(device).float()
    
    # Forward pass
    model.zero_grad()
    logits = model(emb_a, emb_b, lengths_a, lengths_b)
    if logits.dim() > 1:
        logits = logits.squeeze(-1)
    
    loss = criterion(logits, interactions)
    loss.backward()
    
    # Analyze gradients by layer
    gradient_stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            grad_mean = param.grad.data.mean().item()
            grad_std = param.grad.data.std().item()
            param_norm = param.data.norm(2).item()
            
            gradient_stats[name] = {
                'grad_norm': grad_norm,
                'grad_mean': grad_mean,
                'grad_std': grad_std,
                'param_norm': param_norm,
                'grad_to_param_ratio': grad_norm / (param_norm + 1e-8)
            }
    
    # Print analysis
    print(f"Loss: {loss.item():.6f}")
    print(f"Output logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
    print(f"Output logits mean: {logits.mean().item():.6f}")
    
    # Group gradients by component
    components = {}
    for name, stats in gradient_stats.items():
        component = name.split('.')[0]  # Get first part of name
        if component not in components:
            components[component] = []
        components[component].append(stats)
    
    print(f"\nGradient analysis by component:")
    for component, stats_list in components.items():
        avg_grad_norm = np.mean([s['grad_norm'] for s in stats_list])
        avg_ratio = np.mean([s['grad_to_param_ratio'] for s in stats_list])
        print(f"  {component:20s}: avg_grad_norm={avg_grad_norm:.2e}, grad/param_ratio={avg_ratio:.2e}")
    
    # Identify potential issues
    issues = []
    total_grad_norm = sum([s['grad_norm'] for s in gradient_stats.values()])
    
    if total_grad_norm < 1e-6:
        issues.append("Vanishing gradients")
    elif total_grad_norm > 100:
        issues.append("Exploding gradients")
    
    zero_grads = sum([1 for s in gradient_stats.values() if s['grad_norm'] < 1e-10])
    if zero_grads > len(gradient_stats) * 0.3:
        issues.append(f"Many zero gradients ({zero_grads}/{len(gradient_stats)})")
    
    return {
        'total_grad_norm': total_grad_norm,
        'loss': loss.item(),
        'issues': issues,
        'component_stats': components,
        'logits_stats': {
            'mean': logits.mean().item(),
            'std': logits.std().item(),
            'min': logits.min().item(),
            'max': logits.max().item()
        }
    }

def test_component_individually():
    """Test individual components with synthetic data"""
    print(f"\n🧪 COMPONENT TESTING")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 4
    seq_len = 100
    embed_dim = 960
    
    # Create synthetic data
    emb_a = torch.randn(batch_size, seq_len, embed_dim).to(device)
    emb_b = torch.randn(batch_size, seq_len, embed_dim).to(device)
    lengths_a = torch.tensor([seq_len] * batch_size).to(device)
    lengths_b = torch.tensor([seq_len] * batch_size).to(device)
    
    print(f"Testing with synthetic data: batch_size={batch_size}, seq_len={seq_len}")
    
    # Test 1: RoPE Positional Encoding
    print(f"\n1. Testing RoPE Positional Encoding...")
    try:
        rope = RoPEPositionalEncoding(dim=256).to(device)
        test_input = torch.randn(batch_size, seq_len, 256).to(device)
        rope_output = rope(test_input)
        
        print(f"   Input range: [{test_input.min():.3f}, {test_input.max():.3f}]")
        print(f"   Output range: [{rope_output.min():.3f}, {rope_output.max():.3f}]")
        print(f"   Output mean: {rope_output.mean():.6f}")
        print(f"   NaN/Inf check: NaN={torch.isnan(rope_output).any()}, Inf={torch.isinf(rope_output).any()}")
        
        # Check if RoPE is changing the input significantly
        diff = (rope_output - test_input).abs().mean()
        print(f"   Difference from input: {diff:.6f}")
        
    except Exception as e:
        print(f"   ❌ RoPE failed: {e}")
    
    # Test 2: Enhanced Transformer Layer
    print(f"\n2. Testing Enhanced Transformer Layer...")
    try:
        transformer = EnhancedTransformerLayer(d_model=256, nhead=8, dim_feedforward=1024).to(device)
        test_input = torch.randn(batch_size, seq_len, 256).to(device)
        
        # Create padding mask
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool).to(device)
        
        transformer_output = transformer(test_input, src_key_padding_mask=mask)
        
        print(f"   Input range: [{test_input.min():.3f}, {test_input.max():.3f}]")
        print(f"   Output range: [{transformer_output.min():.3f}, {transformer_output.max():.3f}]")
        print(f"   Output mean: {transformer_output.mean():.6f}")
        print(f"   NaN/Inf check: NaN={torch.isnan(transformer_output).any()}, Inf={torch.isinf(transformer_output).any()}")
        
        # Test gradient flow
        loss = transformer_output.sum()
        loss.backward()
        
        total_grad_norm = sum([p.grad.norm().item() for p in transformer.parameters() if p.grad is not None])
        print(f"   Gradient norm: {total_grad_norm:.6f}")
        
    except Exception as e:
        print(f"   ❌ Transformer failed: {e}")
    
    # Test 3: Cross-Attention Interaction
    print(f"\n3. Testing Cross-Attention Interaction...")
    try:
        cross_attn = CrossAttentionInteraction(embed_dim=960).to(device)
        emb_a_flat = torch.randn(batch_size, 960).to(device)
        emb_b_flat = torch.randn(batch_size, 960).to(device)
        
        interaction_output = cross_attn(emb_a_flat, emb_b_flat)
        
        print(f"   Input A range: [{emb_a_flat.min():.3f}, {emb_a_flat.max():.3f}]")
        print(f"   Input B range: [{emb_b_flat.min():.3f}, {emb_b_flat.max():.3f}]")
        print(f"   Output range: [{interaction_output.min():.3f}, {interaction_output.max():.3f}]")
        print(f"   Output mean: {interaction_output.mean():.6f}")
        print(f"   NaN/Inf check: NaN={torch.isnan(interaction_output).any()}, Inf={torch.isinf(interaction_output).any()}")
        
        # Test gradient flow
        loss = interaction_output.sum()
        loss.backward()
        
        total_grad_norm = sum([p.grad.norm().item() for p in cross_attn.parameters() if p.grad is not None])
        print(f"   Gradient norm: {total_grad_norm:.6f}")
        
    except Exception as e:
        print(f"   ❌ Cross-attention failed: {e}")
    
    # Test 4: Enhanced MLP Decoder
    print(f"\n4. Testing Enhanced MLP Decoder...")
    try:
        decoder = EnhancedMLPDecoder(input_dim=960, hidden_dims=[256, 128]).to(device)
        test_input = torch.randn(batch_size, 960).to(device)
        
        decoder_output = decoder(test_input)
        
        print(f"   Input range: [{test_input.min():.3f}, {test_input.max():.3f}]")
        print(f"   Output range: [{decoder_output.min():.3f}, {decoder_output.max():.3f}]")
        print(f"   Output mean: {decoder_output.mean():.6f}")
        print(f"   NaN/Inf check: NaN={torch.isnan(decoder_output).any()}, Inf={torch.isinf(decoder_output).any()}")
        
        # Test gradient flow
        loss = decoder_output.sum()
        loss.backward()
        
        total_grad_norm = sum([p.grad.norm().item() for p in decoder.parameters() if p.grad is not None])
        print(f"   Gradient norm: {total_grad_norm:.6f}")
        
    except Exception as e:
        print(f"   ❌ Decoder failed: {e}")

def test_weight_initialization():
    """Test weight initialization patterns"""
    print(f"\n🔧 WEIGHT INITIALIZATION ANALYSIS")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create models
    v4_model = ProteinInteractionClassifier(
        encoder_layers=1,
        encoder_embed_dim=64,
        encoder_heads=2,
        decoder_hidden_dims=[64, 32]
    ).to(device)
    
    simple_model = SimplifiedProteinClassifier(hidden_dim=256).to(device)
    
    print(f"Analyzing weight initialization patterns...")
    
    def analyze_weights(model, model_name):
        print(f"\n{model_name}:")
        weight_stats = {}
        
        for name, param in model.named_parameters():
            if 'weight' in name:
                w = param.data
                stats = {
                    'mean': w.mean().item(),
                    'std': w.std().item(),
                    'min': w.min().item(),
                    'max': w.max().item(),
                    'shape': list(w.shape)
                }
                weight_stats[name] = stats
                
                print(f"  {name:30s}: mean={stats['mean']:7.4f}, std={stats['std']:7.4f}, "
                      f"range=[{stats['min']:7.4f}, {stats['max']:7.4f}]")
        
        return weight_stats
    
    v4_weights = analyze_weights(v4_model, "V4 Model")
    simple_weights = analyze_weights(simple_model, "Simple Model")
    
    # Check for potential initialization issues
    print(f"\n🔍 Initialization Issues Check:")
    for name, stats in v4_weights.items():
        issues = []
        if abs(stats['mean']) > 0.1:
            issues.append(f"High mean ({stats['mean']:.4f})")
        if stats['std'] > 1.0:
            issues.append(f"High std ({stats['std']:.4f})")
        if stats['std'] < 0.001:
            issues.append(f"Low std ({stats['std']:.4f})")
        
        if issues:
            print(f"  {name}: {', '.join(issues)}")

def test_learning_rate_sensitivity():
    """Test how different learning rates affect both models"""
    print(f"\n📊 LEARNING RATE SENSITIVITY TEST")
    print("=" * 50)
    
    # Load minimal data for testing
    train_data, val_data, _, _, protein_embeddings = load_data()
    train_dataset = ProteinPairDataset(train_data, protein_embeddings)
    
    # Use tiny subset for fast testing
    small_indices = list(range(min(100, len(train_dataset))))
    small_dataset = torch.utils.data.Subset(train_dataset, small_indices)
    
    data_loader = DataLoader(small_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, drop_last=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    learning_rates = [1e-5, 1e-4, 1e-3, 5e-3, 1e-2]
    
    results = {}
    
    for lr in learning_rates:
        print(f"\n📈 Testing LR = {lr}")
        
        # Test V4 model
        print(f"  V4 Model:")
        v4_model = ProteinInteractionClassifier(
            encoder_layers=1,
            encoder_embed_dim=64,
            encoder_heads=2,
            decoder_hidden_dims=[64, 32]
        ).to(device)
        
        v4_results = test_learning_with_lr(v4_model, data_loader, lr, device)
        
        # Test Simple model  
        print(f"  Simple Model:")
        simple_model = SimplifiedProteinClassifier(hidden_dim=128).to(device)
        simple_results = test_learning_with_lr(simple_model, data_loader, lr, device)
        
        results[lr] = {
            'v4': v4_results,
            'simple': simple_results
        }
        
        # Compare
        print(f"    LR {lr}: V4 loss_change={v4_results['loss_change']:.4f}, "
              f"Simple loss_change={simple_results['loss_change']:.4f}")
    
    return results

def test_learning_with_lr(model, data_loader, lr, device):
    """Test learning capability with specific LR"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    
    model.train()
    initial_loss = None
    final_loss = None
    losses = []
    
    # Train for 10 steps
    for step, (emb_a, emb_b, lengths_a, lengths_b, interactions) in enumerate(data_loader):
        if step >= 10:
            break
            
        try:
            emb_a = emb_a.to(device).float()
            emb_b = emb_b.to(device).float()
            lengths_a = lengths_a.to(device)
            lengths_b = lengths_b.to(device)
            interactions = interactions.to(device).float()
            
            optimizer.zero_grad()
            logits = model(emb_a, emb_b, lengths_a, lengths_b)
            if logits.dim() > 1:
                logits = logits.squeeze(-1)
            
            loss = criterion(logits, interactions)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"    NaN/Inf loss at step {step}")
                return {'loss_change': 0, 'stable': False, 'final_loss': float('inf')}
            
            if initial_loss is None:
                initial_loss = loss.item()
            final_loss = loss.item()
            losses.append(loss.item())
            
            loss.backward()
            
            # Check gradient norm
            total_grad_norm = sum([p.grad.norm().item() for p in model.parameters() if p.grad is not None])
            if total_grad_norm > 100:
                print(f"    Large gradients ({total_grad_norm:.2f}) at step {step}")
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        except Exception as e:
            print(f"    Error at step {step}: {e}")
            return {'loss_change': 0, 'stable': False, 'final_loss': float('inf')}
    
    loss_change = initial_loss - final_loss if initial_loss and final_loss else 0
    return {
        'loss_change': loss_change, 
        'stable': True, 
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'losses': losses
    }

def main():
    """Run comprehensive architecture debugging"""
    print("🔍 V4 ARCHITECTURE DEBUGGING")
    print("=" * 60)
    print("Systematic analysis to identify why v4.py fails to learn")
    print("=" * 60)
    
    # Load minimal data
    try:
        train_data, val_data, _, _, protein_embeddings = load_data()
        train_dataset = ProteinPairDataset(train_data, protein_embeddings)
        
        # Create small test dataset
        small_indices = list(range(min(50, len(train_dataset))))
        small_dataset = torch.utils.data.Subset(train_dataset, small_indices)
        data_loader = DataLoader(small_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, drop_last=True)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Create models for comparison
        v4_model = ProteinInteractionClassifier(
            encoder_layers=1,
            encoder_embed_dim=64,
            encoder_heads=2,
            decoder_hidden_dims=[64, 32]
        ).to(device)
        
        simple_model = SimplifiedProteinClassifier(hidden_dim=128).to(device)
        
        print(f"\nModel sizes:")
        print(f"  V4 model: {sum(p.numel() for p in v4_model.parameters()):,} parameters")
        print(f"  Simple model: {sum(p.numel() for p in simple_model.parameters()):,} parameters")
        
        # Gradient analysis with different LRs
        print(f"\n" + "="*60)
        print("GRADIENT ANALYSIS WITH ORIGINAL LR SETTINGS")
        print("="*60)
        
        # Test v4 with its typical LR (1e-3 from your experiment)
        v4_model_test = ProteinInteractionClassifier(
            encoder_layers=1,
            encoder_embed_dim=64,
            encoder_heads=2,
            decoder_hidden_dims=[64, 32]
        ).to(device)
        
        simple_model_test = SimplifiedProteinClassifier(hidden_dim=128).to(device)
        
        # Set optimizers to match your experiment
        v4_optimizer = torch.optim.AdamW(v4_model_test.parameters(), lr=1e-3, weight_decay=0.01)
        simple_optimizer = torch.optim.AdamW(simple_model_test.parameters(), lr=5e-3, weight_decay=0.01)
        
        v4_grads = test_gradient_magnitudes(v4_model_test, data_loader, device, "V4 Model (LR=1e-3)")
        simple_grads = test_gradient_magnitudes(simple_model_test, data_loader, device, "Simple Model (LR=5e-3)")
        
        # Learning rate sensitivity
        lr_results = test_learning_rate_sensitivity()
        
        # Summary
        print(f"\n" + "="*60)
        print("🎯 DIAGNOSTIC SUMMARY")
        print("="*60)
        
        print(f"V4 Model Issues:")
        for issue in v4_grads['issues']:
            print(f"  ❌ {issue}")
        
        print(f"\nGradient Comparison:")
        print(f"  V4 total gradient norm: {v4_grads['total_grad_norm']:.2e}")
        print(f"  Simple total gradient norm: {simple_grads['total_grad_norm']:.2e}")
        print(f"  Ratio (V4/Simple): {v4_grads['total_grad_norm']/simple_grads['total_grad_norm']:.2f}")
        
        print(f"\nOutput Comparison:")
        print(f"  V4 logits mean: {v4_grads['logits_stats']['mean']:.6f}")
        print(f"  Simple logits mean: {simple_grads['logits_stats']['mean']:.6f}")
        
        print(f"\nLearning Rate Analysis:")
        for lr, results in lr_results.items():
            v4_change = results['v4']['loss_change']
            simple_change = results['simple']['loss_change']
            print(f"  LR {lr}: V4={v4_change:.4f}, Simple={simple_change:.4f}, Ratio={v4_change/(simple_change+1e-8):.2f}")
        
        # Find optimal LR for V4
        best_v4_lr = max(lr_results.keys(), key=lambda lr: lr_results[lr]['v4']['loss_change'])
        best_simple_lr = max(lr_results.keys(), key=lambda lr: lr_results[lr]['simple']['loss_change'])
        
        print(f"\nOptimal Learning Rates:")
        print(f"  V4 model: {best_v4_lr} (loss change: {lr_results[best_v4_lr]['v4']['loss_change']:.4f})")
        print(f"  Simple model: {best_simple_lr} (loss change: {lr_results[best_simple_lr]['simple']['loss_change']:.4f})")
        
        # Hypothesis generation
        print(f"\n💡 LIKELY CAUSES:")
        
        grad_ratio = v4_grads['total_grad_norm'] / simple_grads['total_grad_norm']
        if grad_ratio < 0.1:
            print("  1. ❌ V4 has much smaller gradients - architecture absorbs gradients")
        
        if abs(v4_grads['logits_stats']['mean']) < 0.001:
            print("  2. ❌ V4 outputs near zero - poor initialization or saturated activations")
        
        if v4_grads['total_grad_norm'] < 1e-6:
            print("  3. ❌ V4 has vanishing gradients - too complex architecture")
        
        # Check if higher LR helps V4
        if lr_results[best_v4_lr]['v4']['loss_change'] > lr_results[1e-3]['v4']['loss_change'] * 2:
            print(f"  4. ❌ V4 needs much higher LR - current 1e-3 is too low, try {best_v4_lr}")
        
        print(f"\n🔧 RECOMMENDED FIXES:")
        print(f"  1. Use higher learning rate: {best_v4_lr} instead of 1e-3")
        print("  2. Simplify architecture - current transformer is too complex")
        print("  3. Check component initialization - some parts may be broken")
        print("  4. Add proper residual connections")
        
        # Generate fixed configuration
        print(f"\n⚙️ SUGGESTED V4 CONFIGURATION:")
        print("model = ProteinInteractionClassifier(")
        print("    encoder_layers=1,")
        print("    encoder_embed_dim=64,") 
        print("    encoder_heads=2,")
        print("    decoder_hidden_dims=[64, 32]")
        print(")")
        print(f"optimizer = AdamW(model.parameters(), lr={best_v4_lr}, weight_decay=0.01)")
        print("# Add OneCycle scheduler like SimplifiedProteinClassifier")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 