#!/usr/bin/env python3
"""
Focused test of individual v4.py components to identify the specific issue.
"""

import torch
import torch.nn as nn
import numpy as np
from v4 import (RoPEPositionalEncoding, EnhancedTransformerLayer, 
                ProteinEncoder, CrossAttentionInteraction, 
                EnhancedMLPDecoder, ProteinInteractionClassifier)
from v4.DNN_v4 import SimplifiedProteinClassifier

def test_rope_component():
    """Test RoPE positional encoding"""
    print("🔬 Testing RoPE Positional Encoding")
    print("-" * 40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test different dimensions
    dims = [64, 128, 256]
    
    for dim in dims:
        print(f"\nTesting dim={dim}")
        rope = RoPEPositionalEncoding(dim=dim).to(device)
        
        # Test input
        batch_size, seq_len = 4, 100
        x = torch.randn(batch_size, seq_len, dim).to(device)
        
        # Forward pass
        output = rope(x)
        
        # Check for issues
        print(f"  Input mean: {x.mean():.6f}, std: {x.std():.6f}")
        print(f"  Output mean: {output.mean():.6f}, std: {output.std():.6f}")
        print(f"  Difference: {(output - x).abs().mean():.6f}")
        print(f"  NaN check: {torch.isnan(output).any()}")
        print(f"  Inf check: {torch.isinf(output).any()}")
        
        # Test gradient flow
        x.requires_grad_(True)
        output = rope(x)
        loss = output.sum()
        loss.backward()
        
        print(f"  Input gradient norm: {x.grad.norm().item():.2e}" if x.grad is not None else "No input gradients")

def test_transformer_layer():
    """Test Enhanced Transformer Layer"""
    print("\n🔬 Testing Enhanced Transformer Layer")
    print("-" * 40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test different configurations
    configs = [
        {'d_model': 64, 'nhead': 2, 'dim_feedforward': 256},
        {'d_model': 128, 'nhead': 4, 'dim_feedforward': 512},
        {'d_model': 256, 'nhead': 8, 'dim_feedforward': 1024}
    ]
    
    for config in configs:
        print(f"\nTesting config: {config}")
        
        try:
            layer = EnhancedTransformerLayer(**config).to(device)
            
            # Test input
            batch_size, seq_len = 4, 100
            x = torch.randn(batch_size, seq_len, config['d_model']).to(device)
            
            # Create padding mask (no padding for this test)
            mask = torch.zeros(batch_size, seq_len, dtype=torch.bool).to(device)
            
            # Forward pass
            output = layer(x, src_key_padding_mask=mask)
            
            print(f"  Input shape: {x.shape}")
            print(f"  Output shape: {output.shape}")
            print(f"  Input mean: {x.mean():.6f}, std: {x.std():.6f}")
            print(f"  Output mean: {output.mean():.6f}, std: {output.std():.6f}")
            print(f"  NaN check: {torch.isnan(output).any()}")
            print(f"  Inf check: {torch.isinf(output).any()}")
            
            # Test gradient flow
            x.requires_grad_(True)
            output = layer(x, src_key_padding_mask=mask)
            loss = output.sum()
            loss.backward()
            
            # Check gradient norms
            total_grad_norm = sum([p.grad.norm().item() for p in layer.parameters() if p.grad is not None])
            print(f"  Total gradient norm: {total_grad_norm:.2e}")
            
            if total_grad_norm < 1e-8:
                print(f"  ⚠️  Very small gradients - potential vanishing gradient problem")
            elif total_grad_norm > 100:
                print(f"  ⚠️  Large gradients - potential exploding gradient problem")
            else:
                print(f"  ✅ Gradient norm looks healthy")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")

def test_cross_attention():
    """Test Cross Attention Interaction"""
    print("\n🔬 Testing Cross Attention Interaction")
    print("-" * 40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    embed_dims = [64, 128, 256, 960]
    
    for embed_dim in embed_dims:
        print(f"\nTesting embed_dim={embed_dim}")
        
        try:
            cross_attn = CrossAttentionInteraction(embed_dim=embed_dim).to(device)
            
            # Test input
            batch_size = 4
            emb_a = torch.randn(batch_size, embed_dim).to(device)
            emb_b = torch.randn(batch_size, embed_dim).to(device)
            
            # Forward pass
            output = cross_attn(emb_a, emb_b)
            
            print(f"  Input A mean: {emb_a.mean():.6f}, std: {emb_a.std():.6f}")
            print(f"  Input B mean: {emb_b.mean():.6f}, std: {emb_b.std():.6f}")
            print(f"  Output mean: {output.mean():.6f}, std: {output.std():.6f}")
            print(f"  Output shape: {output.shape}")
            print(f"  NaN check: {torch.isnan(output).any()}")
            print(f"  Inf check: {torch.isinf(output).any()}")
            
            # Test gradient flow
            emb_a.requires_grad_(True)
            emb_b.requires_grad_(True)
            output = cross_attn(emb_a, emb_b)
            loss = output.sum()
            loss.backward()
            
            # Check gradient norms
            total_grad_norm = sum([p.grad.norm().item() for p in cross_attn.parameters() if p.grad is not None])
            print(f"  Total gradient norm: {total_grad_norm:.2e}")
            
            if total_grad_norm < 1e-8:
                print(f"  ⚠️  Very small gradients")
            elif total_grad_norm > 100:
                print(f"  ⚠️  Large gradients")
            else:
                print(f"  ✅ Gradient norm looks healthy")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")

def test_mlp_decoder():
    """Test Enhanced MLP Decoder"""
    print("\n🔬 Testing Enhanced MLP Decoder")
    print("-" * 40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    configs = [
        {'input_dim': 64, 'hidden_dims': [32, 16]},
        {'input_dim': 128, 'hidden_dims': [64, 32]},
        {'input_dim': 256, 'hidden_dims': [128, 64]},
        {'input_dim': 960, 'hidden_dims': [256, 128]}
    ]
    
    for config in configs:
        print(f"\nTesting config: {config}")
        
        try:
            decoder = EnhancedMLPDecoder(**config).to(device)
            
            # Test input
            batch_size = 4
            x = torch.randn(batch_size, config['input_dim']).to(device)
            
            # Forward pass
            output = decoder(x)
            
            print(f"  Input mean: {x.mean():.6f}, std: {x.std():.6f}")
            print(f"  Output mean: {output.mean():.6f}, std: {output.std():.6f}")
            print(f"  Output shape: {output.shape}")
            print(f"  NaN check: {torch.isnan(output).any()}")
            print(f"  Inf check: {torch.isinf(output).any()}")
            
            # Test gradient flow
            x.requires_grad_(True)
            output = decoder(x)
            loss = output.sum()
            loss.backward()
            
            # Check gradient norms
            total_grad_norm = sum([p.grad.norm().item() for p in decoder.parameters() if p.grad is not None])
            print(f"  Total gradient norm: {total_grad_norm:.2e}")
            
            if total_grad_norm < 1e-8:
                print(f"  ⚠️  Very small gradients")
            elif total_grad_norm > 100:
                print(f"  ⚠️  Large gradients")
            else:
                print(f"  ✅ Gradient norm looks healthy")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")

def test_protein_encoder():
    """Test Protein Encoder"""
    print("\n🔬 Testing Protein Encoder")
    print("-" * 40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    configs = [
        {'embed_dim': 64, 'num_layers': 1, 'nhead': 2, 'ff_dim': 256},
        {'embed_dim': 128, 'num_layers': 2, 'nhead': 4, 'ff_dim': 512},
    ]
    
    for config in configs:
        print(f"\nTesting config: {config}")
        
        try:
            encoder = ProteinEncoder(
                input_dim=960,
                embed_dim=config['embed_dim'],
                num_layers=config['num_layers'],
                nhead=config['nhead'],
                ff_dim=config['ff_dim']
            ).to(device)
            
            # Test input
            batch_size, seq_len = 4, 100
            x = torch.randn(batch_size, seq_len, 960).to(device)
            lengths = torch.tensor([seq_len] * batch_size).to(device)
            
            # Forward pass
            output = encoder(x, lengths)
            
            print(f"  Input shape: {x.shape}")
            print(f"  Output shape: {output.shape}")
            print(f"  Input mean: {x.mean():.6f}, std: {x.std():.6f}")
            print(f"  Output mean: {output.mean():.6f}, std: {output.std():.6f}")
            print(f"  NaN check: {torch.isnan(output).any()}")
            print(f"  Inf check: {torch.isinf(output).any()}")
            
            # Test gradient flow
            x.requires_grad_(True)
            output = encoder(x, lengths)
            loss = output.sum()
            loss.backward()
            
            # Check gradient norms
            total_grad_norm = sum([p.grad.norm().item() for p in encoder.parameters() if p.grad is not None])
            print(f"  Total gradient norm: {total_grad_norm:.2e}")
            
            if total_grad_norm < 1e-8:
                print(f"  ⚠️  Very small gradients - potential issue")
            elif total_grad_norm > 100:
                print(f"  ⚠️  Large gradients - potential issue")
            else:
                print(f"  ✅ Gradient norm looks healthy")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")

def compare_architectures():
    """Direct comparison of v4 vs simple architectures"""
    print("\n🔬 ARCHITECTURE COMPARISON")
    print("-" * 40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create comparable models
    v4_model = ProteinInteractionClassifier(
        encoder_layers=1,
        encoder_embed_dim=64,
        encoder_heads=2,
        decoder_hidden_dims=[64, 32]
    ).to(device)
    
    simple_model = SimplifiedProteinClassifier(hidden_dim=128).to(device)
    
    print(f"V4 model parameters: {sum(p.numel() for p in v4_model.parameters()):,}")
    print(f"Simple model parameters: {sum(p.numel() for p in simple_model.parameters()):,}")
    
    # Create synthetic test data
    batch_size = 4
    seq_len_a, seq_len_b = 100, 80
    emb_a = torch.randn(batch_size, seq_len_a, 960).to(device)
    emb_b = torch.randn(batch_size, seq_len_b, 960).to(device)
    lengths_a = torch.tensor([seq_len_a] * batch_size).to(device)
    lengths_b = torch.tensor([seq_len_b] * batch_size).to(device)
    
    print(f"\nTesting with synthetic data:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence lengths: {seq_len_a}, {seq_len_b}")
    
    # Test V4 model
    print(f"\nV4 Model:")
    try:
        v4_output = v4_model(emb_a, emb_b, lengths_a, lengths_b)
        print(f"  Output shape: {v4_output.shape}")
        print(f"  Output mean: {v4_output.mean():.6f}")
        print(f"  Output std: {v4_output.std():.6f}")
        print(f"  Output range: [{v4_output.min():.3f}, {v4_output.max():.3f}]")
        print(f"  NaN check: {torch.isnan(v4_output).any()}")
        print(f"  Inf check: {torch.isinf(v4_output).any()}")
        
        # Test gradient
        loss = v4_output.sum()
        loss.backward()
        v4_grad_norm = sum([p.grad.norm().item() for p in v4_model.parameters() if p.grad is not None])
        print(f"  Gradient norm: {v4_grad_norm:.2e}")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        v4_grad_norm = 0
    
    # Test Simple model
    print(f"\nSimple Model:")
    try:
        simple_output = simple_model(emb_a, emb_b, lengths_a, lengths_b)
        print(f"  Output shape: {simple_output.shape}")
        print(f"  Output mean: {simple_output.mean():.6f}")
        print(f"  Output std: {simple_output.std():.6f}")
        print(f"  Output range: [{simple_output.min():.3f}, {simple_output.max():.3f}]")
        print(f"  NaN check: {torch.isnan(simple_output).any()}")
        print(f"  Inf check: {torch.isinf(simple_output).any()}")
        
        # Test gradient
        loss = simple_output.sum()
        loss.backward()
        simple_grad_norm = sum([p.grad.norm().item() for p in simple_model.parameters() if p.grad is not None])
        print(f"  Gradient norm: {simple_grad_norm:.2e}")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        simple_grad_norm = 0
    
    # Compare
    print(f"\nComparison:")
    if v4_grad_norm > 0 and simple_grad_norm > 0:
        ratio = v4_grad_norm / simple_grad_norm
        print(f"  Gradient ratio (V4/Simple): {ratio:.2f}")
        if ratio < 0.1:
            print(f"  ⚠️  V4 has much smaller gradients - learning will be slower")
        elif ratio > 10:
            print(f"  ⚠️  V4 has much larger gradients - may be unstable")
        else:
            print(f"  ✅ Gradient magnitudes are comparable")
    
    return v4_grad_norm, simple_grad_norm

def main():
    """Run all component tests"""
    print("🔍 V4 COMPONENT-BY-COMPONENT TESTING")
    print("=" * 60)
    
    # Test individual components
    test_rope_component()
    test_transformer_layer()
    test_cross_attention()
    test_mlp_decoder()
    test_protein_encoder()
    
    # Compare full architectures
    v4_grad, simple_grad = compare_architectures()
    
    # Summary
    print(f"\n🎯 COMPONENT TEST SUMMARY")
    print("=" * 40)
    print(f"If any component showed ⚠️ warnings, that's likely the issue.")
    print(f"V4 gradient norm: {v4_grad:.2e}")
    print(f"Simple gradient norm: {simple_grad:.2e}")
    
    if v4_grad > 0 and simple_grad > 0:
        if v4_grad / simple_grad < 0.1:
            print(f"🔍 PRIMARY ISSUE: V4 has {simple_grad/v4_grad:.1f}x smaller gradients")
            print(f"   → This explains why it learns slower")
            print(f"   → Need higher learning rate or architecture fixes")

if __name__ == "__main__":
    main() 