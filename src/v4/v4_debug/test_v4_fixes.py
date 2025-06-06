#!/usr/bin/env python3
"""
Test specific fixes for v4.py gradient explosion issues.
This tests our hypothesis about initialization and scaling problems.
"""

import torch
import torch.nn as nn
import numpy as np
from v4 import (ProteinInteractionClassifier, EnhancedTransformerLayer, 
                CrossAttentionInteraction, EnhancedMLPDecoder, ProteinEncoder)
from quick_fix_v4 import SimplifiedProteinClassifier

class FixedEnhancedTransformerLayer(nn.Module):
    """Fixed version with proper initialization"""
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        
        # Multi-head attention with RoPE  
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        from v4 import RoPEPositionalEncoding
        self.rope = RoPEPositionalEncoding(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
        # Layer normalization and residual connections
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # FIX: Add proper initialization
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, x, src_key_padding_mask=None):
        # Apply RoPE to input
        x_rope = self.rope(x)
        
        # Self-attention with residual connection
        attn_out, _ = self.self_attn(
            x_rope, x_rope, x_rope, 
            key_padding_mask=src_key_padding_mask
        )
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x

class FixedCrossAttentionInteraction(nn.Module):
    """Fixed version with proper initialization and scaling"""
    def __init__(self, embed_dim=960, num_heads=8, ff_dim=512):
        super().__init__()
        
        # Bidirectional cross-attention
        self.cross_attn_ab = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn_ba = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # Layer normalization
        self.norm_ab = nn.LayerNorm(embed_dim)
        self.norm_ba = nn.LayerNorm(embed_dim)
        
        # Feed-forward for interaction processing
        self.interaction_ffn = nn.Sequential(
            nn.Linear(embed_dim * 2, ff_dim),
            nn.LayerNorm(ff_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(ff_dim, embed_dim)
        )
        
        # Residual connection
        self.residual_proj = nn.Linear(embed_dim * 2, embed_dim)
        
        # FIX: Add proper initialization
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, emb_a, emb_b):
        # Add sequence dimension for attention
        emb_a_seq = emb_a.unsqueeze(1)  # (B, 1, 960)
        emb_b_seq = emb_b.unsqueeze(1)  # (B, 1, 960)
        
        # Cross-attention: A attends to B
        attended_ab, _ = self.cross_attn_ab(emb_a_seq, emb_b_seq, emb_b_seq)
        attended_ab = self.norm_ab(attended_ab.squeeze(1) + emb_a)  # (B, 960)
        
        # Cross-attention: B attends to A
        attended_ba, _ = self.cross_attn_ba(emb_b_seq, emb_a_seq, emb_a_seq)
        attended_ba = self.norm_ba(attended_ba.squeeze(1) + emb_b)  # (B, 960)
        
        # Combine attended features
        combined = torch.cat([attended_ab, attended_ba], dim=-1)  # (B, 1920)
        
        # Process interaction with residual connection
        interaction_feat = self.interaction_ffn(combined)  # (B, 960)
        residual = self.residual_proj(combined)  # (B, 960)
        
        return interaction_feat + residual

class FixedProteinEncoder(nn.Module):
    """Fixed version with proper initialization and smaller scale"""
    def __init__(self, input_dim=960, embed_dim=256, num_layers=6, nhead=8, ff_dim=1024, 
                 use_variable_length=True, max_fixed_length=512):
        super().__init__()
        
        # Embedding layer
        self.embed = nn.Linear(input_dim, embed_dim)
        self.use_variable_length = use_variable_length
        self.max_fixed_length = max_fixed_length
        
        # Enhanced Transformer encoder layers with RoPE
        self.layers = nn.ModuleList([
            FixedEnhancedTransformerLayer(embed_dim, nhead, ff_dim, dropout=0.1)
            for _ in range(num_layers)
        ])
        
        # Enhanced hierarchical pooling with both local and global attention
        self.global_attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.local_attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        
        # Compression head with residual connection
        self.compress_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, input_dim)
        )
        
        # FIX: Proper initialization for learnable queries
        self.global_query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.local_query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # FIX: Initialize all weights properly
        self.apply(self._init_weights)
        
        # FIX: Special initialization for queries
        nn.init.normal_(self.global_query, 0, 0.02)
        nn.init.normal_(self.local_query, 0, 0.02)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, x, lengths):
        B, L, _ = x.shape
        device = x.device
        
        # Create padding mask
        mask = torch.arange(L, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
        
        # Embed input
        x_emb = self.embed(x)
        
        # Apply enhanced transformer layers
        for layer in self.layers:
            x_emb = layer(x_emb, src_key_padding_mask=mask)
        
        # Enhanced hierarchical pooling
        if self.use_variable_length:
            # Global attention pooling
            global_query = self.global_query.expand(B, -1, -1)
            global_feat, _ = self.global_attn(
                global_query, x_emb, x_emb,
                key_padding_mask=mask
            )
            global_feat = global_feat.squeeze(1)
            
            # Local attention pooling
            local_query = self.local_query.expand(B, -1, -1)
            local_feat, _ = self.local_attn(
                local_query, x_emb, x_emb,
                key_padding_mask=mask
            )
            local_feat = local_feat.squeeze(1)
        else:
            # Simplified pooling for fixed-length
            if L > self.max_fixed_length:
                x_emb = x_emb[:, :self.max_fixed_length]
                mask = mask[:, :self.max_fixed_length]
            
            attn_weights = torch.softmax(
                torch.where(mask, torch.tensor(-1e9, device=device), torch.zeros_like(mask, dtype=torch.float)), 
                dim=1
            )
            global_feat = torch.sum(x_emb * attn_weights.unsqueeze(-1), dim=1)
            local_feat = global_feat
        
        # Combine features
        combined_feat = torch.cat([global_feat, local_feat], dim=-1)
        
        # Compress to original dimension
        refined_emb = self.compress_head(combined_feat)
        
        return refined_emb

class FixedProteinInteractionClassifier(nn.Module):
    """Fixed version of the full classifier"""
    def __init__(self, 
                 encoder_embed_dim=256,
                 encoder_layers=6,
                 encoder_heads=8,
                 use_variable_length=True,
                 decoder_hidden_dims=[256, 128],
                 dropout=0.2):
        super().__init__()
        
        # FIX: Use smaller, properly initialized components
        self.protein_encoder = FixedProteinEncoder(
            input_dim=960,
            embed_dim=encoder_embed_dim,
            num_layers=encoder_layers,
            nhead=encoder_heads,
            ff_dim=encoder_embed_dim * 2,  # FIX: Smaller FF (was *4)
            use_variable_length=use_variable_length
        )
        
        # FIX: Use fixed cross-attention
        self.interaction_layer = FixedCrossAttentionInteraction(
            embed_dim=960, 
            num_heads=8,
            ff_dim=512
        )
        
        # Use existing MLP decoder (it already has good initialization)
        self.decoder = EnhancedMLPDecoder(
            input_dim=960,
            hidden_dims=decoder_hidden_dims,
            dropout=dropout
        )
    
    def forward(self, emb_a, emb_b, lengths_a, lengths_b):
        # Encode both proteins
        refined_a = self.protein_encoder(emb_a, lengths_a)
        refined_b = self.protein_encoder(emb_b, lengths_b)
        
        # Cross-attention interaction
        interaction_emb = self.interaction_layer(refined_a, refined_b)
        
        # MLP decoder
        logits = self.decoder(interaction_emb)
        
        return logits

def test_component_fixes():
    """Test the fixed components"""
    print("🔧 TESTING FIXED V4 COMPONENTS")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test 1: Fixed Transformer Layer
    print("\n1. Testing Fixed Transformer Layer")
    config = {'d_model': 256, 'nhead': 8, 'dim_feedforward': 1024}
    
    # Original vs Fixed
    original_layer = EnhancedTransformerLayer(**config).to(device)
    fixed_layer = FixedEnhancedTransformerLayer(**config).to(device)
    
    batch_size, seq_len = 4, 100
    x = torch.randn(batch_size, seq_len, config['d_model']).to(device)
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool).to(device)
    
    # Test original
    x.requires_grad_(True)
    output_orig = original_layer(x, src_key_padding_mask=mask)
    loss_orig = output_orig.sum()
    loss_orig.backward()
    orig_grad_norm = sum([p.grad.norm().item() for p in original_layer.parameters() if p.grad is not None])
    
    # Test fixed
    x.requires_grad_(True)
    output_fixed = fixed_layer(x, src_key_padding_mask=mask)
    loss_fixed = output_fixed.sum()
    loss_fixed.backward()
    fixed_grad_norm = sum([p.grad.norm().item() for p in fixed_layer.parameters() if p.grad is not None])
    
    print(f"   Original gradient norm: {orig_grad_norm:.2e}")
    print(f"   Fixed gradient norm: {fixed_grad_norm:.2e}")
    print(f"   Improvement ratio: {orig_grad_norm/fixed_grad_norm:.2f}x")
    
    # Test 2: Fixed Cross-Attention
    print("\n2. Testing Fixed Cross-Attention")
    
    original_cross = CrossAttentionInteraction(embed_dim=960).to(device)
    fixed_cross = FixedCrossAttentionInteraction(embed_dim=960).to(device)
    
    emb_a = torch.randn(4, 960).to(device)
    emb_b = torch.randn(4, 960).to(device)
    
    # Test original
    emb_a.requires_grad_(True)
    emb_b.requires_grad_(True)
    output_orig = original_cross(emb_a, emb_b)
    loss_orig = output_orig.sum()
    loss_orig.backward()
    orig_grad_norm = sum([p.grad.norm().item() for p in original_cross.parameters() if p.grad is not None])
    
    # Test fixed
    emb_a.requires_grad_(True)
    emb_b.requires_grad_(True)
    output_fixed = fixed_cross(emb_a, emb_b)
    loss_fixed = output_fixed.sum()
    loss_fixed.backward()
    fixed_grad_norm = sum([p.grad.norm().item() for p in fixed_cross.parameters() if p.grad is not None])
    
    print(f"   Original gradient norm: {orig_grad_norm:.2e}")
    print(f"   Fixed gradient norm: {fixed_grad_norm:.2e}")
    print(f"   Improvement ratio: {orig_grad_norm/fixed_grad_norm:.2f}x")
    
    # Test 3: Full Model Comparison
    print("\n3. Testing Full Model")
    
    original_model = ProteinInteractionClassifier(
        encoder_layers=1,
        encoder_embed_dim=64,
        encoder_heads=2,
        decoder_hidden_dims=[64, 32]
    ).to(device)
    
    fixed_model = FixedProteinInteractionClassifier(
        encoder_layers=1,
        encoder_embed_dim=64,
        encoder_heads=2,
        decoder_hidden_dims=[64, 32]
    ).to(device)
    
    simple_model = SimplifiedProteinClassifier(hidden_dim=128).to(device)
    
    # Test data
    batch_size = 4
    emb_a = torch.randn(batch_size, 100, 960).to(device)
    emb_b = torch.randn(batch_size, 80, 960).to(device)
    lengths_a = torch.tensor([100] * batch_size).to(device)
    lengths_b = torch.tensor([80] * batch_size).to(device)
    
    models = {
        'Original V4': original_model,
        'Fixed V4': fixed_model,
        'Simple': simple_model
    }
    
    print(f"\nFull Model Comparison:")
    results = {}
    
    for name, model in models.items():
        try:
            output = model(emb_a, emb_b, lengths_a, lengths_b)
            loss = output.sum()
            loss.backward()
            
            grad_norm = sum([p.grad.norm().item() for p in model.parameters() if p.grad is not None])
            param_count = sum(p.numel() for p in model.parameters())
            
            results[name] = {
                'grad_norm': grad_norm,
                'param_count': param_count,
                'output_mean': output.mean().item(),
                'output_std': output.std().item()
            }
            
            print(f"   {name:12s}: Grad={grad_norm:.2e}, Params={param_count:,}, "
                  f"Output mean={output.mean().item():.3f}")
            
        except Exception as e:
            print(f"   {name:12s}: ERROR - {e}")
            results[name] = {'error': str(e)}
    
    return results

def main():
    """Run component fix tests"""
    print("🔧 V4 COMPONENT FIX VALIDATION")
    print("=" * 60)
    print("Testing our hypothesis that initialization fixes gradient explosion")
    print("=" * 60)
    
    results = test_component_fixes()
    
    print(f"\n🎯 FIX VALIDATION SUMMARY")
    print("=" * 40)
    
    if 'Fixed V4' in results and 'Original V4' in results:
        fixed_grad = results['Fixed V4']['grad_norm']
        orig_grad = results['Original V4']['grad_norm']
        simple_grad = results['Simple']['grad_norm']
        
        print(f"Original V4 gradient norm: {orig_grad:.2e}")
        print(f"Fixed V4 gradient norm: {fixed_grad:.2e}")
        print(f"Simple model gradient norm: {simple_grad:.2e}")
        
        print(f"\nImprovement analysis:")
        print(f"  Fixed vs Original: {orig_grad/fixed_grad:.1f}x better")
        print(f"  Fixed vs Simple: {fixed_grad/simple_grad:.1f}x ratio")
        
        if fixed_grad < orig_grad * 0.1:
            print(f"✅ MAJOR IMPROVEMENT: Gradient explosion significantly reduced")
        elif fixed_grad < orig_grad * 0.5:
            print(f"🟡 MODERATE IMPROVEMENT: Some reduction in gradient explosion")
        else:
            print(f"❌ MINIMAL IMPROVEMENT: Fixes didn't help much")
        
        if abs(fixed_grad / simple_grad - 1.0) < 2.0:
            print(f"✅ GRADIENT PARITY: Fixed V4 now has comparable gradients to simple model")
        else:
            print(f"⚠️  STILL DIFFERENT: Fixed V4 gradients still differ significantly from simple model")
    
    print(f"\n💡 CONCLUSIONS:")
    print("If fixes worked, we've identified the core v4.py issues:")
    print("1. Missing weight initialization in transformer layers")
    print("2. Poor parameter initialization (std=1.0 too large)")
    print("3. Too large feed-forward dimensions")
    print("4. Need proper Xavier/Kaiming initialization throughout")

if __name__ == "__main__":
    main() 