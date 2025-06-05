#!/usr/bin/env python3
"""
Architectural breakdown test - systematically remove complexity to find the culprit.
Since initialization fixes didn't work, the architecture itself is the problem.
"""

import torch
import torch.nn as nn
from v4 import (ProteinInteractionClassifier, RoPEPositionalEncoding, 
                EnhancedTransformerLayer, CrossAttentionInteraction, 
                EnhancedMLPDecoder, ProteinEncoder)
from quick_fix_v4 import SimplifiedProteinClassifier

class MinimalTransformerLayer(nn.Module):
    """Minimal transformer - just self-attention + FFN, no RoPE"""
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Proper initialization
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x, src_key_padding_mask=None):
        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=src_key_padding_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x

class RoPEOnlyTransformerLayer(nn.Module):
    """Transformer with RoPE but standard architecture"""
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.rope = RoPEPositionalEncoding(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x, src_key_padding_mask=None):
        # Apply RoPE
        x_rope = self.rope(x)
        
        # Self-attention with residual
        attn_out, _ = self.self_attn(x_rope, x_rope, x_rope, key_padding_mask=src_key_padding_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x

class SimpleProteinEncoder(nn.Module):
    """Simplified encoder - just embedding + pooling, no transformer layers"""
    def __init__(self, input_dim=960, embed_dim=256):
        super().__init__()
        self.embed = nn.Linear(input_dim, embed_dim)
        self.pool = nn.Linear(embed_dim, input_dim)
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x, lengths):
        B, L, _ = x.shape
        device = x.device
        
        # Create mask for averaging
        mask = torch.arange(L, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        
        # Embed and average pool
        x_emb = self.embed(x)
        x_pooled = (x_emb * mask.unsqueeze(-1).float()).sum(dim=1) / lengths.unsqueeze(-1).float()
        
        # Project back to original dimension
        return self.pool(x_pooled)

class SingleLayerProteinEncoder(nn.Module):
    """Encoder with just one transformer layer"""
    def __init__(self, input_dim=960, embed_dim=256, transformer_type='minimal'):
        super().__init__()
        self.embed = nn.Linear(input_dim, embed_dim)
        
        if transformer_type == 'minimal':
            self.transformer = MinimalTransformerLayer(embed_dim, nhead=4, dim_feedforward=embed_dim*2)
        elif transformer_type == 'rope':
            self.transformer = RoPEOnlyTransformerLayer(embed_dim, nhead=4, dim_feedforward=embed_dim*2)
        elif transformer_type == 'enhanced':
            self.transformer = EnhancedTransformerLayer(embed_dim, nhead=4, dim_feedforward=embed_dim*2)
            
        self.pool = nn.Linear(embed_dim, input_dim)
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x, lengths):
        B, L, _ = x.shape
        device = x.device
        
        # Create padding mask
        mask = torch.arange(L, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
        
        # Embed input
        x_emb = self.embed(x)
        
        # Apply transformer
        x_trans = self.transformer(x_emb, src_key_padding_mask=mask)
        
        # Average pool with mask
        valid_mask = ~mask
        x_pooled = (x_trans * valid_mask.unsqueeze(-1).float()).sum(dim=1) / lengths.unsqueeze(-1).float()
        
        # Project back
        return self.pool(x_pooled)

class SimpleCrossAttention(nn.Module):
    """Minimal cross-attention without complex processing"""
    def __init__(self, embed_dim=960):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, emb_a, emb_b):
        # Simple cross-attention: A attends to B
        emb_a_seq = emb_a.unsqueeze(1)
        emb_b_seq = emb_b.unsqueeze(1)
        
        attended, _ = self.cross_attn(emb_a_seq, emb_b_seq, emb_b_seq)
        attended = attended.squeeze(1)
        
        # Simple residual connection
        return self.norm(attended + emb_a)

def test_architectural_components():
    """Test different levels of architectural complexity"""
    print("🔬 ARCHITECTURAL BREAKDOWN TEST")
    print("=" * 60)
    print("Testing gradient norms at different complexity levels")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 4
    
    # Test data
    emb_a = torch.randn(batch_size, 100, 960).to(device)
    emb_b = torch.randn(batch_size, 80, 960).to(device)
    lengths_a = torch.tensor([100] * batch_size).to(device)
    lengths_b = torch.tensor([80] * batch_size).to(device)
    
    results = {}
    
    # Test 1: Simple concatenation (baseline)
    print("\n1. Testing Simple Concatenation (no attention)")
    class SimpleConcatModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(960 * 2, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )
            self.apply(self._init_weights)
            
        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
        
        def forward(self, emb_a, emb_b, lengths_a, lengths_b):
            # Average pool
            mask_a = torch.arange(emb_a.size(1), device=emb_a.device).unsqueeze(0) < lengths_a.unsqueeze(1)
            mask_b = torch.arange(emb_b.size(1), device=emb_b.device).unsqueeze(0) < lengths_b.unsqueeze(1)
            
            avg_a = (emb_a * mask_a.unsqueeze(-1).float()).sum(dim=1) / lengths_a.unsqueeze(-1).float()
            avg_b = (emb_b * mask_b.unsqueeze(-1).float()).sum(dim=1) / lengths_b.unsqueeze(-1).float()
            
            # Concatenate and classify
            combined = torch.cat([avg_a, avg_b], dim=-1)
            return self.fc(combined)
    
    model = SimpleConcatModel().to(device)
    results['Simple Concat'] = test_model_gradients(model, emb_a, emb_b, lengths_a, lengths_b)
    
    # Test 2: Single layer encoders
    encoder_types = ['minimal', 'rope', 'enhanced']
    for enc_type in encoder_types:
        print(f"\n2.{enc_type[:1].upper()}. Testing Single Layer Encoder ({enc_type})")
        
        class SingleLayerModel(nn.Module):
            def __init__(self, transformer_type):
                super().__init__()
                self.encoder = SingleLayerProteinEncoder(transformer_type=transformer_type)
                self.decoder = nn.Sequential(
                    nn.Linear(960 * 2, 256),
                    nn.ReLU(), 
                    nn.Linear(256, 1)
                )
                self.apply(self._init_weights)
                
            def _init_weights(self, module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
            
            def forward(self, emb_a, emb_b, lengths_a, lengths_b):
                enc_a = self.encoder(emb_a, lengths_a)
                enc_b = self.encoder(emb_b, lengths_b)
                combined = torch.cat([enc_a, enc_b], dim=-1)
                return self.decoder(combined)
        
        model = SingleLayerModel(enc_type).to(device)
        results[f'Single {enc_type.title()}'] = test_model_gradients(model, emb_a, emb_b, lengths_a, lengths_b)
    
    # Test 3: Cross-attention complexity
    print("\n3. Testing Cross-Attention Complexity")
    
    class CrossAttentionModel(nn.Module):
        def __init__(self, cross_type='simple'):
            super().__init__()
            self.encoder = SimpleProteinEncoder()
            
            if cross_type == 'simple':
                self.cross_attn = SimpleCrossAttention()
            elif cross_type == 'enhanced':
                self.cross_attn = CrossAttentionInteraction()
                
            self.decoder = nn.Sequential(
                nn.Linear(960, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )
            self.apply(self._init_weights)
            
        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
        
        def forward(self, emb_a, emb_b, lengths_a, lengths_b):
            enc_a = self.encoder(emb_a, lengths_a)
            enc_b = self.encoder(emb_b, lengths_b)
            interaction = self.cross_attn(enc_a, enc_b)
            return self.decoder(interaction)
    
    for cross_type in ['simple', 'enhanced']:
        model = CrossAttentionModel(cross_type).to(device)
        results[f'Cross-Attn {cross_type.title()}'] = test_model_gradients(model, emb_a, emb_b, lengths_a, lengths_b)
    
    # Test 4: Full complexity (reference)
    print("\n4. Testing Full V4 Complexity")
    try:
        model = ProteinInteractionClassifier(
            encoder_layers=1,
            encoder_embed_dim=64,
            encoder_heads=2,
            decoder_hidden_dims=[64, 32]
        ).to(device)
        results['Full V4'] = test_model_gradients(model, emb_a, emb_b, lengths_a, lengths_b)
    except Exception as e:
        results['Full V4'] = {'error': str(e)}
    
    # Test 5: Simple model (reference)
    print("\n5. Testing Simple Model (reference)")
    model = SimplifiedProteinClassifier(hidden_dim=128).to(device)
    results['Simple Model'] = test_model_gradients(model, emb_a, emb_b, lengths_a, lengths_b)
    
    return results

def test_model_gradients(model, emb_a, emb_b, lengths_a, lengths_b):
    """Test gradient norms for a model"""
    try:
        output = model(emb_a, emb_b, lengths_a, lengths_b)
        loss = output.sum()
        loss.backward()
        
        grad_norm = sum([p.grad.norm().item() for p in model.parameters() if p.grad is not None])
        param_count = sum(p.numel() for p in model.parameters())
        
        print(f"   Gradient norm: {grad_norm:.2e}, Parameters: {param_count:,}")
        
        return {
            'grad_norm': grad_norm,
            'param_count': param_count,
            'output_mean': output.mean().item(),
            'output_std': output.std().item()
        }
    except Exception as e:
        print(f"   ERROR: {e}")
        return {'error': str(e)}

def main():
    """Run architectural breakdown analysis"""
    print("🏗️  V4 ARCHITECTURAL BREAKDOWN ANALYSIS")
    print("=" * 60)
    print("Systematically testing complexity levels to find the culprit")
    print("=" * 60)
    
    results = test_architectural_components()
    
    print(f"\n📊 GRADIENT NORM SUMMARY")
    print("=" * 40)
    
    # Sort by gradient norm
    valid_results = {k: v for k, v in results.items() if 'grad_norm' in v}
    sorted_results = sorted(valid_results.items(), key=lambda x: x[1]['grad_norm'])
    
    print("Model complexity (lowest to highest gradient norm):")
    for i, (name, stats) in enumerate(sorted_results):
        grad_norm = stats['grad_norm']
        param_count = stats['param_count']
        print(f"{i+1:2d}. {name:20s}: {grad_norm:8.2e} ({param_count:7,} params)")
    
    # Identify the problem components
    print(f"\n🎯 CULPRIT IDENTIFICATION")
    print("=" * 40)
    
    if len(sorted_results) >= 2:
        lowest_grad = sorted_results[0][1]['grad_norm']
        highest_grad = sorted_results[-1][1]['grad_norm']
        
        print(f"Gradient range: {lowest_grad:.2e} to {highest_grad:.2e}")
        print(f"Explosion factor: {highest_grad/lowest_grad:.1f}x")
        
        # Find where gradient explosion starts
        print(f"\nGradient explosion progression:")
        for i, (name, stats) in enumerate(sorted_results):
            if i == 0:
                baseline = stats['grad_norm']
                print(f"  {name:20s}: {stats['grad_norm']:8.2e} (baseline)")
            else:
                ratio = stats['grad_norm'] / baseline
                if ratio > 5:
                    print(f"  {name:20s}: {stats['grad_norm']:8.2e} (🚨 {ratio:.1f}x explosion)")
                elif ratio > 2:
                    print(f"  {name:20s}: {stats['grad_norm']:8.2e} (⚠️  {ratio:.1f}x increase)")
                else:
                    print(f"  {name:20s}: {stats['grad_norm']:8.2e} (✅ {ratio:.1f}x ok)")

if __name__ == "__main__":
    main() 