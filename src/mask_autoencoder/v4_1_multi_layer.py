"""
Enhanced v4.1 with configurable transformer layers (2-3 layers)
Middle ground between complex v4 and ultra-simple v4.1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from v4_1 import load_data, ProteinPairDataset, collate_fn, SimpleInteractionLayer, SimpleMLPDecoder

class MultiLayerProteinEncoder(nn.Module):
    """
    Enhanced protein encoder with 2-3 lightweight transformer layers
    Still targeting 200-500K parameters but with more representational capacity
    """
    def __init__(self, input_dim=960, embed_dim=128, num_layers=2, num_heads=4, max_length=512):
        super().__init__()
        
        self.max_length = max_length
        self.num_layers = num_layers
        
        # Input projection to smaller dimension
        self.input_proj = nn.Linear(input_dim, embed_dim)
        
        # Multiple lightweight transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                feedforward_dim=embed_dim * 2,  # Keep FFN smaller than typical 4x
                dropout=0.1
            ) for _ in range(num_layers)
        ])
        
        # Simple pooling for sequence compression
        self.pool_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x, lengths):
        """
        Args:
            x: (B, L, 960) padded protein embeddings
            lengths: (B,) actual sequence lengths
        Returns:
            (B, embed_dim) protein representation
        """
        B, L, _ = x.shape
        device = x.device
        
        # Truncate if too long for memory efficiency
        if L > self.max_length:
            x = x[:, :self.max_length]
            lengths = torch.clamp(lengths, max=self.max_length)
            L = self.max_length
        
        # Create padding mask
        mask = torch.arange(L, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
        
        # Project to smaller dimension
        x = self.input_proj(x)  # (B, L, embed_dim)
        
        # Apply multiple transformer layers
        for layer in self.transformer_layers:
            x = layer(x, mask)
        
        # Simple average pooling with masking
        valid_mask = ~mask  # (B, L)
        valid_counts = valid_mask.sum(dim=1, keepdim=True).float()  # (B, 1)
        
        # Masked average pooling
        x_masked = x * valid_mask.unsqueeze(-1).float()  # (B, L, embed_dim)
        pooled = x_masked.sum(dim=1) / torch.clamp(valid_counts, min=1.0)  # (B, embed_dim)
        
        # Final projection
        output = self.pool_proj(pooled)  # (B, embed_dim)
        
        return output


class TransformerEncoderLayer(nn.Module):
    """
    Lightweight transformer encoder layer
    Simpler than standard transformer but more powerful than single attention
    """
    def __init__(self, embed_dim, num_heads, feedforward_dim, dropout=0.1):
        super().__init__()
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Simple feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, embed_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, padding_mask=None):
        """
        Args:
            x: (B, L, embed_dim) input sequence
            padding_mask: (B, L) boolean mask (True for padding)
        Returns:
            (B, L, embed_dim) output sequence
        """
        # Self-attention with residual connection
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=padding_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feedforward with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x


class EnhancedProteinInteractionClassifier(nn.Module):
    """
    Enhanced v4.1 with 2-3 transformer layers
    
    Improvements over basic v4.1:
    1. 2-3 lightweight transformer layers (vs 1)
    2. Better representational capacity
    3. Still maintains parameter efficiency (200-500K target)
    4. Stable training with higher learning rates
    
    Architecture:
    1. Protein A/B → MultiLayerProteinEncoder → Protein Representations
    2. Simple concatenation and interaction processing  
    3. Simple MLP decoder for binary classification
    """
    def __init__(self, embed_dim=128, num_layers=2, num_heads=4, max_length=512):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # Shared multi-layer protein encoder
        self.protein_encoder = MultiLayerProteinEncoder(
            input_dim=960,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_length=max_length
        )
        
        # Simple interaction layer (concatenation-based)
        self.interaction_layer = SimpleInteractionLayer(embed_dim=embed_dim)
        
        # Simple MLP decoder
        self.decoder = SimpleMLPDecoder(
            input_dim=embed_dim,
            hidden_dim=embed_dim // 2
        )
        
        # Initialize weights properly
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, emb_a, emb_b, lengths_a, lengths_b):
        """
        Args:
            emb_a: (B, L_a, 960) protein A embeddings
            emb_b: (B, L_b, 960) protein B embeddings  
            lengths_a: (B,) actual lengths for protein A
            lengths_b: (B,) actual lengths for protein B
        Returns:
            (B, 1) interaction prediction logits
        """
        # Encode both proteins with shared multi-layer encoder
        repr_a = self.protein_encoder(emb_a, lengths_a)  # (B, embed_dim)
        repr_b = self.protein_encoder(emb_b, lengths_b)  # (B, embed_dim)
        
        # Simple interaction processing
        interaction_repr = self.interaction_layer(repr_a, repr_b)  # (B, embed_dim)
        
        # Classification
        logits = self.decoder(interaction_repr)  # (B, 1)
        
        return logits


def count_parameters(model):
    """Count model parameters for analysis"""
    return sum(p.numel() for p in model.parameters())


def create_enhanced_v4_1_variants():
    """Create different enhanced v4.1 model variants for comparison"""
    
    variants = []
    
    # Test different layer counts and embedding dimensions
    configs = [
        # 2-layer variants
        {'layers': 2, 'embed_dim': 96, 'heads': 4, 'name': 'Enhanced-v4.1-2L-96d'},
        {'layers': 2, 'embed_dim': 128, 'heads': 4, 'name': 'Enhanced-v4.1-2L-128d'},
        {'layers': 2, 'embed_dim': 160, 'heads': 8, 'name': 'Enhanced-v4.1-2L-160d'},
        
        # 3-layer variants
        {'layers': 3, 'embed_dim': 96, 'heads': 4, 'name': 'Enhanced-v4.1-3L-96d'},
        {'layers': 3, 'embed_dim': 128, 'heads': 4, 'name': 'Enhanced-v4.1-3L-128d'},
        
        # Single layer for comparison (original v4.1)
        {'layers': 1, 'embed_dim': 128, 'heads': 4, 'name': 'Original-v4.1-1L-128d'},
    ]
    
    for config in configs:
        # Create model
        if config['layers'] == 1:
            # Use original v4.1 SimplifiedProteinInteractionClassifier
            from v4_1 import SimplifiedProteinInteractionClassifier
            model = SimplifiedProteinInteractionClassifier(
                embed_dim=config['embed_dim']
            )
        else:
            # Use enhanced multi-layer version
            model = EnhancedProteinInteractionClassifier(
                embed_dim=config['embed_dim'],
                num_layers=config['layers'],
                num_heads=config['heads']
            )
        
        num_params = count_parameters(model)
        
        variants.append({
            'model': model,
            'config': config,
            'num_parameters': num_params,
            'name': config['name']
        })
        
        print(f"{config['name']}: {num_params:,} parameters")
    
    return variants


if __name__ == "__main__":
    print("🔧 Enhanced v4.1 Architecture Analysis")
    print("=" * 50)
    print("Testing multi-layer variants of v4.1 architecture")
    print("=" * 50)
    
    # Create and analyze different variants
    variants = create_enhanced_v4_1_variants()
    
    print(f"\n📊 Parameter Analysis:")
    print(f"{'Model':<25} {'Layers':<7} {'Embed':<6} {'Heads':<6} {'Parameters':<12}")
    print("-" * 60)
    
    for variant in variants:
        config = variant['config']
        print(f"{config['name']:<25} {config['layers']:<7} {config['embed_dim']:<6} "
              f"{config['heads']:<6} {variant['num_parameters']:,<12}")
    
    # Identify models in target range
    target_models = [v for v in variants if 150000 <= v['num_parameters'] <= 500000]
    
    print(f"\n🎯 Models in target range (150K-500K parameters):")
    for model in target_models:
        config = model['config']
        print(f"   ✅ {config['name']}: {model['num_parameters']:,} params")
    
    # Analysis and recommendations
    print(f"\n💡 Analysis:")
    print(f"   • Original v4.1 (1 layer): Very simple but may lack capacity")
    print(f"   • 2-layer variants: Good balance of complexity vs efficiency")
    print(f"   • 3-layer variants: More capacity but higher parameter count")
    print(f"   • All variants stay within reasonable parameter budgets")
    
    print(f"\n🚀 Recommended for testing:")
    recommended = [v for v in target_models if v['config']['layers'] >= 2][:3]
    for i, model in enumerate(recommended, 1):
        config = model['config']
        print(f"   {i}. {config['name']}: {config['layers']} layers, {config['embed_dim']} embed_dim, {model['num_parameters']:,} params") 