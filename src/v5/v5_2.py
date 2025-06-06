import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# Import v2's TransformerMAE
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'mask_autoencoder'))
from v2 import TransformerMAE

# Import v5's downstream components
from v5 import InteractionCrossAttention, InteractionMLPHead

class V2ToV5Adapter(nn.Module):
    """Adapter to convert v2 MAE output (512-dim) to v5 format (768-dim)"""
    def __init__(self, v2_dim=512, v5_dim=768, target_length=47):
        super().__init__()
        self.v2_dim = v2_dim
        self.v5_dim = v5_dim
        self.target_length = target_length
        
        # Dimension adapter
        self.dim_adapter = nn.Linear(v2_dim, v5_dim)
        
        # CLS token for v5 compatibility (v2 doesn't have CLS token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, v5_dim))
        
        # Position embedding for adapted tokens
        self.pos_embed = nn.Parameter(torch.randn(1, target_length + 1, v5_dim) * 0.02)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.dim_adapter.weight)
        nn.init.zeros_(self.dim_adapter.bias)
    
    def forward(self, v2_output, lengths):
        """
        Convert v2 MAE output to v5 format
        
        Args:
            v2_output: (B, L, 512) - output from v2 encoder
            lengths: (B,) - sequence lengths
            
        Returns:
            adapted_output: (B, target_length+1, 768) - v5 compatible format
        """
        B, L, _ = v2_output.shape
        
        # 1. Adapt dimensions from 512 to 768
        adapted = self.dim_adapter(v2_output)  # (B, L, 768)
        
        # 2. Truncate or pad to target_length (same as v5 patchify)
        if L > self.target_length:
            adapted = adapted[:, :self.target_length]
        elif L < self.target_length:
            pad_size = self.target_length - L
            padding = torch.zeros(B, pad_size, self.v5_dim, device=adapted.device, dtype=adapted.dtype)
            adapted = torch.cat([adapted, padding], dim=1)
        
        # 3. Add positional embedding
        adapted = adapted + self.pos_embed[:, 1:]  # Skip CLS position
        
        # 4. Add CLS token (v5 style)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_tokens = cls_tokens + self.pos_embed[:, :1]  # Add pos embed for CLS
        adapted = torch.cat([cls_tokens, adapted], dim=1)  # (B, target_length+1, 768)
        
        return adapted

class V2MaeEncoder(nn.Module):
    """Wrapper around pretrained v2 MAE encoder for inference only"""
    def __init__(self, pretrained_path: str):
        super().__init__()
        
        # Load pretrained v2 MAE
        self.v2_mae = TransformerMAE(
            input_dim=960,
            embed_dim=512,
            mask_ratio=0.0,  # No masking during inference
            num_layers=4,
            nhead=16,
            ff_dim=2048,
            max_len=1502
        )
        
        # Load pretrained weights
        self.load_pretrained_weights(pretrained_path)
        
        # Freeze v2 MAE weights
        for param in self.v2_mae.parameters():
            param.requires_grad = False
        
        # Adapter to convert to v5 format
        self.adapter = V2ToV5Adapter()
        
    def load_pretrained_weights(self, checkpoint_path: str):
        """Load pretrained v2 MAE weights"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            self.v2_mae.load_state_dict(state_dict, strict=False)
            print(f"✅ Loaded pretrained v2 MAE weights from {checkpoint_path}")
            
        except Exception as e:
            print(f"❌ Error loading v2 MAE weights: {e}")
            raise e
    
    def forward(self, x, lengths):
        """
        Forward pass using pretrained v2 MAE
        
        Args:
            x: (B, L, 960) - protein embeddings
            lengths: (B,) - sequence lengths
            
        Returns:
            adapted_output: (B, 48, 768) - v5 compatible format
        """
        with torch.no_grad():
            # Get v2 MAE encoder output (no masking during inference)
            # We only need the encoder part, not reconstruction
            device = x.device
            B, L, _ = x.shape
            
            # Embed and add position encoding (from v2)
            x_emb = self.v2_mae.embed(x)  # (B, L, 512)
            x_emb = x_emb + self.v2_mae.pos_embed[:, :L]
            
            # Create padding mask
            arange = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
            mask_pad = arange >= lengths.unsqueeze(1)
            
            # Pass through transformer encoder
            enc_out = self.v2_mae.encoder(x_emb, src_key_padding_mask=mask_pad)  # (B, L, 512)
        
        # Adapt to v5 format
        adapted_output = self.adapter(enc_out, lengths)  # (B, 48, 768)
        
        return adapted_output

class PPIClassifierV52(nn.Module):
    """V5.2 PPI Classifier using pretrained v2 MAE with v5 downstream components"""
    def __init__(self, v2_mae_path: str):
        super().__init__()
        
        # Pretrained v2 MAE encoder (frozen)
        self.encoder = V2MaeEncoder(v2_mae_path)
        
        # V5 downstream components (trainable)
        self.cross_attn = InteractionCrossAttention(d_model=768, n_heads=8, n_layers=2)
        self.mlp_head = InteractionMLPHead(input_dim=768, hidden_dim1=512, hidden_dim2=128)
        
        print("🔧 V5.2 Architecture:")
        print("  - Encoder: Pretrained v2 MAE (frozen, 512-dim → adapter → 768-dim)")
        print("  - Cross-attention: Trainable v5 components")
        print("  - MLP head: Trainable v5 components")
    
    def forward(self, emb_a, emb_b, lengths_a, lengths_b):
        """
        Forward pass for PPI classification
        
        Args:
            emb_a: (B, L, 960) - protein A embeddings
            emb_b: (B, L, 960) - protein B embeddings
            lengths_a: (B,) - protein A sequence lengths
            lengths_b: (B,) - protein B sequence lengths
            
        Returns:
            logits: (B,) - classification logits
        """
        # Encode through pretrained v2 MAE + adapter
        tok_a = self.encoder(emb_a, lengths_a)  # (B, 48, 768)
        tok_b = self.encoder(emb_b, lengths_b)  # (B, 48, 768)
        
        # Cross-attention for interaction (v5 components)
        z_int = self.cross_attn(tok_a, tok_b)  # (B, 768)
        
        # Final classification (v5 components)
        logits = self.mlp_head(z_int)  # (B, 1)
        
        return logits.squeeze(-1)  # (B,)
    
    def get_interaction_embeddings(self, emb_a, emb_b, lengths_a, lengths_b):
        """Get interaction embeddings without final classification"""
        with torch.no_grad():
            tok_a = self.encoder(emb_a, lengths_a)
            tok_b = self.encoder(emb_b, lengths_b)
            z_int = self.cross_attn(tok_a, tok_b)
        return z_int

def create_ppi_classifier_v52(v2_mae_path: str) -> PPIClassifierV52:
    """
    Create V5.2 PPI classifier using pretrained v2 MAE
    
    Args:
        v2_mae_path: Path to pretrained v2 MAE checkpoint
    
    Returns:
        PPIClassifierV52 model
    """
    model = PPIClassifierV52(v2_mae_path)
    return model

def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

if __name__ == "__main__":
    # Test the V5.2 architecture
    print("🧪 Testing V5.2 PPI Classifier architecture...")
    
    # Note: Replace with your actual v2 MAE checkpoint path
    v2_mae_path = "path/to/your/v2_mae_checkpoint.pth"
    
    try:
        # Create model
        model = create_ppi_classifier_v52(v2_mae_path)
        
        # Count parameters
        total_params, trainable_params = count_parameters(model)
        print(f"\n📊 Model Statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Frozen parameters: {total_params - trainable_params:,}")
        print(f"  Trainable ratio: {trainable_params/total_params*100:.1f}%")
        
        # Test forward pass
        batch_size = 2
        seq_len = 100
        emb_a = torch.randn(batch_size, seq_len, 960)
        emb_b = torch.randn(batch_size, seq_len, 960)
        lengths_a = torch.tensor([80, 100])
        lengths_b = torch.tensor([90, 100])
        
        print(f"\n🔄 Testing forward pass:")
        print(f"  Input shapes: emb_a={emb_a.shape}, emb_b={emb_b.shape}")
        
        # Forward pass
        logits = model(emb_a, emb_b, lengths_a, lengths_b)
        print(f"  Output logits shape: {logits.shape}")
        
        # Test interaction embeddings
        interaction_emb = model.get_interaction_embeddings(emb_a, emb_b, lengths_a, lengths_b)
        print(f"  Interaction embeddings shape: {interaction_emb.shape}")
        
        print("\n✅ V5.2 architecture test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print("Make sure to set the correct v2_mae_path") 