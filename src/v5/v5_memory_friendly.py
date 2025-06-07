import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class PatchEmbedding(nn.Module):
    """Converts 32x960 residue chunks into 512-d tokens"""
    def __init__(self, input_dim=960, embed_dim=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.projection = nn.Linear(input_dim, embed_dim)
        
    def forward(self, x):
        # x: (B, L, 960) -> (B, L, 512)
        return self.projection(x)

class MAEEncoder(nn.Module):
    """Memory-friendly MAE Encoder with 512-d embeddings"""
    def __init__(self,
                 input_dim=960,
                 embed_dim=512,
                 num_layers=8,
                 nhead=8,
                 ff_dim=2048,
                 max_len=1502,
                 dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(input_dim, embed_dim)
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
    def patchify(self, x, target_length=47):
        """
        Convert input embeddings to patches and add CLS token
        x: (B, L, 960) -> patches: (B, T+1, 512) where T=target_length
        """
        B, L, _ = x.shape
        
        # Project to embed_dim
        x = self.patch_embed(x)  # (B, L, 512)
        
        # Truncate or pad to target_length
        if L > target_length:
            x = x[:, :target_length]  # Truncate
        elif L < target_length:
            # Pad with zeros
            pad_size = target_length - L
            padding = torch.zeros(B, pad_size, self.embed_dim, device=x.device, dtype=x.dtype)
            x = torch.cat([x, padding], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed[:, :target_length]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_tokens = cls_tokens + self.pos_embed[:, :1]  # Add pos embed for CLS
        x = torch.cat([cls_tokens, x], dim=1)  # (B, T+1, 512)
        
        return x
    
    def forward(self, x):
        """
        x: (B, T+1, 512) - already patchified with CLS token
        Returns: (B, T+1, 512)
        """
        # Pass through transformer encoder
        x = self.encoder(x)
        x = self.norm(x)
        return x

class InteractionCrossAttention(nn.Module):
    """Memory-friendly cross-attention module for protein-protein interactions"""
    def __init__(self, d_model=512, n_heads=8, n_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # Learnable CLS_int token
        self.cls_int = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,  # 2048
                dropout=dropout,
                batch_first=True,
                norm_first=True
            ) for _ in range(n_layers)
        ])
        
        # Layer norm
        self.norm = nn.LayerNorm(d_model)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_int, std=0.02)
    
    def forward(self, tok_a, tok_b):
        """
        tok_a: (B, T+1, 512) - encoded tokens from protein A
        tok_b: (B, T+1, 512) - encoded tokens from protein B
        Returns: (B, 512) - interaction vector
        """
        B = tok_a.shape[0]
        
        # Concatenate protein tokens (excluding their CLS tokens)
        # We use tokens from position 1: to exclude CLS tokens from individual proteins
        protein_tokens = torch.cat([tok_a[:, 1:], tok_b[:, 1:]], dim=1)  # (B, 2*T, 512)
        
        # Add learnable CLS_int token
        cls_int = self.cls_int.expand(B, -1, -1)  # (B, 1, 512)
        
        # Combine: [CLS_int] + [protein_A_tokens] + [protein_B_tokens]
        combined = torch.cat([cls_int, protein_tokens], dim=1)  # (B, 2*T+1, 512)
        
        # Pass through cross-attention layers
        for layer in self.cross_attn_layers:
            combined = layer(combined)
        
        # Extract and normalize the CLS_int token
        interaction_vector = self.norm(combined[:, 0])  # (B, 512)
        
        return interaction_vector

class InteractionMLPHead(nn.Module):
    """Memory-friendly MLP head for final classification"""
    def __init__(self, input_dim=512, hidden_dim1=256, hidden_dim2=64, output_dim=1, dropout=0.2):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.LayerNorm(hidden_dim1),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.LayerNorm(hidden_dim2),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim2, output_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        x: (B, 512) -> (B, 1)
        """
        return self.layers(x)

class PPIClassifier(nn.Module):
    """Memory-friendly PPI classifier with 512-d embeddings"""
    def __init__(self, 
                 mae_encoder_path: Optional[str] = None,
                 freeze_encoder: bool = True,
                 use_lora: bool = False,
                 lora_r: int = 4,
                 lora_alpha: int = 16):
        super().__init__()
        
        # Initialize memory-friendly MAE encoder
        self.encoder = MAEEncoder()
        
        # Load pre-trained weights if provided
        if mae_encoder_path:
            self.load_mae_weights(mae_encoder_path)
        
        # Freeze encoder weights if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Optional LoRA adaptation
        if use_lora and not freeze_encoder:
            self.apply_lora(lora_r, lora_alpha)
        
        # Memory-friendly cross-attention and MLP head
        self.cross_attn = InteractionCrossAttention()
        self.mlp_head = InteractionMLPHead()
    
    def load_mae_weights(self, checkpoint_path: str):
        """Load pre-trained MAE weights"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Adapt state dict keys if necessary
            adapted_state_dict = {}
            for key, value in state_dict.items():
                # Remove 'mae.' prefix if present and only take encoder parts
                if key.startswith('mae.'):
                    new_key = key.replace('mae.', '')
                else:
                    new_key = key
                
                # Only load encoder-related parameters
                if any(enc_key in new_key for enc_key in ['patch_embed', 'pos_embed', 'cls_token', 'encoder', 'norm']):
                    adapted_state_dict[new_key] = value
            
            # Load compatible weights
            missing_keys, unexpected_keys = self.encoder.load_state_dict(adapted_state_dict, strict=False)
            print(f"Loaded pre-trained MAE weights from {checkpoint_path}")
            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")
            
        except Exception as e:
            print(f"Warning: Could not load pre-trained weights from {checkpoint_path}: {e}")
    
    def apply_lora(self, r: int, alpha: int):
        """Apply LoRA adaptation to attention layers"""
        try:
            from peft import LoraConfig, get_peft_model
            
            lora_config = LoraConfig(
                r=r,
                lora_alpha=alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1,
            )
            
            self.encoder = get_peft_model(self.encoder, lora_config)
            print(f"Applied LoRA with r={r}, alpha={alpha}")
            
        except ImportError:
            print("Warning: peft library not available, skipping LoRA")
        except Exception as e:
            print(f"Warning: Could not apply LoRA: {e}")
    
    def forward(self, emb_a, emb_b):
        """
        Forward pass for PPI classification
        
        Args:
            emb_a: (B, L, 960) - protein A embeddings
            emb_b: (B, L, 960) - protein B embeddings
            
        Returns:
            logits: (B,) - classification logits
        """
        # Patchify inputs (convert to patches and add CLS tokens)
        patches_a = self.encoder.patchify(emb_a)  # (B, T+1, 512)
        patches_b = self.encoder.patchify(emb_b)  # (B, T+1, 512)
        
        # Encode through MAE encoder
        tok_a = self.encoder(patches_a)  # (B, T+1, 512)
        tok_b = self.encoder(patches_b)  # (B, T+1, 512)
        
        # Cross-attention for interaction
        z_int = self.cross_attn(tok_a, tok_b)  # (B, 512)
        
        # Final classification
        logits = self.mlp_head(z_int)  # (B, 1)
        
        return logits.squeeze(-1)  # (B,)
    
    def get_interaction_embeddings(self, emb_a, emb_b):
        """
        Get interaction embeddings without final classification
        Useful for downstream analysis or clustering
        """
        with torch.no_grad():
            patches_a = self.encoder.patchify(emb_a)
            patches_b = self.encoder.patchify(emb_b)
            tok_a = self.encoder(patches_a)
            tok_b = self.encoder(patches_b)
            z_int = self.cross_attn(tok_a, tok_b)
        return z_int

# Factory function for easy model creation
def create_ppi_classifier(mae_checkpoint_path: str = "models/v5_compatible_mae_best_20250607-012709.pth", 
                         freeze_encoder: bool = True,
                         use_lora: bool = False) -> PPIClassifier:
    """
    Create a memory-friendly PPI classifier with optional pre-trained MAE weights
    
    Args:
        mae_checkpoint_path: Path to pre-trained MAE checkpoint
        freeze_encoder: Whether to freeze the encoder weights
        use_lora: Whether to use LoRA for fine-tuning
    
    Returns:
        PPIClassifier model
    """
    model = PPIClassifier(
        mae_encoder_path=mae_checkpoint_path,
        freeze_encoder=freeze_encoder,
        use_lora=use_lora
    )
    return model

if __name__ == "__main__":
    # Test the memory-friendly model architecture
    print("Testing Memory-Friendly PPI Classifier architecture...")
    
    # Create model
    model = create_ppi_classifier()
    
    # Test forward pass
    batch_size = 2
    seq_len = 100
    emb_a = torch.randn(batch_size, seq_len, 960)
    emb_b = torch.randn(batch_size, seq_len, 960)
    
    print(f"Input shapes: emb_a={emb_a.shape}, emb_b={emb_b.shape}")
    
    # Forward pass
    logits = model(emb_a, emb_b)
    print(f"Output logits shape: {logits.shape}")
    
    # Test interaction embeddings
    interaction_emb = model.get_interaction_embeddings(emb_a, emb_b)
    print(f"Interaction embeddings shape: {interaction_emb.shape}")
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nMemory-Friendly Model Summary:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    # Compare with original v5 dimensions
    print(f"\nDimensionality:")
    print(f"Embedding dimension: 512 (vs 768 original)")
    print(f"Encoder layers: 8 (vs 12 original)")
    print(f"Attention heads: 8 (vs 12 original)")
    print(f"Feed-forward dim: 2048 (vs 3072 original)")
    print(f"MLP hidden dims: 256, 64 (vs 512, 128 original)")
    
    print("\nMemory-friendly architecture test completed successfully!") 