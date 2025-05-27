import torch
import torch.nn as nn
import math


class AttnPool(nn.Module):
    """Attention pooling layer that converts variable-length sequences to fixed-size vectors."""
    
    def __init__(self, d_model):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d_model))  # (C,)
    
    def forward(self, x, mask):
        """
        Args:
            x: (B, C, L) - feature tensor
            mask: (B, L) - boolean mask where True indicates valid positions
        Returns:
            (B, C) - pooled features
        """
        q = self.q.unsqueeze(0).unsqueeze(-1)        # (1, C, 1)
        logits = (x * q).sum(1) / math.sqrt(x.size(1))   # (B, L)
        logits = logits.masked_fill(~mask, -1e9)
        w = torch.softmax(logits, dim=-1)            # (B, L)
        return (w.unsqueeze(1) @ x.transpose(1,2)).squeeze(1)  # (B, C)


class TCNBlock(nn.Module):
    """Residual TCN block with dilated convolutions."""
    
    def __init__(self, C, dilation):
        super().__init__()
        self.conv1 = nn.Conv1d(C, C, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(C, C, 3, padding=1)
        self.norm = nn.LayerNorm(C)
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.1)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, L) - input features
        Returns:
            (B, C, L) - output features with residual connection
        """
        y = self.act(self.conv1(x))
        y = self.drop(self.conv2(y))
        # Apply LayerNorm over channels (need to transpose for LayerNorm)
        y = self.norm((x + y).transpose(1, 2)).transpose(1, 2)   # Res + LN
        return y


class ProteinTCNEncoder(nn.Module):
    """
    Protein TCN Encoder that converts ESM-C embeddings to fixed-length protein vectors.
    
    Architecture:
    - Input: (B, L, 960) ESM-C residue embeddings
    - Stem: 1D conv to project to d_out channels
    - 5 residual TCN blocks with dilations [1, 2, 4, 8, 16]
    - Attention pooling to get fixed-length representation
    - Output: (B, d_out) protein vectors
    """
    
    def __init__(self, d_out=512):
        super().__init__()
        self.d_out = d_out
        
        # Stem: project from 960 to d_out channels
        self.stem = nn.Conv1d(960, d_out, 3, padding=1)
        self.stem_act = nn.GELU()
        
        # 5 residual blocks with increasing dilations
        dilations = [1, 2, 4, 8, 16]
        self.blocks = nn.ModuleList([TCNBlock(d_out, d) for d in dilations])
        
        # Attention pooling
        self.pool = AttnPool(d_out)
        
        # Final projection head
        self.norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(0.1)
    
    def forward(self, emb, mask):
        """
        Args:
            emb: (B, L, 960) - ESM-C residue embeddings
            mask: (B, L) - boolean mask where True indicates valid positions
        Returns:
            (B, d_out) - fixed-length protein vectors
        """
        # Permute to (B, C, L) for Conv1d
        x = emb.permute(0, 2, 1)  # (B, 960, L)
        
        # Stem convolution
        x = self.stem_act(self.stem(x))  # (B, d_out, L)
        
        # Apply residual TCN blocks
        for blk in self.blocks:
            x = blk(x)  # (B, d_out, L)
        
        # Attention pooling
        h = self.pool(x, mask)  # (B, d_out)
        
        # Final projection
        return self.drop(self.norm(h))  # (B, d_out)