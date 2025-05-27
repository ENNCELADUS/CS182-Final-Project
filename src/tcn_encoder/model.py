import torch
import torch.nn as nn
from .encoder import ProteinTCNEncoder


class SingleProteinModel(nn.Module):
    """Model for single protein tasks (classification or regression)."""
    
    def __init__(self, n_classes=None, d_encoder=512, d_hidden=256, dropout=0.2, task_type='classification'):
        """
        Args:
            n_classes: Number of classes for classification (None for regression)
            d_encoder: Dimension of encoder output
            d_hidden: Hidden dimension in MLP head
            dropout: Dropout rate
            task_type: 'classification' or 'regression'
        """
        super().__init__()
        self.task_type = task_type
        self.n_classes = n_classes
        
        # Encoder
        self.encoder = ProteinTCNEncoder(d_out=d_encoder)
        
        # Task head
        if task_type == 'classification':
            assert n_classes is not None, "n_classes must be specified for classification"
            self.head = nn.Sequential(
                nn.Linear(d_encoder, d_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_hidden, n_classes)
            )
        elif task_type == 'regression':
            self.head = nn.Sequential(
                nn.Linear(d_encoder, d_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_hidden, 1)
            )
        else:
            raise ValueError(f"Unknown task_type: {task_type}")
    
    def forward(self, emb, mask):
        """
        Args:
            emb: (B, L, 960) - ESM-C residue embeddings
            mask: (B, L) - boolean mask
        Returns:
            (B, n_classes) for classification or (B, 1) for regression
        """
        h = self.encoder(emb, mask)  # (B, d_encoder)
        return self.head(h)


class ProteinPairModel(nn.Module):
    """Model for protein-pair tasks like PPI (Protein-Protein Interaction)."""
    
    def __init__(self, n_classes=1, d_encoder=512, d_hidden=512, dropout=0.2, task_type='binary_classification'):
        """
        Args:
            n_classes: Number of classes (1 for binary, >1 for multi-class)
            d_encoder: Dimension of encoder output
            d_hidden: Hidden dimension in MLP head
            dropout: Dropout rate
            task_type: 'binary_classification', 'multi_classification', or 'regression'
        """
        super().__init__()
        self.task_type = task_type
        self.n_classes = n_classes
        
        # Shared encoder for both proteins
        self.encoder = ProteinTCNEncoder(d_out=d_encoder)
        
        # Pair interaction features: [hA, hB, |hA-hB|, hA*hB]
        pair_dim = 4 * d_encoder  # 2048 for default d_encoder=512
        
        # Task head
        if task_type in ['binary_classification', 'multi_classification']:
            self.head = nn.Sequential(
                nn.Linear(pair_dim, d_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_hidden, d_hidden // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_hidden // 2, n_classes)
            )
        elif task_type == 'regression':
            self.head = nn.Sequential(
                nn.Linear(pair_dim, d_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_hidden, d_hidden // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_hidden // 2, 1)
            )
        else:
            raise ValueError(f"Unknown task_type: {task_type}")
    
    def forward(self, embA, maskA, embB, maskB):
        """
        Args:
            embA: (B, L_A, 960) - ESM-C embeddings for protein A
            maskA: (B, L_A) - mask for protein A
            embB: (B, L_B, 960) - ESM-C embeddings for protein B
            maskB: (B, L_B) - mask for protein B
        Returns:
            (B, n_classes) for classification or (B, 1) for regression
        """
        # Encode both proteins
        hA = self.encoder(embA, maskA)  # (B, d_encoder)
        hB = self.encoder(embB, maskB)  # (B, d_encoder)
        
        # Create pair interaction features
        feat = torch.cat([
            hA,                    # (B, d_encoder)
            hB,                    # (B, d_encoder)
            torch.abs(hA - hB),    # (B, d_encoder) - element-wise difference
            hA * hB                # (B, d_encoder) - element-wise product
        ], dim=-1)  # (B, 4 * d_encoder)
        
        return self.head(feat)


class MultiTaskModel(nn.Module):
    """Model that can handle multiple tasks simultaneously."""
    
    def __init__(self, task_configs, d_encoder=512):
        """
        Args:
            task_configs: Dict of task_name -> config dict
                Each config should have: 'type', 'n_classes', 'd_hidden', 'dropout'
            d_encoder: Dimension of encoder output
        """
        super().__init__()
        self.task_configs = task_configs
        
        # Shared encoder
        self.encoder = ProteinTCNEncoder(d_out=d_encoder)
        
        # Task-specific heads
        self.heads = nn.ModuleDict()
        for task_name, config in task_configs.items():
            if config['type'] == 'classification':
                self.heads[task_name] = nn.Sequential(
                    nn.Linear(d_encoder, config['d_hidden']),
                    nn.GELU(),
                    nn.Dropout(config['dropout']),
                    nn.Linear(config['d_hidden'], config['n_classes'])
                )
            elif config['type'] == 'regression':
                self.heads[task_name] = nn.Sequential(
                    nn.Linear(d_encoder, config['d_hidden']),
                    nn.GELU(),
                    nn.Dropout(config['dropout']),
                    nn.Linear(config['d_hidden'], 1)
                )
    
    def forward(self, emb, mask, task_name=None):
        """
        Args:
            emb: (B, L, 960) - ESM-C residue embeddings
            mask: (B, L) - boolean mask
            task_name: If specified, only return output for this task
        Returns:
            Dict of task_name -> output tensor, or single tensor if task_name specified
        """
        h = self.encoder(emb, mask)  # (B, d_encoder)
        
        if task_name is not None:
            return self.heads[task_name](h)
        else:
            return {task: head(h) for task, head in self.heads.items()}