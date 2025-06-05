import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
import json
import matplotlib
matplotlib.use('Agg')
import gc
import pickle
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import math

# First, let's examine the data structure to identify column names
def examine_dataframe(df):
    """Print the structure of the dataframe to identify column names"""
    print("DataFrame columns:", df.columns.tolist())
    print("First row sample:", df.iloc[0].to_dict())
    return df.columns.tolist()


def load_data():
    """Load the actual data from the project structure"""
    # Get the current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Navigate to the project root (CS182-Final-Project)
    # From src/mask_autoencoder/ go up two levels to reach project root
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    print(f"Script directory: {current_dir}")
    print(f"Project root: {project_root}")
    print("Loading data...")
    
    # Construct full paths to data files
    data_dir = os.path.join(project_root, 'data', 'full_dataset')
    embeddings_dir = os.path.join(data_dir, 'embeddings')
    
    print(f"Looking for data in: {data_dir}")
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Load data files with full paths
    train_path = os.path.join(data_dir, 'train_data.pkl')
    cv_path = os.path.join(data_dir, 'validation_data.pkl')
    test1_path = os.path.join(data_dir, 'test1_data.pkl')
    test2_path = os.path.join(data_dir, 'test2_data.pkl')
    embeddings_path = os.path.join(embeddings_dir, 'embeddings_standardized.pkl')
    
    # Check if all files exist
    for path, name in [(train_path, 'train_data.pkl'), 
                       (cv_path, 'validation_data.pkl'),
                       (test1_path, 'test1_data.pkl'), 
                       (test2_path, 'test2_data.pkl'),
                       (embeddings_path, 'embeddings_standardized.pkl')]:
        if not os.path.exists(path):
            print(f"❌ Missing: {path}")
        else:
            print(f"✅ Found: {name}")
    
    # Load the data
    train_data = pickle.load(open(train_path, 'rb'))
    cv_data = pickle.load(open(cv_path, 'rb'))
    test1_data = pickle.load(open(test1_path, 'rb'))
    test2_data = pickle.load(open(test2_path, 'rb'))

    # Examine structure of the first dataframe to understand its format
    print("\nExamining training data structure:")
    examine_dataframe(train_data)

    print("\nLoading protein embeddings...")
    protein_embeddings = pickle.load(open(embeddings_path, 'rb'))
    print(f"Loaded {len(protein_embeddings)} protein embeddings")

    print(train_data.head())
    for i, (key, value) in enumerate(protein_embeddings.items()):
        if i >= 5:
            break
        print(f"Protein ID: {key}, Embedding shape: {value.shape}")
    
    return train_data, cv_data, test1_data, test2_data, protein_embeddings


def detect_column_names(data_df, embeddings_dict):
    """Automatically detect column names for protein IDs and interaction labels"""
    columns = data_df.columns.tolist()
    
    # Determine protein ID and interaction columns
    protein_a_col = None
    protein_b_col = None
    interaction_col = None
    
    # Common column name patterns
    protein_a_patterns = ['protein_a', 'protein_id_a', 'proteinA', 'proteinIDA', 'protein_A', 'protein_id_A']
    protein_b_patterns = ['protein_b', 'protein_id_b', 'proteinB', 'proteinIDB', 'protein_B', 'protein_id_B']
    interaction_patterns = ['isInteraction', 'is_interaction', 'interaction', 'label']
    
    # Find protein ID columns
    for col in columns:
        col_lower = col.lower()
        if any(pattern.lower() in col_lower for pattern in protein_a_patterns):
            protein_a_col = col
        elif any(pattern.lower() in col_lower for pattern in protein_b_patterns):
            protein_b_col = col
        elif any(pattern.lower() in col_lower for pattern in interaction_patterns):
            interaction_col = col
    
    # If we still can't find the columns, look for any that might contain protein IDs
    if protein_a_col is None or protein_b_col is None:
        # Check the first row to see if any column contains values that match keys in embeddings_dict
        first_row = data_df.iloc[0].to_dict()
        for col, val in first_row.items():
            if isinstance(val, str) and val in embeddings_dict:
                if protein_a_col is None:
                    protein_a_col = col
                elif protein_b_col is None and col != protein_a_col:
                    protein_b_col = col
    
    if protein_a_col is None or protein_b_col is None or interaction_col is None:
        print("Column detection failed. Please specify column names manually.")
        print("Available columns:", columns)
        raise ValueError("Could not detect required columns")
    
    print(f"Using columns: Protein A = '{protein_a_col}', Protein B = '{protein_b_col}', Interaction = '{interaction_col}'")
    return protein_a_col, protein_b_col, interaction_col


class ProteinPairDataset(Dataset):
    def __init__(self, pairs_df, embeddings_dict, protein_a_col=None, protein_b_col=None, interaction_col=None):
        """
        Dataset for protein pair interaction prediction
        
        Args:
            pairs_df: DataFrame with protein pair data
            embeddings_dict: Dict mapping uniprotID -> embedding tensor (seq_len, 960)
            protein_a_col: Column name for protein A IDs (auto-detected if None)
            protein_b_col: Column name for protein B IDs (auto-detected if None) 
            interaction_col: Column name for interaction labels (auto-detected if None)
        """
        self.pairs_df = pairs_df.reset_index(drop=True)
        self.embeddings_dict = embeddings_dict
        
        # Auto-detect column names if not provided
        if protein_a_col is None or protein_b_col is None or interaction_col is None:
            self.protein_a_col, self.protein_b_col, self.interaction_col = detect_column_names(pairs_df, embeddings_dict)
        else:
            self.protein_a_col = protein_a_col
            self.protein_b_col = protein_b_col
            self.interaction_col = interaction_col
        
        # Filter valid pairs
        valid_indices = []
        for idx in range(len(self.pairs_df)):
            row = self.pairs_df.iloc[idx]
            if (row[self.protein_a_col] in self.embeddings_dict and 
                row[self.protein_b_col] in self.embeddings_dict):
                valid_indices.append(idx)
        
        self.valid_indices = valid_indices
        print(f"Dataset: {len(valid_indices)} valid pairs out of {len(self.pairs_df)} total pairs")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        data_idx = self.valid_indices[idx]
        row = self.pairs_df.iloc[data_idx]
        
        # Get embeddings
        emb_a = self.embeddings_dict[row[self.protein_a_col]]
        emb_b = self.embeddings_dict[row[self.protein_b_col]]
        
        # Convert to tensors if needed
        if not isinstance(emb_a, torch.Tensor):
            emb_a = torch.from_numpy(emb_a).float()
        if not isinstance(emb_b, torch.Tensor):
            emb_b = torch.from_numpy(emb_b).float()
        
        # Get interaction label
        interaction = int(row[self.interaction_col])
        
        return {
            'emb_a': emb_a,           # (seq_len_a, 960)
            'emb_b': emb_b,           # (seq_len_b, 960)
            'interaction': interaction,
            'id_a': row[self.protein_a_col],
            'id_b': row[self.protein_b_col]
        }


def collate_fn(batch):
    """
    Collate function for protein pair batches with padding
    """
    # Extract components
    embs_a = [item['emb_a'] for item in batch]
    embs_b = [item['emb_b'] for item in batch]
    interactions = torch.tensor([item['interaction'] for item in batch], dtype=torch.long)
    
    # Pad sequences
    max_len_a = max(emb.shape[0] for emb in embs_a)
    max_len_b = max(emb.shape[0] for emb in embs_b)
    
    # Create padded tensors and length masks
    batch_size = len(batch)
    padded_a = torch.zeros(batch_size, max_len_a, 960)
    padded_b = torch.zeros(batch_size, max_len_b, 960)
    lengths_a = torch.zeros(batch_size, dtype=torch.long)
    lengths_b = torch.zeros(batch_size, dtype=torch.long)
    
    for i, (emb_a, emb_b) in enumerate(zip(embs_a, embs_b)):
        len_a, len_b = emb_a.shape[0], emb_b.shape[0]
        padded_a[i, :len_a] = emb_a
        padded_b[i, :len_b] = emb_b
        lengths_a[i] = len_a
        lengths_b[i] = len_b
    
    return padded_a, padded_b, lengths_a, lengths_b, interactions


class TransformerEnhancedProteinClassifier(nn.Module):
    """
    Enhanced version of SimplifiedProteinClassifier with transformer layers
    Based on the proven architecture but with added attention mechanism
    """
    def __init__(self, input_dim=960, hidden_dim=256, num_transformer_layers=2, 
                 num_heads=8, dropout=0.3):
        super().__init__()
        
        # Initial projection to hidden dimension
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Transformer encoder layers for sequence modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True  # Pre-norm for better training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_transformer_layers,
            norm=nn.LayerNorm(hidden_dim)
        )
        
        # Keep the proven protein encoder structure after transformer
        self.protein_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Keep the proven interaction layer exactly the same
        self.interaction_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Better weight initialization
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
        """
        device = emb_a.device
        
        # Project to hidden dimension
        emb_a_proj = self.input_projection(emb_a)  # (B, L_a, hidden_dim)
        emb_b_proj = self.input_projection(emb_b)  # (B, L_b, hidden_dim)
        
        # Create attention masks (True for padding positions)
        max_len_a = emb_a.size(1)
        max_len_b = emb_b.size(1)
        
        mask_a = torch.arange(max_len_a, device=device).unsqueeze(0) >= lengths_a.unsqueeze(1)
        mask_b = torch.arange(max_len_b, device=device).unsqueeze(0) >= lengths_b.unsqueeze(1)
        
        # Apply transformer encoder with attention masks
        emb_a_transformed = self.transformer_encoder(emb_a_proj, src_key_padding_mask=mask_a)
        emb_b_transformed = self.transformer_encoder(emb_b_proj, src_key_padding_mask=mask_b)
        
        # Average pooling with mask (same as proven approach)
        mask_a_float = ~mask_a  # Convert to actual sequence mask (False for padding)
        mask_b_float = ~mask_b
        
        emb_a_avg = (emb_a_transformed * mask_a_float.unsqueeze(-1).float()).sum(dim=1) / lengths_a.unsqueeze(-1).float()
        emb_b_avg = (emb_b_transformed * mask_b_float.unsqueeze(-1).float()).sum(dim=1) / lengths_b.unsqueeze(-1).float()
        
        # Use the proven encoder and interaction layers
        enc_a = self.protein_encoder(emb_a_avg)  # (B, hidden_dim)
        enc_b = self.protein_encoder(emb_b_avg)  # (B, hidden_dim)
        
        # Combine and predict interaction (exactly as before)
        combined = torch.cat([enc_a, enc_b], dim=-1)  # (B, 2*hidden_dim)
        logits = self.interaction_layer(combined)  # (B, 1)
        
        return logits


class SimplifiedProteinClassifier(nn.Module):
    """
    Proven simplified model architecture - keep as fallback
    """
    def __init__(self, input_dim=960, hidden_dim=256, dropout=0.3):
        super().__init__()
        
        # Simple encoder for each protein
        self.protein_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Simple interaction layer
        self.interaction_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Better weight initialization
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
        """
        # Simple average pooling for variable length sequences
        device = emb_a.device
        
        # Create masks for averaging
        mask_a = torch.arange(emb_a.size(1), device=device).unsqueeze(0) < lengths_a.unsqueeze(1)
        mask_b = torch.arange(emb_b.size(1), device=device).unsqueeze(0) < lengths_b.unsqueeze(1)
        
        # Average pooling with mask
        emb_a_avg = (emb_a * mask_a.unsqueeze(-1).float()).sum(dim=1) / lengths_a.unsqueeze(-1).float()
        emb_b_avg = (emb_b * mask_b.unsqueeze(-1).float()).sum(dim=1) / lengths_b.unsqueeze(-1).float()
        
        # Encode proteins
        enc_a = self.protein_encoder(emb_a_avg)  # (B, hidden_dim)
        enc_b = self.protein_encoder(emb_b_avg)  # (B, hidden_dim)
        
        # Combine and predict interaction
        combined = torch.cat([enc_a, enc_b], dim=-1)  # (B, 2*hidden_dim)
        logits = self.interaction_layer(combined)  # (B, 1)
        
        return logits


def create_model(model_type='enhanced', **kwargs):
    """
    Factory function to create models
    
    Args:
        model_type: 'simple' or 'enhanced'
        **kwargs: model parameters
    """
    if model_type == 'simple':
        return SimplifiedProteinClassifier(**kwargs)
    elif model_type == 'enhanced':
        return TransformerEnhancedProteinClassifier(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Training utilities
def train_model(model, train_loader, val_loader, num_epochs=20, lr=5e-3, device='cuda'):
    """
    Train a protein interaction model with the proven training setup
    
    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs to train
        lr: Learning rate
        device: Device to train on
    """
    print(f"Training model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    model = model.to(device)
    
    # Use the proven training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=lr, 
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1
    )
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_auc = 0
    history = []
    
    for epoch in range(1, num_epochs + 1):
        # Training phase
        model.train()
        train_losses = []
        train_preds = []
        train_probs = []
        train_labels = []
        
        for emb_a, emb_b, lengths_a, lengths_b, interactions in train_loader:
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
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Track metrics
            train_losses.append(loss.item())
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                train_preds.extend(preds.cpu().numpy())
                train_probs.extend(probs.cpu().numpy())
                train_labels.extend(interactions.cpu().numpy())
        
        # Validation phase
        model.eval()
        val_losses = []
        val_preds = []
        val_probs = []
        val_labels = []
        
        with torch.no_grad():
            for emb_a, emb_b, lengths_a, lengths_b, interactions in val_loader:
                emb_a = emb_a.to(device).float()
                emb_b = emb_b.to(device).float()
                lengths_a = lengths_a.to(device)
                lengths_b = lengths_b.to(device)
                interactions = interactions.to(device).float()
                
                logits = model(emb_a, emb_b, lengths_a, lengths_b)
                if logits.dim() > 1:
                    logits = logits.squeeze(-1)
                
                loss = criterion(logits, interactions)
                val_losses.append(loss.item())
                
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                val_preds.extend(preds.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())
                val_labels.extend(interactions.cpu().numpy())
        
        # Calculate metrics
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        train_acc = accuracy_score(train_labels, train_preds)
        val_acc = accuracy_score(val_labels, val_preds)
        train_auc = roc_auc_score(train_labels, train_probs)
        val_auc = roc_auc_score(val_labels, val_probs)
        val_f1 = f1_score(val_labels, val_preds)
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
        
        # Log
        epoch_log = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'train_auc': train_auc,
            'val_auc': val_auc,
            'val_f1': val_f1,
            'lr': scheduler.get_last_lr()[0]
        }
        history.append(epoch_log)
        
        if epoch % 5 == 0 or epoch <= 3:
            print(f'Epoch {epoch:2d}: Train AUC={train_auc:.4f}, Val AUC={val_auc:.4f}, '
                  f'Val F1={val_f1:.4f}, LR={scheduler.get_last_lr()[0]:.2e}')
    
    print(f'Training completed! Best validation AUC: {best_val_auc:.4f}')
    return history, best_val_auc


