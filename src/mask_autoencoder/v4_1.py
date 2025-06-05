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


# Reuse data loading functions from v4.py
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


class SimplifiedProteinEncoder(nn.Module):
    """
    Drastically simplified protein encoder - targeting 200-500K parameters total
    """
    def __init__(self, input_dim=960, embed_dim=128, max_length=512):
        super().__init__()
        
        self.max_length = max_length
        
        # Simple projection to smaller dimension
        self.input_proj = nn.Linear(input_dim, embed_dim)
        
        # Single lightweight self-attention layer
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads=4, dropout=0.1, batch_first=True
        )
        
        # Simple normalization and feedforward
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
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
        
        # Single self-attention layer with residual connection
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + attn_out)
        
        # Simple feedforward with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        # Simple average pooling with masking
        # Create inverse mask for pooling (1 for valid positions, 0 for padding)
        valid_mask = ~mask  # (B, L)
        valid_counts = valid_mask.sum(dim=1, keepdim=True).float()  # (B, 1)
        
        # Masked average pooling
        x_masked = x * valid_mask.unsqueeze(-1).float()  # (B, L, embed_dim)
        pooled = x_masked.sum(dim=1) / torch.clamp(valid_counts, min=1.0)  # (B, embed_dim)
        
        # Final projection
        output = self.pool_proj(pooled)  # (B, embed_dim)
        
        return output


class SimpleInteractionLayer(nn.Module):
    """
    Simple concatenation-based interaction (no cross-attention)
    """
    def __init__(self, embed_dim=128):
        super().__init__()
        
        # Simple processing of concatenated features
        self.interaction_net = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, emb_a, emb_b):
        """
        Args:
            emb_a: (B, embed_dim) protein A representation
            emb_b: (B, embed_dim) protein B representation
        Returns:
            (B, embed_dim) interaction representation
        """
        # Simple concatenation
        combined = torch.cat([emb_a, emb_b], dim=-1)  # (B, 2*embed_dim)
        
        # Process interaction
        interaction = self.interaction_net(combined)  # (B, embed_dim)
        
        return interaction


class SimpleMLPDecoder(nn.Module):
    """
    Simple MLP decoder for classification
    """
    def __init__(self, input_dim=128, hidden_dim=64):
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Args:
            x: (B, input_dim) interaction representation
        Returns:
            (B, 1) classification logits
        """
        return self.decoder(x)


class SimplifiedProteinInteractionClassifier(nn.Module):
    """
    Simplified protein-protein interaction classifier (v4.1)
    
    Key improvements based on debug analysis:
    1. Removed complex cross-attention → Simple concatenation
    2. Drastically reduced parameters → Target 200-500K range
    3. Simplified architecture → Single attention layer
    4. Better initialization and normalization
    5. Higher learning rates supported (5e-3 to 1e-2)
    
    Architecture:
    1. Protein A/B → SimplifiedProteinEncoder → Protein Representations
    2. Simple concatenation and interaction processing  
    3. Simple MLP decoder for binary classification
    """
    def __init__(self, embed_dim=128, max_length=512):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Shared simplified protein encoder
        self.protein_encoder = SimplifiedProteinEncoder(
            input_dim=960,
            embed_dim=embed_dim,
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
        # Encode both proteins with shared encoder
        repr_a = self.protein_encoder(emb_a, lengths_a)  # (B, embed_dim)
        repr_b = self.protein_encoder(emb_b, lengths_b)  # (B, embed_dim)
        
        # Simple interaction processing
        interaction_repr = self.interaction_layer(repr_a, repr_b)  # (B, embed_dim)
        
        # Classification
        logits = self.decoder(interaction_repr)  # (B, 1)
        
        return logits


def train_model(train_data, val_data, embeddings_dict, 
                embed_dim=128, max_length=512, epochs=50, batch_size=32, 
                learning_rate=5e-3, weight_decay=0.01, use_scheduler=True,
                save_every_epochs=10, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Train the simplified protein interaction prediction model (v4.1)
    """
    # Create datasets
    train_dataset = ProteinPairDataset(train_data, embeddings_dict)
    val_dataset = ProteinPairDataset(val_data, embeddings_dict)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    # Calculate training statistics
    num_train_batches = len(train_loader)
    num_val_batches = len(val_loader)
    total_train_steps = num_train_batches * epochs
    
    print(f"📊 TRAINING STATISTICS (v4.1):")
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")
    print(f"Batch size: {batch_size}")
    print(f"Training batches per epoch: {num_train_batches}")
    print(f"Total epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Use scheduler: {use_scheduler}")
    print(f"Target parameters: 200-500K")
    
    # Create simplified model
    model = SimplifiedProteinInteractionClassifier(
        embed_dim=embed_dim,
        max_length=max_length
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} (Target: 200-500K)")
    
    if num_params > 600000:
        print(f"⚠️ Warning: Model has {num_params:,} parameters, may be too large")
    elif num_params < 150000:
        print(f"⚠️ Warning: Model has {num_params:,} parameters, may be too small")
    else:
        print(f"✅ Model size in target range")
    
    # Use DataParallel if multiple GPUs available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    # Optimizer with higher learning rate
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler - OneCycle for better training
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=learning_rate,
            epochs=epochs,
            steps_per_epoch=num_train_batches,
            pct_start=0.1  # 10% warmup
        )
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Training setup
    best_val_auc = 0
    best_val_loss = float('inf')
    history = []
    
    # Create log directory
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_path = f'logs/ppi_v4_1_{ts}.json'
    best_path = f'models/ppi_v4_1_best_{ts}.pth'
    checkpoint_dir = f'models/checkpoints_v4_1_{ts}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"Best model will be saved to: {best_path}")
    print(f"Checkpoints will be saved to: {checkpoint_dir}/")
    
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_losses = []
        train_preds = []
        train_probs = []
        train_labels = []
        
        for batch_idx, (emb_a, emb_b, lengths_a, lengths_b, interactions) in enumerate(train_loader):
            # Move to device
            emb_a = emb_a.to(device).float()
            emb_b = emb_b.to(device).float()
            lengths_a = lengths_a.to(device)
            lengths_b = lengths_b.to(device)
            interactions = interactions.to(device).float()
            
            # Forward pass
            logits = model(emb_a, emb_b, lengths_a, lengths_b)
            
            # Fix dimension mismatch
            if logits.dim() > 1:
                logits = logits.squeeze(-1)
            if interactions.dim() == 0:
                interactions = interactions.unsqueeze(0)
                
            loss = criterion(logits, interactions)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            if scheduler:
                scheduler.step()
            
            # Track metrics
            train_losses.append(loss.item())
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                train_preds.extend(preds.cpu().numpy())
                train_probs.extend(probs.cpu().numpy())
                train_labels.extend(interactions.cpu().numpy())
            
            if batch_idx % 50 == 0:
                current_lr = scheduler.get_last_lr()[0] if scheduler else learning_rate
                print(f'Epoch {epoch}/{epochs} Batch {batch_idx}/{num_train_batches} '
                      f'Loss: {loss.item():.4f}, LR: {current_lr:.6f}')
        
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
                if interactions.dim() == 0:
                    interactions = interactions.unsqueeze(0)
                    
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
        val_f1 = f1_score(val_labels, val_preds)
        
        # Calculate AUC scores
        train_auc = roc_auc_score(train_labels, train_probs) if len(set(train_labels)) > 1 else 0
        val_auc = roc_auc_score(val_labels, val_probs) if len(set(val_labels)) > 1 else 0
        
        # Save best model based on AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_auc': val_auc,
                'val_acc': val_acc,
                'val_loss': val_loss,
                'train_auc': train_auc,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'config': {
                    'embed_dim': embed_dim,
                    'max_length': max_length,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'weight_decay': weight_decay,
                    'use_scheduler': use_scheduler,
                    'num_parameters': num_params
                },
                'history': history
            }, best_path)
            print(f'>>> Saved BEST model: Epoch {epoch}, Val AUC: {val_auc:.4f}, Val Acc: {val_acc:.4f}')
        
        # Save regular checkpoints
        if epoch % save_every_epochs == 0 or epoch == epochs:
            checkpoint_path = f'{checkpoint_dir}/checkpoint_epoch_{epoch:03d}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_auc': val_auc,
                'val_acc': val_acc,
                'val_loss': val_loss,
                'train_auc': train_auc,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'config': {
                    'embed_dim': embed_dim,
                    'max_length': max_length,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'weight_decay': weight_decay,
                    'use_scheduler': use_scheduler,
                    'num_parameters': num_params
                },
                'history': history
            }, checkpoint_path)
            print(f'>>> Saved checkpoint: {checkpoint_path}')
        
        # Log progress
        epoch_log = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'train_auc': train_auc,
            'val_auc': val_auc,
            'val_f1': val_f1,
            'lr': scheduler.get_last_lr()[0] if scheduler else learning_rate,
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        history.append(epoch_log)
        
        with open(log_path, 'a') as f:
            f.write(json.dumps(epoch_log) + '\n')
        
        print(f'Epoch {epoch:3d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, '
              f'Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, Train AUC={train_auc:.4f}, '
              f'Val AUC={val_auc:.4f}, Val F1={val_f1:.4f}')
        
        # Memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f'\n✅ Training completed! Best validation AUC: {best_val_auc:.4f}')
    print(f'📁 Best model saved to: {best_path}')
    print(f'📁 Model parameters: {num_params:,}')
    
    return history, best_path


def evaluate_model(model_path, test_data, embeddings_dict, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Evaluate the trained simplified model on test data
    """
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    model = SimplifiedProteinInteractionClassifier(
        embed_dim=config.get('embed_dim', 128),
        max_length=config.get('max_length', 512)
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create test dataset and loader
    test_dataset = ProteinPairDataset(test_data, embeddings_dict)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=2
    )
    
    # Evaluation
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for emb_a, emb_b, lengths_a, lengths_b, interactions in test_loader:
            emb_a = emb_a.to(device).float()
            emb_b = emb_b.to(device).float()
            lengths_a = lengths_a.to(device)
            lengths_b = lengths_b.to(device)
            
            logits = model(emb_a, emb_b, lengths_a, lengths_b)
            
            if logits.dim() > 1:
                logits = logits.squeeze(-1)
                
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(interactions.numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    
    print(f"\nTest Results (v4.1):")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {auc:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': all_labels
    }


if __name__ == '__main__':
    # DEFAULT CONFIGURATION FOR v4.1
    print("Simplified Protein-Protein Interaction Prediction v4.1")
    print("=" * 60)
    print("Key Improvements:")
    print("✓ Removed complex cross-attention → Simple concatenation")
    print("✓ Reduced parameters → Target 200-500K range")
    print("✓ Higher learning rates → 5e-3 default")
    print("✓ Simplified architecture → Single attention layer")
    print("✓ Better initialization and stability")
    print("=" * 60)
    
    try:
        # Load data
        train_data, cv_data, test1_data, test2_data, protein_embeddings = load_data()
        
        print(f"\nData loaded successfully:")
        print(f"Training data: {len(train_data)} pairs")
        print(f"Validation data: {len(cv_data)} pairs") 
        print(f"Test1 data: {len(test1_data)} pairs")
        print(f"Test2 data: {len(test2_data)} pairs")
        print(f"Protein embeddings: {len(protein_embeddings)} proteins")
        
        # DEFAULT TRAINING CONFIGURATION
        default_config = {
            'embed_dim': 128,          # Smaller embedding dimension
            'max_length': 512,         # Truncate long sequences
            'epochs': 30,              # Fewer epochs
            'batch_size': 32,          # Larger batch size for stability
            'learning_rate': 5e-3,     # Higher learning rate as recommended
            'weight_decay': 0.01,      # L2 regularization
            'use_scheduler': True      # OneCycle scheduler
        }
        
        print(f"\nTraining with DEFAULT configuration:")
        for key, value in default_config.items():
            print(f"  {key}: {value}")
        
        # Train model with default configuration
        history, model_path = train_model(
            train_data, cv_data, protein_embeddings,
            **default_config
        )
        
        # Load best model and get validation AUC
        checkpoint = torch.load(model_path, map_location='cpu')
        best_auc = checkpoint['val_auc']
        num_params = checkpoint['config']['num_parameters']
        
        print(f"\n{'='*60}")
        print(f"TRAINING SUMMARY (v4.1)")
        print(f"{'='*60}")
        print(f"Model parameters: {num_params:,}")
        print(f"Best validation AUC: {best_auc:.4f}")
        print(f"Model path: {model_path}")
        
        # Evaluate on test sets
        print(f"\nEvaluating on test sets...")
        
        try:
            print(f"\nTest1 Results:")
            results_test1 = evaluate_model(model_path, test1_data, protein_embeddings)
        except Exception as e:
            print(f"Error evaluating on test1: {str(e)}")
        
        try:
            print(f"\nTest2 Results:")
            results_test2 = evaluate_model(model_path, test2_data, protein_embeddings)
        except Exception as e:
            print(f"Error evaluating on test2: {str(e)}")
        
        # Save summary
        summary = {
            'model_version': 'v4.1',
            'improvements': [
                'Removed complex cross-attention',
                'Reduced parameters to 200-500K range',
                'Higher learning rates (5e-3)',
                'Simplified architecture',
                'Better initialization'
            ],
            'default_config': default_config,
            'results': {
                'num_parameters': num_params,
                'best_val_auc': best_auc,
                'model_path': model_path,
                'test1_auc': results_test1.get('auc', 0) if 'results_test1' in locals() else 0,
                'test2_auc': results_test2.get('auc', 0) if 'results_test2' in locals() else 0
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        summary_path = f'logs/v4_1_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSummary saved to: {summary_path}")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc() 