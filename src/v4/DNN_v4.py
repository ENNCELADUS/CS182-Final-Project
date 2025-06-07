#!/usr/bin/env python3
"""
Quick fixes for the protein interaction model based on diagnostic results.
This script implements the most likely solutions for the random performance issue.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

# Import from the existing v4 module
from v4 import load_data, ProteinPairDataset, collate_fn, ProteinInteractionClassifier

class SimplifiedProteinClassifier(nn.Module):
    """
    Simplified model for testing - much smaller to avoid memory issues
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

def test_simple_model():
    # Load data
    train_data, val_data, _, _, protein_embeddings = load_data()
    
    # Create datasets
    train_dataset = ProteinPairDataset(train_data, protein_embeddings)
    val_dataset = ProteinPairDataset(val_data, protein_embeddings)
    
    # Small batch size for testing
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create simple model
    model = SimplifiedProteinClassifier(hidden_dim=128).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Use higher learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)  # Higher LR
    criterion = nn.BCEWithLogitsLoss()
    
    print("\nTesting training for 10 steps...")
    model.train()
    
    for i, (emb_a, emb_b, lengths_a, lengths_b, interactions) in enumerate(train_loader):
        if i >= 10:
            break
            
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
        optimizer.step()
        
        if i % 3 == 0:
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            acc = (preds == interactions).float().mean()
            print(f"  Step {i}: Loss={loss.item():.4f}, Acc={acc.item():.4f}")
    
    return model

def train_fixed_model():
    # Load data
    train_data, val_data, _, _, protein_embeddings = load_data()
    
    # Create datasets
    train_dataset = ProteinPairDataset(train_data, protein_embeddings)
    val_dataset = ProteinPairDataset(val_data, protein_embeddings)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32,  # Larger batch size for stability
        shuffle=True, 
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=2
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Use simplified model that fits in memory
    model = SimplifiedProteinClassifier(hidden_dim=256).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # FIXED TRAINING SETUP
    # 1. Higher learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.01)
    
    # 2. Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=5e-3, 
        epochs=20,
        steps_per_epoch=len(train_loader),
        pct_start=0.1
    )
    
    # 3. Standard BCE loss (data is balanced)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    best_val_auc = 0
    history = []
    
    for epoch in range(1, 21):  # 20 epochs
        # Training
        model.train()
        train_losses = []
        train_preds = []
        train_probs = []
        train_labels = []
        
        for batch_idx, (emb_a, emb_b, lengths_a, lengths_b, interactions) in enumerate(train_loader):
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
            
            # Gradient clipping
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
        
        # Validation
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
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_auc': val_auc,
                'epoch': epoch
            }, 'models/DNN_v4.pth')
        
        # Log
        epoch_log = {
            'epoch': 94,
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
        
        print(f'Epoch {epoch:2d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, '
              f'Train AUC={train_auc:.4f}, Val AUC={val_auc:.4f}, Val F1={val_f1:.4f}, '
              f'LR={scheduler.get_last_lr()[0]:.2e}')
    
    print(f'\n Training completed! Best validation AUC: {best_val_auc:.4f}')
    return history, best_val_auc

def main():
    # Ensure directories exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Test simple model first
    test_simple_model()
    
    # Train fixed model
    history, best_auc = train_fixed_model()
    
    # Save results
    with open(f'logs/fixed_model_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
        json.dump({
            'best_val_auc': best_auc,
            'history': history,
        }, f, indent=2)

if __name__ == "__main__":
    main()