#!/usr/bin/env python3
"""
Quick real-data training script: small batch, few epochs to check validation AUC.
"""
import torch
from torch.utils.data import DataLoader
from v4_1 import load_data, ProteinPairDataset, collate_fn, TransformerEnhancedProteinClassifier, train_model

def main():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load real data
    train_df, val_df, test1_df, test2_df, embeddings = load_data()
    
    # Create datasets
    train_dataset = ProteinPairDataset(train_df, embeddings)
    val_dataset   = ProteinPairDataset(val_df, embeddings)
    
    # Small batch for fast iteration
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Batch size: {batch_size}")
    
    # Initialize model with recommended settings
    model = TransformerEnhancedProteinClassifier(
        input_dim=960,
        hidden_dim=256,
        num_transformer_layers=2,
        num_heads=8,
        dropout=0.1
    ).to(device)
    
    # Train for a few epochs with lower LR
    num_epochs = 3
    lr = 3e-4
    print(f"Training for {num_epochs} epochs with lr={lr}")
    history, best_val_auc = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        lr=lr,
        device=device,
        debug_mode=False
    )
    
    print(f"\n✅ Quick training completed. Best validation AUC: {best_val_auc:.4f}")

if __name__ == '__main__':
    main() 