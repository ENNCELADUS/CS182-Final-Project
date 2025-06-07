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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import pickle

# Import data loading functions from v4_1 (same as v5_2_train.py)
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'v4'))
from v4_1 import load_data, ProteinPairDataset

from v5_memory_friendly import create_ppi_classifier

class PPIDataset(Dataset):
    def __init__(self, ppi_data, protein_embeddings, max_len=1502):
        """
        Args:
            ppi_data: List of tuples (protein_a_id, protein_b_id, label)
            protein_embeddings: Dict mapping protein_id -> embedding array
            max_len: Maximum sequence length for padding/truncation
        """
        self.ppi_data = ppi_data
        self.protein_embeddings = protein_embeddings
        self.max_len = max_len
        
        # Filter out pairs where embeddings are missing
        self.valid_pairs = []
        for prot_a, prot_b, label in ppi_data:
            if prot_a in protein_embeddings and prot_b in protein_embeddings:
                self.valid_pairs.append((prot_a, prot_b, label))
        
        print(f"Valid pairs: {len(self.valid_pairs)} / {len(ppi_data)}")
        
    def __len__(self):
        return len(self.valid_pairs)
    
    def pad_or_truncate(self, embedding):
        """Pad or truncate embedding to max_len"""
        seq_len = embedding.shape[0]
        
        if seq_len < self.max_len:
            # Pad
            pad_size = (self.max_len - seq_len, 960)
            padding = np.zeros(pad_size, dtype=embedding.dtype)
            embedding = np.concatenate([embedding, padding], axis=0)
        else:
            # Truncate
            embedding = embedding[:self.max_len]
            
        return torch.from_numpy(embedding).float()
    
    def __getitem__(self, idx):
        prot_a, prot_b, label = self.valid_pairs[idx]
        
        # Get embeddings
        emb_a = self.protein_embeddings[prot_a]  # (seq_len, 960)
        emb_b = self.protein_embeddings[prot_b]  # (seq_len, 960)
        
        # Pad/truncate to consistent length
        emb_a = self.pad_or_truncate(emb_a)  # (max_len, 960)
        emb_b = self.pad_or_truncate(emb_b)  # (max_len, 960)
        
        return {
            "emb_a": emb_a,
            "emb_b": emb_b,
            "label": torch.tensor(label, dtype=torch.float32),
            "protein_a": prot_a,
            "protein_b": prot_b
        }

def collate_fn(batch):
    """Collate function for PPI data"""
    emb_a = torch.stack([item["emb_a"] for item in batch], dim=0)
    emb_b = torch.stack([item["emb_b"] for item in batch], dim=0)
    labels = torch.stack([item["label"] for item in batch], dim=0)
    
    return {
        "emb_a": emb_a,
        "emb_b": emb_b,
        "labels": labels
    }

def load_ppi_data(ppi_file_path):
    """
    Load PPI data from file
    Expected format: CSV with columns [protein_a, protein_b, label]
    """
    try:
        if ppi_file_path.endswith('.csv'):
            df = pd.read_csv(ppi_file_path)
        elif ppi_file_path.endswith('.pkl'):
            df = pd.read_pickle(ppi_file_path)
        else:
            raise ValueError("Unsupported file format. Use .csv or .pkl")
        
        # Convert to list of tuples
        ppi_data = [(row['protein_a'], row['protein_b'], int(row['label'])) 
                   for _, row in df.iterrows()]
        
        print(f"Loaded {len(ppi_data)} PPI pairs")
        print(f"Positive pairs: {sum(1 for _, _, label in ppi_data if label == 1)}")
        print(f"Negative pairs: {sum(1 for _, _, label in ppi_data if label == 0)}")
        
        return ppi_data
    
    except Exception as e:
        print(f"Error loading PPI data: {e}")
        # Create dummy data for testing
        print("Creating dummy PPI data for testing...")
        return create_dummy_ppi_data()

def create_dummy_ppi_data(n_pairs=1000):
    """Create dummy PPI data for testing"""
    # This would be replaced with actual protein IDs from your dataset
    protein_ids = [f"protein_{i}" for i in range(100)]
    
    ppi_data = []
    for i in range(n_pairs):
        prot_a = np.random.choice(protein_ids)
        prot_b = np.random.choice(protein_ids)
        # Ensure no self-interactions
        while prot_a == prot_b:
            prot_b = np.random.choice(protein_ids)
        
        label = np.random.randint(0, 2)  # Random binary label
        ppi_data.append((prot_a, prot_b, label))
    
    return ppi_data

def create_dummy_embeddings(protein_ids, max_len=1502):
    """Create dummy embeddings for testing"""
    embeddings = {}
    for protein_id in protein_ids:
        # Random sequence length between 50 and max_len
        seq_len = np.random.randint(50, max_len + 1)
        # Random embedding
        embedding = np.random.randn(seq_len, 960).astype(np.float32)
        embeddings[protein_id] = embedding
    
    return embeddings

def compute_metrics(y_true, y_pred, y_prob):
    """Compute classification metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    }
    return metrics

def evaluate_model(model, dataloader, device):
    """Evaluate model on validation/test set"""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            emb_a = batch["emb_a"].to(device)
            emb_b = batch["emb_b"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            logits = model(emb_a, emb_b)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            # Compute loss
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            total_loss += loss.item()
            
            # Collect predictions
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute metrics
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics

def train_ppi_classifier(mae_checkpoint_path, 
                        epochs=20,
                        batch_size=8,
                        learning_rate=1e-4):
    """
    Train PPI classifier with pre-trained MAE
    """
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data using the same function as v5_2_train.py
    print("\n📊 Loading data...")
    train_data, val_data, test1_data, test2_data, protein_embeddings = load_data()
    
    # Create datasets using the same classes as v5_2_train.py
    train_dataset = ProteinPairDataset(train_data, protein_embeddings)
    val_dataset = ProteinPairDataset(val_data, protein_embeddings)
    test1_dataset = ProteinPairDataset(test1_data, protein_embeddings)
    test2_dataset = ProteinPairDataset(test2_data, protein_embeddings)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test1 samples: {len(test1_dataset)}")
    print(f"Test2 samples: {len(test2_dataset)}")
    
    # Create collate function compatible with v5_memory_friendly (no lengths needed)
    def collate_fn_v5_compatible(batch):
        """Collate function compatible with v5_memory_friendly model"""
        embs_a = [item['emb_a'] for item in batch]
        embs_b = [item['emb_b'] for item in batch]
        interactions = torch.tensor([item['interaction'] for item in batch], dtype=torch.float32)
        
        # Pad sequences to same length within batch
        max_len_a = max(emb.shape[0] for emb in embs_a)
        max_len_b = max(emb.shape[0] for emb in embs_b)
        
        # Create padded tensors
        batch_size = len(batch)
        padded_a = torch.zeros(batch_size, max_len_a, 960, dtype=torch.float32)
        padded_b = torch.zeros(batch_size, max_len_b, 960, dtype=torch.float32)
        
        for i, (emb_a, emb_b) in enumerate(zip(embs_a, embs_b)):
            len_a, len_b = emb_a.shape[0], emb_b.shape[0]
            emb_a = emb_a.float() if emb_a.dtype != torch.float32 else emb_a
            emb_b = emb_b.float() if emb_b.dtype != torch.float32 else emb_b
            padded_a[i, :len_a] = emb_a
            padded_b[i, :len_b] = emb_b
        
        return {
            "emb_a": padded_a,
            "emb_b": padded_b,
            "labels": interactions
        }
    
    # Create dataloaders using the same data as v5_2_train.py
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=1, collate_fn=collate_fn_v5_compatible)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                          num_workers=1, collate_fn=collate_fn_v5_compatible)
    test1_loader = DataLoader(test1_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=1, collate_fn=collate_fn_v5_compatible)
    test2_loader = DataLoader(test2_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=1, collate_fn=collate_fn_v5_compatible)
    
    # Create model with pre-trained MAE
    model = create_ppi_classifier(
        mae_checkpoint_path=mae_checkpoint_path,
        freeze_encoder=True,  # Freeze the MAE encoder
        use_lora=False
    ).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Summary:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    # Setup optimizer (only for trainable parameters)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                lr=learning_rate, weight_decay=0.01)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # Setup logging
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_path = f'logs/ppi_train_{ts}.json'
    best_path = f'models/ppi_best_{ts}.pth'
    
    best_val_auc = 0.0
    best_epoch = 0
    history = []
    
    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            emb_a = batch["emb_a"].to(device, non_blocking=True)
            emb_b = batch["emb_b"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(emb_a, emb_b)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            
            if batch_idx % 20 == 0:
                print(f'Epoch {epoch}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        scheduler.step()
        
        # Validation
        val_metrics = evaluate_model(model, val_loader, device)
        train_loss = np.mean(train_losses)
        
        print(f'Epoch {epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_metrics["loss"]:.4f}')
        print(f'Val Metrics - Acc: {val_metrics["accuracy"]:.4f}, F1: {val_metrics["f1"]:.4f}, AUC: {val_metrics["auc"]:.4f}')
        
        # Save best model based on AUC
        is_best = val_metrics["auc"] > best_val_auc
        if is_best:
            best_val_auc = val_metrics["auc"]
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_auc': best_val_auc,
                'val_metrics': val_metrics
            }, best_path)
            print(f'>>> New best model saved! Val AUC: {best_val_auc:.4f}')
        
        # Log results
        log_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_metrics': val_metrics,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        history.append(log_entry)
        
        # Memory cleanup
        gc.collect()
        torch.cuda.empty_cache()
    
    # Final evaluation on test sets
    print("\n" + "="*50)
    print("FINAL EVALUATION ON TEST SETS")
    print("="*50)
    
    # Load best model
    checkpoint = torch.load(best_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on both test sets
    test1_metrics = evaluate_model(model, test1_loader, device)
    test2_metrics = evaluate_model(model, test2_loader, device)
    
    print(f"Test1 Metrics:")
    for metric, value in test1_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\nTest2 Metrics:")
    for metric, value in test2_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save final results
    final_results = {
        'best_val_auc': best_val_auc,
        'best_epoch': best_epoch,
        'test1_metrics': test1_metrics,
        'test2_metrics': test2_metrics,
        'training_history': history
    }
    
    with open(f'logs/ppi_final_results_{ts}.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Plot training curves
    train_losses = [entry['train_loss'] for entry in history]
    val_losses = [entry['val_metrics']['loss'] for entry in history]
    val_f1s = [entry['val_metrics']['f1'] for entry in history]
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(val_f1s, label='Val F1', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Validation F1 Score')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot([entry['val_metrics']['accuracy'] for entry in history], label='Accuracy')
    plt.plot([entry['val_metrics']['precision'] for entry in history], label='Precision')
    plt.plot([entry['val_metrics']['recall'] for entry in history], label='Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation Metrics')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'logs/ppi_training_curves_{ts}.png', dpi=150)
    plt.close()
    
    print(f"\nTraining completed! Best model saved to: {best_path}")
    print(f"Best validation AUC: {best_val_auc:.4f} (epoch {best_epoch})")
    print(f"Results saved to: logs/ppi_final_results_{ts}.json")
    
    return model, history

if __name__ == "__main__":
    # Configuration - using the compatible v5 MAE model
    MAE_CHECKPOINT_PATH = "models/v5_compatible_mae_best_20250607-012709.pth"  # Compatible v5 MAE
    
    # Check if MAE checkpoint exists
    if not os.path.exists(MAE_CHECKPOINT_PATH):
        print(f"❌ Error: Compatible MAE checkpoint not found: {MAE_CHECKPOINT_PATH}")
        print("Please train the compatible MAE first:")
        print("Run: python src/v5/v5_mae_compatible_train.py")
        exit(1)
    
    # Start training
    print("🧬 Starting PPI Classifier Training with v5_memory_friendly...")
    print(f"📁 MAE checkpoint: {MAE_CHECKPOINT_PATH}")
    print("📊 Using same data loading as v5_2_train.py")
    
    model, history = train_ppi_classifier(
        mae_checkpoint_path=MAE_CHECKPOINT_PATH,
        epochs=30,
        batch_size=8,
        learning_rate=1e-4
    ) 