#!/usr/bin/env python3
"""
Training script for V5.2 PPI Classifier
Uses pretrained v2 MAE with v5 downstream components
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, average_precision_score, 
                           roc_curve, precision_recall_curve)

# Import data loading functions
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'v4'))
from v4_1 import load_data, ProteinPairDataset

# Import v5.2 model
from v5_2 import PPIClassifierV52, create_ppi_classifier_v52, count_parameters

def collate_fn_v52(batch):
    """
    Collate function for v5.2 PPI classifier
    Returns embeddings with length information for v2 MAE compatibility
    """
    # Extract components
    embs_a = [item['emb_a'] for item in batch]
    embs_b = [item['emb_b'] for item in batch]
    interactions = torch.tensor([item['interaction'] for item in batch], dtype=torch.float)
    
    # Get original lengths before padding
    lengths_a = torch.tensor([emb.shape[0] for emb in embs_a])
    lengths_b = torch.tensor([emb.shape[0] for emb in embs_b])
    
    # Pad sequences to same length within batch
    max_len_a = max(emb.shape[0] for emb in embs_a)
    max_len_b = max(emb.shape[0] for emb in embs_b)
    
    # Create padded tensors
    batch_size = len(batch)
    padded_a = torch.zeros(batch_size, max_len_a, 960)
    padded_b = torch.zeros(batch_size, max_len_b, 960)
    
    for i, (emb_a, emb_b) in enumerate(zip(embs_a, embs_b)):
        len_a, len_b = emb_a.shape[0], emb_b.shape[0]
        padded_a[i, :len_a] = emb_a
        padded_b[i, :len_b] = emb_b
    
    return padded_a, padded_b, lengths_a, lengths_b, interactions

def train_epoch(model, train_loader, optimizer, criterion, device, scheduler=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (emb_a, emb_b, lengths_a, lengths_b, interactions) in enumerate(pbar):
        try:
            emb_a = emb_a.to(device).float()
            emb_b = emb_b.to(device).float()
            lengths_a = lengths_a.to(device)
            lengths_b = lengths_b.to(device)
            interactions = interactions.to(device).float()
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(emb_a, emb_b, lengths_a, lengths_b)
            
            # Calculate loss
            loss = criterion(logits, interactions)
            
            # Check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss at batch {batch_idx}: {loss.item()}")
                continue
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            if scheduler:
                scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(interactions.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Clear CUDA cache periodically
            if batch_idx % 100 == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            if "CUDA" in str(e):
                print(f"CUDA error at batch {batch_idx}: {e}")
                print("Attempting to recover by clearing CUDA cache...")
                torch.cuda.empty_cache()
                continue
            else:
                raise e
        except Exception as e:
            print(f"Unexpected error at batch {batch_idx}: {e}")
            continue
    
    # Calculate epoch metrics
    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Calculate AUC if we have both classes
    if len(set(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_probs)
        auprc = average_precision_score(all_labels, all_probs)
    else:
        auc = 0.5
        auprc = 0.5
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'auc': auc,
        'auprc': auprc,
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': all_labels
    }

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for emb_a, emb_b, lengths_a, lengths_b, interactions in pbar:
            emb_a = emb_a.to(device).float()
            emb_b = emb_b.to(device).float()
            lengths_a = lengths_a.to(device)
            lengths_b = lengths_b.to(device)
            interactions = interactions.to(device).float()
            
            # Forward pass
            logits = model(emb_a, emb_b, lengths_a, lengths_b)
            
            # Calculate loss
            loss = criterion(logits, interactions)
            total_loss += loss.item()
            
            # Track metrics
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(interactions.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Calculate epoch metrics
    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    
    # Calculate AUC if we have both classes
    if len(set(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_probs)
        auprc = average_precision_score(all_labels, all_probs)
    else:
        auc = 0.5
        auprc = 0.5
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'auc': auc,
        'auprc': auprc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': all_labels
    }

def evaluate_test_set(model, test_loader, device, test_name="Test"):
    """Evaluate model on test set"""
    model.eval()
    all_preds = []
    all_probs = []
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc=f"Evaluating {test_name}")
        for emb_a, emb_b, lengths_a, lengths_b, interactions in pbar:
            emb_a = emb_a.to(device).float()
            emb_b = emb_b.to(device).float()
            lengths_a = lengths_a.to(device)
            lengths_b = lengths_b.to(device)
            interactions = interactions.to(device).float()
            
            # Forward pass
            logits = model(emb_a, emb_b, lengths_a, lengths_b)
            
            # Track metrics
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            all_logits.extend(logits.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(interactions.cpu().numpy())
    
    # Calculate comprehensive metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    
    # Calculate AUC metrics
    if len(set(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_probs)
        auprc = average_precision_score(all_labels, all_probs)
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs)
    else:
        auc = 0.5
        auprc = 0.5
        fpr, tpr = [0, 1], [0, 1]
        precision_curve, recall_curve = [1, 0], [0, 1]
    
    results = {
        'test_name': test_name,
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc': auc,
        'auprc': auprc,
        'predictions': all_preds,
        'probabilities': all_probs,
        'logits': all_logits,
        'labels': all_labels,
        'fpr': fpr.tolist() if hasattr(fpr, 'tolist') else fpr,
        'tpr': tpr.tolist() if hasattr(tpr, 'tolist') else tpr,
        'precision_curve': precision_curve.tolist() if hasattr(precision_curve, 'tolist') else precision_curve,
        'recall_curve': recall_curve.tolist() if hasattr(recall_curve, 'tolist') else recall_curve
    }
    
    print(f"{test_name} Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  AUROC: {auc:.4f}")
    print(f"  AUPRC: {auprc:.4f}")
    
    return results

def save_checkpoint(model, optimizer, scheduler, epoch, train_metrics, val_metrics, 
                   config, save_path, is_best=False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'config': config,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    torch.save(checkpoint, save_path)
    if is_best:
        best_path = save_path.replace('.pth', '_best.pth')
        torch.save(checkpoint, best_path)
        print(f"💾 Saved best model: {best_path}")

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint.get('config', {}), checkpoint.get('val_metrics', {})

def plot_training_curves(history, save_path):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('V5.2 Training Progress (Pretrained v2 MAE)', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # AUC curves
    axes[0, 1].plot(epochs, history['train_auc'], 'b-', label='Train AUC')
    axes[0, 1].plot(epochs, history['val_auc'], 'r-', label='Val AUC')
    axes[0, 1].set_title('AUC Curves')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('AUC')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[1, 0].plot(epochs, history['train_acc'], 'b-', label='Train Accuracy')
    axes[1, 0].plot(epochs, history['val_acc'], 'r-', label='Val Accuracy')
    axes[1, 0].set_title('Accuracy Curves')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate
    if 'learning_rate' in history:
        axes[1, 1].plot(epochs, history['learning_rate'], 'g-')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No LR Schedule', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Learning Rate Schedule')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved training curves: {save_path}")

def train_model(config):
    """Main training function"""
    print("🧬 V5.2 PPI CLASSIFIER TRAINING (Pretrained v2 MAE)")
    print("=" * 60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
        torch.cuda.empty_cache()
    
    # Create output directories
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    # Load data
    print("\n📊 Loading data...")
    train_data, val_data, test1_data, test2_data, protein_embeddings = load_data()
    
    # Create datasets
    train_dataset = ProteinPairDataset(train_data, protein_embeddings)
    val_dataset = ProteinPairDataset(val_data, protein_embeddings)
    test1_dataset = ProteinPairDataset(test1_data, protein_embeddings)
    test2_dataset = ProteinPairDataset(test2_data, protein_embeddings)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test1 samples: {len(test1_dataset)}")
    print(f"Test2 samples: {len(test2_dataset)}")
    
    # Create data loaders
    print(f"Creating data loaders with batch_size={config['batch_size']}")
    
    dataloader_kwargs = {
        'batch_size': config['batch_size'],
        'collate_fn': collate_fn_v52,
        'num_workers': 0,
        'pin_memory': False,
        'persistent_workers': False
    }
    
    train_loader = DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **dataloader_kwargs)
    test1_loader = DataLoader(test1_dataset, shuffle=False, **dataloader_kwargs)
    test2_loader = DataLoader(test2_dataset, shuffle=False, **dataloader_kwargs)
    
    # Create model
    print(f"\n🔧 Creating V5.2 model...")
    model = create_ppi_classifier_v52(config['v2_mae_path'])
    model = model.to(device)
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    print(f"Trainable ratio: {trainable_params/total_params*100:.1f}%")
    
    # Setup optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 0.01),
        betas=(0.9, 0.999)
    )
    
    scheduler = None
    if config.get('use_scheduler', False):
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['learning_rate'],
            epochs=config['num_epochs'],
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            anneal_strategy='cos'
        )
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [], 'train_auc': [], 'val_auc': [],
        'train_acc': [], 'val_acc': [], 'learning_rate': []
    }
    
    best_val_auc = 0.0
    best_epoch = 0
    
    # Training loop
    print(f"\n🚀 Starting training for {config['num_epochs']} epochs...")
    print("Note: v2 MAE encoder is frozen, only training cross-attention and MLP head")
    
    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['num_epochs']}")
        print("-" * 40)
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, scheduler)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_auc'].append(train_metrics['auc'])
        history['val_auc'].append(val_metrics['auc'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_acc'].append(val_metrics['accuracy'])
        
        if scheduler:
            history['learning_rate'].append(scheduler.get_last_lr()[0])
        else:
            history['learning_rate'].append(config['learning_rate'])
        
        # Print metrics
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, AUC: {train_metrics['auc']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, AUC: {val_metrics['auc']:.4f}, F1: {val_metrics['f1']:.4f}")
        
        # Save checkpoint
        is_best = val_metrics['auc'] > best_val_auc
        if is_best:
            best_val_auc = val_metrics['auc']
            best_epoch = epoch
            print(f"🎉 New best validation AUC: {best_val_auc:.4f}")
        
        checkpoint_path = os.path.join(config['save_dir'], f"checkpoint_epoch_{epoch}.pth")
        save_checkpoint(model, optimizer, scheduler, epoch, train_metrics, val_metrics, 
                       config, checkpoint_path, is_best)
        
        # Early stopping
        if config.get('early_stopping', False) and epoch - best_epoch >= config.get('patience', 10):
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model for final evaluation
    best_checkpoint_path = os.path.join(config['save_dir'], f"checkpoint_epoch_{best_epoch}_best.pth")
    if os.path.exists(best_checkpoint_path):
        print(f"\n📥 Loading best model from epoch {best_epoch}")
        load_checkpoint(best_checkpoint_path, model)
    
    # Final evaluation on test sets
    print("\n🧪 Final evaluation on test sets...")
    test_results = {}
    
    test_results['Test1'] = evaluate_test_set(model, test1_loader, device, "Test1")
    test_results['Test2'] = evaluate_test_set(model, test2_loader, device, "Test2")
    
    # Save results
    results_dict = {
        'config': config,
        'history': history,
        'best_epoch': best_epoch,
        'best_val_auc': best_val_auc,
        'test_results': test_results,
        'model_parameters': {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': total_params - trainable_params
        }
    }
    
    # Save results JSON
    results_path = os.path.join(config['log_dir'], 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2, default=str)
    print(f"💾 Saved results: {results_path}")
    
    # Generate plots
    plot_path = os.path.join(config['log_dir'], 'training_curves.png')
    plot_training_curves(history, plot_path)
    
    # Print final summary
    print("\n" + "=" * 60)
    print("🎉 V5.2 TRAINING COMPLETED!")
    print("=" * 60)
    print(f"Best validation AUC: {best_val_auc:.4f} (epoch {best_epoch})")
    
    for test_name, test_result in test_results.items():
        print(f"{test_name} - AUROC: {test_result['auc']:.4f}, AUPRC: {test_result['auprc']:.4f}")
    
    print(f"\nModel saved to: {config['save_dir']}")
    print(f"Logs saved to: {config['log_dir']}")
    
    return results_dict

def main():
    """Main function"""
    # Training configuration
    config = {
        'batch_size': 8,
        'learning_rate': 1e-4,
        'num_epochs': 50,
        'weight_decay': 0.01,
        'use_scheduler': True,
        'early_stopping': True,
        'patience': 10,
        'num_workers': 0,
        
        # V5.2 specific configuration
        'v2_mae_path': 'src/mask_autoencoder/model/mae_best_20250528-174157.pth',  # TODO: ⚠️ UPDATE THIS PATH
        
        # Paths
        'save_dir': 'models/v5_2_training',
        'log_dir': 'logs/v5_2_training'
    }
    
    # Add timestamp to directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config['save_dir'] = f"{config['save_dir']}_{timestamp}"
    config['log_dir'] = f"{config['log_dir']}_{timestamp}"
    
    print("V5.2 Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Validate v2 MAE path
    if not os.path.exists(config['v2_mae_path']):
        print(f"❌ Error: v2 MAE checkpoint not found: {config['v2_mae_path']}")
        print("Please update the 'v2_mae_path' in the config")
        return
    
    # Train model
    results = train_model(config)
    
    return results

if __name__ == "__main__":
    main() 