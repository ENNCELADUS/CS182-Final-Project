#!/usr/bin/env python3
"""
Comprehensive comparison of v4.1 simplified architecture configurations.
This script tests various configurations of the v4.1 model to find optimal settings.
Enhanced with model saving functionality and detailed analysis.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
from datetime import datetime
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, Subset
import warnings
warnings.filterwarnings('ignore')

# Import from the v4.1 module
from v4_1 import load_data, ProteinPairDataset, collate_fn, SimplifiedProteinInteractionClassifier
from v4_1_multi_layer import EnhancedProteinInteractionClassifier

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0

def save_model_checkpoint(model, config, metrics, save_dir, checkpoint_type="final"):
    """Save model checkpoint with configuration and metrics"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Create descriptive filename
    model_name = config['name'].replace(' ', '_').lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{checkpoint_type}_{timestamp}.pth"
    filepath = os.path.join(save_dir, filename)
    
    # Prepare checkpoint data
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'metrics': metrics,
        'timestamp': timestamp,
        'checkpoint_type': checkpoint_type,
        'model_class': config.get('model_class', 'simplified_v4_1')
    }
    
    # Add architecture-specific parameters for model reconstruction
    checkpoint['model_params'] = {
        'embed_dim': config['embed_dim'],
        'max_length': config.get('max_length', 512)
    }
    
    # Add enhanced model parameters if applicable
    if config.get('num_layers'):
        checkpoint['model_params']['num_layers'] = config['num_layers']
        checkpoint['model_params']['num_heads'] = config.get('num_heads', 4)
    
    # Save checkpoint
    torch.save(checkpoint, filepath)
    print(f"   💾 Saved {checkpoint_type} model: {filename}")
    
    return filepath

def load_model_from_checkpoint(checkpoint_path, device='cpu'):
    """Load model from checkpoint file"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Reconstruct model
    model = SimplifiedProteinInteractionClassifier(**checkpoint['model_params'])
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    return model, checkpoint['config'], checkpoint['metrics']

def create_stable_validation_set(val_dataset, min_samples=1000):
    """Create a stable validation set with sufficient samples"""
    if len(val_dataset) < min_samples:
        # Replicate the dataset if it's too small
        indices = list(range(len(val_dataset))) * (min_samples // len(val_dataset) + 1)
        indices = indices[:min_samples]
        return Subset(val_dataset, indices)
    else:
        # Use first min_samples for consistency
        return Subset(val_dataset, list(range(min_samples)))

def count_model_parameters(model):
    """Count total model parameters"""
    return sum(p.numel() for p in model.parameters())

def train_and_evaluate_model(model, train_loader, val_loader, config, device, save_dir=None):
    """Train and evaluate a v4.1 model with proper metrics and early stopping"""
    print(f"\n🔧 Training: {config['name']}")
    
    num_params = count_model_parameters(model)
    print(f"   Parameters: {num_params:,}")
    print(f"   Learning Rate: {config['lr']}")
    print(f"   Expected epochs: {config['max_epochs']}")
    print(f"   Embed dim: {config['embed_dim']}")
    if 'num_layers' in config:
        print(f"   Layers: {config['num_layers']}")
    
    # Setup training
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['lr'],
        weight_decay=config.get('weight_decay', 0.01)
    )
    
    # Add learning rate scheduler if specified
    scheduler = None
    if config.get('use_scheduler', False):
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config['lr'], epochs=config['max_epochs'],
            steps_per_epoch=len(train_loader), pct_start=0.1
        )
    
    criterion = nn.BCEWithLogitsLoss()
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [], 'val_auc': [], 'val_acc': [],
        'train_auc': [], 'val_f1': [], 'learning_rate': []
    }
    
    best_val_auc = 0
    best_epoch = 0
    best_model_state = None
    stable_training = True
    
    for epoch in range(1, config['max_epochs'] + 1):
        # Training phase
        model.train()
        train_losses = []
        train_preds = []
        train_probs = []
        train_labels = []
        
        for batch_idx, (emb_a, emb_b, lengths_a, lengths_b, interactions) in enumerate(train_loader):
            # Memory management - limit batches if needed
            if config.get('limit_batches') and batch_idx >= config['limit_batches']:
                break
                
            try:
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
                
                # Check for training instability
                if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 100:
                    print(f"   ⚠️ Training instability detected at epoch {epoch}, batch {batch_idx}")
                    print(f"   Loss: {loss.item():.4f}")
                    stable_training = False
                    break
                
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
                    
            except torch.cuda.OutOfMemoryError:
                print(f"   ❌ GPU OOM at epoch {epoch}, batch {batch_idx}")
                stable_training = False
                break
            except Exception as e:
                print(f"   ❌ Error at epoch {epoch}, batch {batch_idx}: {str(e)}")
                stable_training = False
                break
        
        if not stable_training or not train_losses:
            break
            
        # Validation phase
        model.eval()
        val_losses = []
        val_preds = []
        val_probs = []
        val_labels = []
        
        with torch.no_grad():
            for batch_idx, (emb_a, emb_b, lengths_a, lengths_b, interactions) in enumerate(val_loader):
                try:
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
                    
                except:
                    continue
        
        if not val_losses:
            print(f"   ⚠️ No valid validation batches at epoch {epoch}")
            break
            
        # Calculate metrics
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        # Only calculate metrics if we have both classes
        if len(set(train_labels)) > 1 and len(set(val_labels)) > 1:
            train_auc = roc_auc_score(train_labels, train_probs)
            val_auc = roc_auc_score(val_labels, val_probs)
            val_acc = accuracy_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds)
        else:
            train_auc = 0.5
            val_auc = 0.5
            val_acc = accuracy_score(val_labels, val_preds) if val_preds else 0.5
            val_f1 = 0.0
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Track best performance and save best model state
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
        
        # Progress reporting
        if epoch <= 5 or epoch % 10 == 0:
            print(f"   Epoch {epoch:2d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                  f"Val AUC={val_auc:.4f}, Val Acc={val_acc:.4f}")
        
        # Early stopping check
        early_stopping(val_auc)
        if early_stopping.early_stop:
            print(f"   🛑 Early stopping at epoch {epoch}")
            break
    
    # Restore best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation
    final_metrics = {
        'best_val_auc': best_val_auc,
        'best_epoch': best_epoch,
        'final_val_auc': history['val_auc'][-1] if history['val_auc'] else 0,
        'final_val_acc': history['val_acc'][-1] if history['val_acc'] else 0,
        'final_val_f1': history['val_f1'][-1] if history['val_f1'] else 0,
        'stable_training': stable_training,
        'epochs_completed': len(history['train_loss']),
        'converged': len(history['train_loss']) > 5 and stable_training,
        'learning_occurred': len(history['train_loss']) > 3 and 
                           (history['train_loss'][0] - history['train_loss'][-1] > 0.1),
        'num_parameters': num_params,
        'history': history
    }
    
    # Save models if save_dir is provided
    saved_models = {}
    if save_dir:
        # Save final model
        final_path = save_model_checkpoint(model, config, final_metrics, save_dir, checkpoint_type="final")
        saved_models['final_model_path'] = final_path
        
        # If we have a best model state different from final, save it too
        if best_model_state is not None and best_epoch != len(history['train_loss']):
            model.load_state_dict(best_model_state)
            best_metrics = {
                'epoch': best_epoch,
                'val_auc': best_val_auc,
                'best_model': True,
                'num_parameters': num_params
            }
            best_path = save_model_checkpoint(model, config, best_metrics, save_dir, checkpoint_type="best")
            saved_models['best_model_path'] = best_path
    
    final_metrics['saved_models'] = saved_models
    return final_metrics

def main():
    """Run comprehensive v4.1 model comparison"""
    print("🎯 COMPREHENSIVE V4.1 MODEL COMPARISON WITH MODEL SAVING")
    print("=" * 60)
    print("Testing various configurations of the simplified v4.1 architecture")
    print("Including both single-layer and enhanced multi-layer variants")
    print("=" * 60)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create directories for saving models and results
    results_dir = f"v4_1_comparison_results_{timestamp}"
    models_dir = os.path.join(results_dir, "saved_models")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"📁 Results will be saved to: {results_dir}")
    print(f"💾 Models will be saved to: {models_dir}")
    
    # Load data
    train_data, val_data, _, _, protein_embeddings = load_data()
    
    # Create datasets with full validation set
    train_dataset = ProteinPairDataset(train_data, protein_embeddings)
    val_dataset = ProteinPairDataset(val_data, protein_embeddings)
    
    print(f"📊 Dataset sizes:")
    print(f"   Training: {len(train_dataset):,} samples")
    print(f"   Validation: {len(val_dataset):,} samples (full)")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    # Model configurations to test - including both single-layer and multi-layer variants
    configs = [
        # Original v4.1 Single-layer variants
        {
            'name': 'V4.1 Tiny (1L)',
            'embed_dim': 64,
            'max_length': 256,
            'lr': 1e-2,
            'weight_decay': 0.01,
            'batch_size': 64,
            'max_epochs': 40,
            'use_scheduler': True,
            'description': 'Ultra-small single-layer v4.1',
            'model_class': 'simplified_v4_1'
        },
        {
            'name': 'V4.1 Default (1L)',
            'embed_dim': 128,
            'max_length': 512,
            'lr': 5e-3,
            'weight_decay': 0.01,
            'batch_size': 32,
            'max_epochs': 40,
            'use_scheduler': True,
            'description': 'Default single-layer v4.1',
            'model_class': 'simplified_v4_1'
        },
        
        # Enhanced Multi-layer variants (2-3 layers)
        {
            'name': 'V4.1 Enhanced 2L-96d',
            'embed_dim': 96,
            'num_layers': 2,
            'num_heads': 4,
            'max_length': 384,
            'lr': 4e-3,
            'weight_decay': 0.01,
            'batch_size': 48,
            'max_epochs': 40,
            'use_scheduler': True,
            'description': '2-layer enhanced v4.1 (284K params)',
            'model_class': 'enhanced_v4_1'
        },
        {
            'name': 'V4.1 Enhanced 2L-128d',
            'embed_dim': 128,
            'num_layers': 2,
            'num_heads': 4,
            'max_length': 512,
            'lr': 3e-3,
            'weight_decay': 0.01,
            'batch_size': 32,
            'max_epochs': 40,
            'use_scheduler': True,
            'description': '2-layer enhanced v4.1 (463K params)',
            'model_class': 'enhanced_v4_1'
        },
        {
            'name': 'V4.1 Enhanced 3L-96d',
            'embed_dim': 96,
            'num_layers': 3,
            'num_heads': 4,
            'max_length': 384,
            'lr': 3e-3,
            'weight_decay': 0.01,
            'batch_size': 48,
            'max_epochs': 40,
            'use_scheduler': True,
            'description': '3-layer enhanced v4.1 (359K params)',
            'model_class': 'enhanced_v4_1'
        },
        
        # High learning rate tests
        {
            'name': 'V4.1 High LR (1L)',
            'embed_dim': 128,
            'max_length': 512,
            'lr': 1e-2,
            'weight_decay': 0.005,
            'batch_size': 32,
            'max_epochs': 40,
            'use_scheduler': False,
            'description': 'High LR single-layer v4.1',
            'model_class': 'simplified_v4_1'
        },
        {
            'name': 'V4.1 Enhanced High LR (2L)',
            'embed_dim': 96,
            'num_layers': 2,
            'num_heads': 4,
            'max_length': 384,
            'lr': 8e-3,
            'weight_decay': 0.005,
            'batch_size': 48,
            'max_epochs': 40,
            'use_scheduler': False,
            'description': 'High LR 2-layer enhanced v4.1',
            'model_class': 'enhanced_v4_1'
        }
    ]
    
    results = {}
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {config['name']}")
        print(f"Description: {config['description']}")
        print(f"{'='*60}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True,
            num_workers=2 if config['batch_size'] >= 16 else 0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2 if config['batch_size'] >= 16 else 0
        )
        
        # Create model based on type
        try:
            if config['model_class'] == 'enhanced_v4_1':
                # Enhanced multi-layer model
                model = EnhancedProteinInteractionClassifier(
                    embed_dim=config['embed_dim'],
                    num_layers=config['num_layers'],
                    num_heads=config['num_heads'],
                    max_length=config['max_length']
                ).to(device)
            else:
                # Original single-layer model
                model = SimplifiedProteinInteractionClassifier(
                    embed_dim=config['embed_dim'],
                    max_length=config['max_length']
                ).to(device)
            
            # Train and evaluate (with model saving)
            metrics = train_and_evaluate_model(model, train_loader, val_loader, config, device, save_dir=models_dir)
            metrics['config'] = config
            results[config['name']] = metrics
            
            # Summary
            status = "✅ SUCCESS" if metrics['best_val_auc'] > 0.6 else "⚠️ PARTIAL" if metrics['best_val_auc'] > 0.55 else "❌ FAILED"
            print(f"\n📊 Results: {status}")
            print(f"   Best AUC: {metrics['best_val_auc']:.4f} (epoch {metrics['best_epoch']})")
            print(f"   Parameters: {metrics['num_parameters']:,}")
            print(f"   Stable training: {metrics['stable_training']}")
            print(f"   Epochs completed: {metrics['epochs_completed']}")
            
            # Print saved model info
            if 'saved_models' in metrics:
                saved = metrics['saved_models']
                if saved:
                    print(f"   💾 Saved models:")
                    for model_type, path in saved.items():
                        print(f"      {model_type}: {os.path.basename(path)}")
                        
        except Exception as e:
            print(f"❌ Failed to test {config['name']}: {str(e)}")
            results[config['name']] = {
                'error': str(e),
                'best_val_auc': 0.0,
                'stable_training': False,
                'config': config,
                'saved_models': {},
                'num_parameters': 0
            }
        
        # Memory cleanup
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()
    
    # Final analysis
    print(f"\n{'='*60}")
    print("🎯 COMPREHENSIVE V4.1 ANALYSIS")
    print(f"{'='*60}")
    
    # Sort by performance
    working_models = [(name, r) for name, r in results.items() 
                     if r.get('best_val_auc', 0) > 0.55 and r.get('stable_training', False)]
    working_models.sort(key=lambda x: x[1]['best_val_auc'], reverse=True)
    
    failing_models = [(name, r) for name, r in results.items() 
                     if r.get('best_val_auc', 0) <= 0.55 or not r.get('stable_training', False)]
    
    print(f"\n✅ WORKING MODELS ({len(working_models)}):")
    for name, result in working_models:
        config = result['config']
        layers = config.get('num_layers', 1)
        print(f"   {name}:")
        print(f"      AUC: {result['best_val_auc']:.4f}")
        print(f"      Parameters: {result.get('num_parameters', 0):,}")
        print(f"      Config: {layers}L, Embed={config['embed_dim']}, LR={config['lr']}, Batch={config['batch_size']}")
        # Show saved models
        saved = result.get('saved_models', {})
        if saved:
            print(f"      Saved models: {', '.join(saved.keys())}")
    
    print(f"\n❌ FAILING MODELS ({len(failing_models)}):")
    for name, result in failing_models:
        config = result['config']
        issue = "Training unstable" if not result.get('stable_training', False) else f"Low AUC ({result.get('best_val_auc', 0):.3f})"
        print(f"   {name}: {issue}")
        if 'error' in result:
            print(f"      Error: {result['error']}")
    
    # Architecture comparison
    if working_models:
        single_layer = [(n, r) for n, r in working_models if r['config'].get('num_layers', 1) == 1]
        multi_layer = [(n, r) for n, r in working_models if r['config'].get('num_layers', 1) > 1]
        
        print(f"\n🔬 ARCHITECTURE COMPARISON:")
        if single_layer:
            avg_single = np.mean([r['best_val_auc'] for _, r in single_layer])
            print(f"   Single-layer (1L): {len(single_layer)} models, {avg_single:.3f} avg AUC")
        if multi_layer:
            avg_multi = np.mean([r['best_val_auc'] for _, r in multi_layer])
            print(f"   Multi-layer (2-3L): {len(multi_layer)} models, {avg_multi:.3f} avg AUC")
            
            # Breakdown by layer count
            two_layer = [(n, r) for n, r in multi_layer if r['config']['num_layers'] == 2]
            three_layer = [(n, r) for n, r in multi_layer if r['config']['num_layers'] == 3]
            
            if two_layer:
                avg_2l = np.mean([r['best_val_auc'] for _, r in two_layer])
                print(f"   2-layer models: {len(two_layer)} models, {avg_2l:.3f} avg AUC")
            if three_layer:
                avg_3l = np.mean([r['best_val_auc'] for _, r in three_layer])
                print(f"   3-layer models: {len(three_layer)} models, {avg_3l:.3f} avg AUC")
    
    # Save comprehensive results
    results_file = os.path.join(results_dir, 'v4_1_comparison_results.json')
    
    # Prepare results for JSON (remove non-serializable items)
    json_results = {}
    for name, result in results.items():
        json_result = {k: v for k, v in result.items() if k != 'history'}
        if 'history' in result:
            # Only save summary of history
            history = result['history']
            json_result['history_summary'] = {
                'epochs': len(history.get('train_loss', [])),
                'final_train_loss': history['train_loss'][-1] if history.get('train_loss') else None,
                'best_val_auc': max(history['val_auc']) if history.get('val_auc') else 0
            }
        json_results[name] = json_result
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n📁 All v4.1 results saved to: {results_dir}")
    print(f"   📊 Results summary: v4_1_comparison_results.json")
    print(f"   💾 Model files: saved_models/")
    
    # Final recommendation
    if working_models:
        best_model = working_models[0]
        print(f"\n🏆 BEST MODEL: {best_model[0]}")
        print(f"   AUC: {best_model[1]['best_val_auc']:.4f}")
        print(f"   Parameters: {best_model[1]['num_parameters']:,}")
        config = best_model[1]['config']
        layers = config.get('num_layers', 1)
        print(f"   Architecture: {layers} layer(s), {config['embed_dim']} embed_dim")
        print(f"   Training: LR={config['lr']}, Batch={config['batch_size']}")
    
    return results

if __name__ == "__main__":
    results = main() 