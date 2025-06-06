#!/usr/bin/env python3
"""
Comprehensive and reliable comparison of v4.py vs quick_fix_v4.py models.
This test addresses the flaws in previous analysis to get accurate results.
Enhanced with model saving functionality.
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

# Import from the existing modules
from v4 import load_data, ProteinPairDataset, collate_fn, ProteinInteractionClassifier
from quick_fix_v4 import SimplifiedProteinClassifier

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
        'model_class': config['model_class']
    }
    
    # Add architecture-specific parameters for model reconstruction
    if config['model_class'] == 'simplified':
        checkpoint['model_params'] = {
            'hidden_dim': config['hidden_dim']
        }
    else:  # complex
        checkpoint['model_params'] = {
            'encoder_layers': config['encoder_layers'],
            'encoder_embed_dim': config['encoder_embed_dim'],
            'encoder_heads': config['encoder_heads'],
            'decoder_hidden_dims': config['decoder_hidden_dims'],
            'use_variable_length': True
        }
    
    # Save checkpoint
    torch.save(checkpoint, filepath)
    print(f"   💾 Saved {checkpoint_type} model: {filename}")
    
    return filepath

def load_model_from_checkpoint(checkpoint_path, device='cpu'):
    """Load model from checkpoint file"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Reconstruct model
    if checkpoint['model_class'] == 'simplified':
        model = SimplifiedProteinClassifier(**checkpoint['model_params'])
    else:  # complex
        model = ProteinInteractionClassifier(**checkpoint['model_params'])
    
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

def train_and_evaluate_model(model, train_loader, val_loader, config, device, save_dir=None):
    """Train and evaluate a model with proper metrics and early stopping"""
    print(f"\n🔧 Training: {config['name']}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Learning Rate: {config['lr']}")
    print(f"   Expected epochs: {config['max_epochs']}")
    
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
                'best_model': True
            }
            best_path = save_model_checkpoint(model, config, best_metrics, save_dir, checkpoint_type="best")
            saved_models['best_model_path'] = best_path
    
    final_metrics['saved_models'] = saved_models
    return final_metrics

def main():
    """Run comprehensive model comparison"""
    print("🎯 COMPREHENSIVE MODEL COMPARISON WITH MODEL SAVING")
    print("=" * 60)
    print("Training multiple model configurations with full validation sets")
    print("=" * 60)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create directories for saving models and results
    results_dir = f"model_comparison_results_{timestamp}"
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
    
    # Model configurations to test - diverse set for proper training
    configs = [
        {
            'name': 'QuickFix Baseline',
            'model_class': 'simplified',
            'hidden_dim': 256,
            'lr': 5e-3,
            'weight_decay': 0.01,
            'batch_size': 32,
            'max_epochs': 100,
            'use_scheduler': False,
            'description': 'Proven SimplifiedProteinClassifier configuration'
        },
        {
            'name': 'QuickFix Large',
            'model_class': 'simplified',
            'hidden_dim': 512,
            'lr': 3e-3,
            'weight_decay': 0.01,
            'batch_size': 32,
            'max_epochs': 100,
            'use_scheduler': False,
            'description': 'Larger SimplifiedProteinClassifier'
        },
        {
            'name': 'V4 Compact',
            'model_class': 'complex',
            'encoder_layers': 1,
            'encoder_embed_dim': 64,
            'encoder_heads': 2,
            'decoder_hidden_dims': [64, 32],
            'lr': 1e-3,
            'weight_decay': 0.01,
            'batch_size': 32,
            'max_epochs': 100,
            'use_scheduler': True,
            'description': 'Compact transformer with scheduler'
        },
        {
            'name': 'V4 Standard',
            'model_class': 'complex',
            'encoder_layers': 2,
            'encoder_embed_dim': 128,
            'encoder_heads': 4,
            'decoder_hidden_dims': [128, 64],
            'lr': 5e-4,
            'weight_decay': 0.01,
            'batch_size': 24,
            'max_epochs': 100,
            'use_scheduler': True,
            'description': 'Standard transformer configuration'
        },
        {
            'name': 'V4 Large',
            'model_class': 'complex',
            'encoder_layers': 3,
            'encoder_embed_dim': 256,
            'encoder_heads': 8,
            'decoder_hidden_dims': [256, 128, 64],
            'lr': 2e-4,
            'weight_decay': 0.01,
            'batch_size': 16,
            'max_epochs': 100,
            'use_scheduler': True,
            'description': 'Large transformer with 3 layers'
        },
        {
            'name': 'V4 Deep',
            'model_class': 'complex',
            'encoder_layers': 4,
            'encoder_embed_dim': 128,
            'encoder_heads': 4,
            'decoder_hidden_dims': [256, 128, 64, 32],
            'lr': 1e-4,
            'weight_decay': 0.01,
            'batch_size': 16,
            'max_epochs': 100,
            'use_scheduler': True,
            'description': 'Deep transformer with 4 layers'
        },
        {
            'name': 'V4 High LR',
            'model_class': 'complex',
            'encoder_layers': 2,
            'encoder_embed_dim': 128,
            'encoder_heads': 4,
            'decoder_hidden_dims': [128, 64],
            'lr': 5e-3,
            'weight_decay': 0.01,
            'batch_size': 16,
            'max_epochs': 100,
            'use_scheduler': False,
            'description': 'Transformer with high learning rate like SimplifiedProteinClassifier'
        },
        {
            'name': 'V4 Low LR',
            'model_class': 'complex',
            'encoder_layers': 2,
            'encoder_embed_dim': 128,
            'encoder_heads': 4,
            'decoder_hidden_dims': [128, 64],
            'lr': 1e-5,
            'weight_decay': 0.01,
            'batch_size': 16,
            'max_epochs': 100,
            'use_scheduler': False,
            'description': 'Transformer with very low learning rate'
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
        
        # Create model
        try:
            if config['model_class'] == 'simplified':
                model = SimplifiedProteinClassifier(
                    hidden_dim=config['hidden_dim']
                ).to(device)
            else:  # complex
                model = ProteinInteractionClassifier(
                    encoder_layers=config['encoder_layers'],
                    encoder_embed_dim=config['encoder_embed_dim'], 
                    encoder_heads=config['encoder_heads'],
                    use_variable_length=True,
                    decoder_hidden_dims=config['decoder_hidden_dims']
                ).to(device)
            
            # Train and evaluate (with model saving)
            metrics = train_and_evaluate_model(model, train_loader, val_loader, config, device, save_dir=models_dir)
            metrics['config'] = config
            results[config['name']] = metrics
            
            # Summary
            status = "✅ SUCCESS" if metrics['best_val_auc'] > 0.6 else "⚠️ PARTIAL" if metrics['best_val_auc'] > 0.55 else "❌ FAILED"
            print(f"\n📊 Results: {status}")
            print(f"   Best AUC: {metrics['best_val_auc']:.4f} (epoch {metrics['best_epoch']})")
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
                'saved_models': {}
            }
        
        # Memory cleanup
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()
    
    # Final analysis
    print(f"\n{'='*60}")
    print("🎯 COMPREHENSIVE ANALYSIS")
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
        print(f"   {name}:")
        print(f"      AUC: {result['best_val_auc']:.4f}")
        print(f"      LR: {config['lr']}, Batch: {config['batch_size']}")
        print(f"      Architecture: {config['model_class']}")
        if config['model_class'] == 'complex':
            print(f"      Layers: {config['encoder_layers']}, Embed: {config['encoder_embed_dim']}, Heads: {config['encoder_heads']}")
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
    
    # Save comprehensive results
    results_file = os.path.join(results_dir, 'comprehensive_comparison_results.json')
    
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
    
    # Create model loading instructions
    instructions_file = os.path.join(results_dir, 'model_loading_instructions.py')
    with open(instructions_file, 'w') as f:
        f.write('"""\nInstructions for loading saved models from this comparison run.\n"""\n\n')
        f.write('import torch\n')
        f.write('from v4 import ProteinInteractionClassifier\n')
        f.write('from quick_fix_v4 import SimplifiedProteinClassifier\n\n')
        
        f.write('def load_model_from_checkpoint(checkpoint_path, device="cpu"):\n')
        f.write('    """Load model from checkpoint file"""\n')
        f.write('    checkpoint = torch.load(checkpoint_path, map_location=device)\n')
        f.write('    \n')
        f.write('    # Reconstruct model\n')
        f.write('    if checkpoint["model_class"] == "simplified":\n')
        f.write('        model = SimplifiedProteinClassifier(**checkpoint["model_params"])\n')
        f.write('    else:  # complex\n')
        f.write('        model = ProteinInteractionClassifier(**checkpoint["model_params"])\n')
        f.write('    \n')
        f.write('    # Load state dict\n')
        f.write('    model.load_state_dict(checkpoint["model_state_dict"])\n')        
        f.write('    model.to(device)\n')
        f.write('    \n')
        f.write('    return model, checkpoint["config"], checkpoint["metrics"]\n\n')
        
        if working_models:
            f.write('# Top performing models:\n')
            for i, (name, result) in enumerate(working_models[:3], 1):
                saved = result.get('saved_models', {})
                if saved:
                    best_path = saved.get('best_model_path', saved.get('final_model_path'))
                    if best_path:
                        f.write(f'# {i}. {name} (AUC: {result["best_val_auc"]:.4f})\n')
                        f.write(f'# model_{i}, config_{i}, metrics_{i} = load_model_from_checkpoint("{os.path.basename(best_path)}")\n\n')
        
        f.write('# All available models from this run:\n')
        for name, result in results.items():
            saved = result.get('saved_models', {})
            if saved:
                f.write(f'# {name}:\n')
                for model_type, path in saved.items():
                    f.write(f'#   {model_type}: {os.path.basename(path)}\n')
    
    # Recommendations with model loading info
    print(f"\n{'='*60}")
    print("💡 FINAL RECOMMENDATIONS")
    print(f"{'='*60}")
    
    if working_models:
        print(f"🏆 TOP 3 MODELS:")
        for i, (name, result) in enumerate(working_models[:3], 1):
            config = result['config']
            print(f"   {i}. {name}")
            print(f"      Performance: {result['best_val_auc']:.4f} AUC")
            print(f"      Architecture: {config['model_class']}")
            print(f"      Learning Rate: {config['lr']}")
            if config['model_class'] == 'complex':
                print(f"      Transformer: {config['encoder_layers']} layers, {config['encoder_embed_dim']} dim, {config['encoder_heads']} heads")
            
            # Show how to load the model
            saved = result.get('saved_models', {})
            if saved:
                best_path = saved.get('best_model_path', saved.get('final_model_path'))
                if best_path:
                    print(f"      Load: load_model_from_checkpoint('{os.path.basename(best_path)}')")
            print()
        
        # Analysis insights
        print(f"🎯 KEY INSIGHTS:")
        
        # Architecture comparison
        simple_models = [(n, r) for n, r in working_models if r['config']['model_class'] == 'simplified']
        complex_models = [(n, r) for n, r in working_models if r['config']['model_class'] == 'complex']
        
        if simple_models and complex_models:
            simple_avg = np.mean([r['best_val_auc'] for _, r in simple_models])
            complex_avg = np.mean([r['best_val_auc'] for _, r in complex_models])
            print(f"   SimplifiedProteinClassifier average: {simple_avg:.3f} ({len(simple_models)} models)")
            print(f"   ProteinInteractionClassifier average: {complex_avg:.3f} ({len(complex_models)} models)")
            
            if complex_avg > simple_avg + 0.02:
                print(f"   → Complex transformer architecture shows clear advantage")
            elif simple_avg > complex_avg + 0.02:
                print(f"   → Simple architecture is more effective for this dataset")
            else:
                print(f"   → Both architectures perform similarly")
        
        # Learning rate analysis
        lr_performance = {}
        for name, result in working_models:
            lr = result['config']['lr']
            if lr not in lr_performance:
                lr_performance[lr] = []
            lr_performance[lr].append(result['best_val_auc'])
        
        print(f"   Learning rate analysis:")
        for lr in sorted(lr_performance.keys()):
            aucs = lr_performance[lr]
            print(f"      LR {lr}: {np.mean(aucs):.3f} avg AUC ({len(aucs)} models)")
        
    else:
        print("❌ No models achieved satisfactory performance")
        print("   Check data quality, model implementations, and resource constraints")
    
    print(f"\n📁 All results saved to: {results_dir}")
    print(f"   📊 Results summary: comprehensive_comparison_results.json")
    print(f"   💾 Model files: saved_models/")
    print(f"   📝 Loading instructions: model_loading_instructions.py")
    
    return results

if __name__ == "__main__":
    results = main()