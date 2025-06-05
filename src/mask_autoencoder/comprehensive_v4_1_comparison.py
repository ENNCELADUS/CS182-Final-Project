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
from v4_1 import load_data, ProteinPairDataset, collate_fn, SimplifiedProteinClassifier, TransformerEnhancedProteinClassifier, create_model

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
    
    final_metrics['saved_models'] = saved_models
    return final_metrics

def main():
    """
    Comprehensive comparison of v4.1 model configurations with enhanced evaluation
    """
    print("🧬 COMPREHENSIVE V4.1 MODEL COMPARISON")
    print("=" * 60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    save_dir = "models/v4_1_comparison"
    results_dir = "results/v4_1_comparison"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
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
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             collate_fn=collate_fn, drop_last=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           collate_fn=collate_fn, num_workers=2)
    test1_loader = DataLoader(test1_dataset, batch_size=batch_size, shuffle=False, 
                             collate_fn=collate_fn, num_workers=2)
    test2_loader = DataLoader(test2_dataset, batch_size=batch_size, shuffle=False, 
                             collate_fn=collate_fn, num_workers=2)
    
    # Model configurations to test
    model_configs = [
        {
            'name': 'Simplified_v4.1_256',
            'model_class': 'simplified',
            'embed_dim': 256,
            'lr': 5e-3,
            'max_epochs': 25,
            'use_scheduler': True,
            'weight_decay': 0.01
        },
        {
            'name': 'Enhanced_v4.1_256_2layers',
            'model_class': 'enhanced',
            'embed_dim': 256,
            'num_transformer_layers': 2,
            'num_heads': 8,
            'lr': 3e-3,  # Slightly lower LR for transformer
            'max_epochs': 30,
            'use_scheduler': True,
            'weight_decay': 0.01
        },
        {
            'name': 'Enhanced_v4.1_256_3layers',
            'model_class': 'enhanced',
            'embed_dim': 256,
            'num_transformer_layers': 3,
            'num_heads': 8,
            'lr': 2e-3,  # Even lower LR for deeper model
            'max_epochs': 35,
            'use_scheduler': True,
            'weight_decay': 0.01
        }
    ]
    
    # Store results
    all_results = {}
    
    # Train and evaluate each configuration
    for config in model_configs:
        print(f"\n{'='*60}")
        print(f"🚀 TRAINING: {config['name']}")
        print(f"{'='*60}")
        
        try:
            # Create model
            if config['model_class'] == 'simplified':
                model = create_model('simple', 
                                   input_dim=960, 
                                   hidden_dim=config['embed_dim'],
                                   dropout=0.3)
            elif config['model_class'] == 'enhanced':
                model = create_model('enhanced',
                                   input_dim=960,
                                   hidden_dim=config['embed_dim'],
                                   num_transformer_layers=config['num_transformer_layers'],
                                   num_heads=config['num_heads'],
                                   dropout=0.3)
            
            model = model.to(device)
            
            # Train model
            results = train_and_evaluate_model(
                model, train_loader, val_loader, config, device, save_dir
            )
            
            # Evaluate on test sets
            print(f"\n📈 Evaluating {config['name']} on test sets...")
            
            test1_results = evaluate_on_test_set(model, test1_loader, device, "Test1")
            test2_results = evaluate_on_test_set(model, test2_loader, device, "Test2")
            
            # Store comprehensive results
            all_results[config['name']] = {
                'config': config,
                'training_results': results,
                'test1_results': test1_results,
                'test2_results': test2_results
            }
            
            # Save test predictions and logits
            save_test_predictions(model, test1_loader, device, 
                                os.path.join(results_dir, f"{config['name']}_test1_predictions.json"))
            save_test_predictions(model, test2_loader, device, 
                                os.path.join(results_dir, f"{config['name']}_test2_predictions.json"))
            
            # Generate plots
            create_evaluation_plots(config['name'], results, test1_results, test2_results, results_dir)
            
            print(f"✅ {config['name']} completed successfully!")
            
        except Exception as e:
            print(f"❌ Error training {config['name']}: {str(e)}")
            all_results[config['name']] = {'error': str(e)}
            continue
    
    # Generate comparison report
    generate_comparison_report(all_results, results_dir)
    
    print(f"\n🎉 COMPARISON COMPLETE!")
    print(f"Results saved to: {results_dir}")
    print(f"Models saved to: {save_dir}")


def evaluate_on_test_set(model, test_loader, device, test_name):
    """Evaluate model on test set and return comprehensive metrics"""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    model.eval()
    all_preds = []
    all_probs = []
    all_logits = []
    all_labels = []
    all_ids = []
    
    with torch.no_grad():
        for emb_a, emb_b, lengths_a, lengths_b, interactions in test_loader:
            emb_a = emb_a.to(device).float()
            emb_b = emb_b.to(device).float()
            lengths_a = lengths_a.to(device)
            lengths_b = lengths_b.to(device)
            interactions = interactions.to(device).float()
            
            logits = model(emb_a, emb_b, lengths_a, lengths_b)
            if logits.dim() > 1:
                logits = logits.squeeze(-1)
            
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            all_logits.extend(logits.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(interactions.cpu().numpy())
    
    # Calculate metrics
    auroc = roc_auc_score(all_labels, all_probs)
    auprc = average_precision_score(all_labels, all_probs)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    
    # Get precision-recall curve
    precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs)
    
    results = {
        'test_name': test_name,
        'auroc': auroc,
        'auprc': auprc,
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'predictions': all_preds,
        'probabilities': all_probs,
        'logits': all_logits,
        'labels': all_labels,
        'precision_curve': precision_curve.tolist(),
        'recall_curve': recall_curve.tolist()
    }
    
    print(f"   {test_name} - AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}, Acc: {accuracy:.4f}, F1: {f1:.4f}")
    
    return results


def save_test_predictions(model, test_loader, device, save_path):
    """Save detailed test predictions with logits"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        batch_idx = 0
        for emb_a, emb_b, lengths_a, lengths_b, interactions in test_loader:
            emb_a = emb_a.to(device).float()
            emb_b = emb_b.to(device).float()
            lengths_a = lengths_a.to(device)
            lengths_b = lengths_b.to(device)
            interactions = interactions.to(device).float()
            
            logits = model(emb_a, emb_b, lengths_a, lengths_b)
            if logits.dim() > 1:
                logits = logits.squeeze(-1)
            
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            # Store batch predictions
            for i in range(len(interactions)):
                predictions.append({
                    'batch_idx': batch_idx,
                    'sample_idx': i,
                    'true_label': int(interactions[i].item()),
                    'prediction': int(preds[i].item()),
                    'probability': float(probs[i].item()),
                    'logit': float(logits[i].item())
                })
            
            batch_idx += 1
    
    # Save to JSON
    with open(save_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"   💾 Saved predictions to: {save_path}")


def create_evaluation_plots(model_name, training_results, test1_results, test2_results, results_dir):
    """Create comprehensive evaluation plots"""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{model_name} - Comprehensive Evaluation', fontsize=16, fontweight='bold')
    
    # Training curves
    history = training_results['history']
    
    # Plot 1: Loss curves
    axes[0, 0].plot(history['train_loss'], label='Train Loss', color='blue')
    axes[0, 0].plot(history['val_loss'], label='Val Loss', color='red')
    axes[0, 0].set_title('Training & Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: AUC curves
    axes[0, 1].plot(history['train_auc'], label='Train AUC', color='blue')
    axes[0, 1].plot(history['val_auc'], label='Val AUC', color='red')
    axes[0, 1].set_title('Training & Validation AUC')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('AUC')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Learning rate
    axes[0, 2].plot(history['learning_rate'], color='green')
    axes[0, 2].set_title('Learning Rate Schedule')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Learning Rate')
    axes[0, 2].set_yscale('log')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: ROC curves for test sets
    # Test1 ROC
    fpr1, tpr1, _ = roc_curve(test1_results['labels'], test1_results['probabilities'])
    axes[1, 0].plot(fpr1, tpr1, label=f"Test1 (AUC={test1_results['auroc']:.3f})", color='blue')
    
    # Test2 ROC
    fpr2, tpr2, _ = roc_curve(test2_results['labels'], test2_results['probabilities'])
    axes[1, 0].plot(fpr2, tpr2, label=f"Test2 (AUC={test2_results['auroc']:.3f})", color='red')
    
    axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[1, 0].set_title('ROC Curves - Test Sets')
    axes[1, 0].set_xlabel('False Positive Rate')
    axes[1, 0].set_ylabel('True Positive Rate')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Precision-Recall curves
    axes[1, 1].plot(test1_results['recall_curve'], test1_results['precision_curve'], 
                    label=f"Test1 (AUPRC={test1_results['auprc']:.3f})", color='blue')
    axes[1, 1].plot(test2_results['recall_curve'], test2_results['precision_curve'], 
                    label=f"Test2 (AUPRC={test2_results['auprc']:.3f})", color='red')
    axes[1, 1].set_title('Precision-Recall Curves')
    axes[1, 1].set_xlabel('Recall')
    axes[1, 1].set_ylabel('Precision')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Test metrics comparison
    metrics = ['AUROC', 'AUPRC', 'Accuracy', 'F1 Score']
    test1_vals = [test1_results['auroc'], test1_results['auprc'], 
                  test1_results['accuracy'], test1_results['f1_score']]
    test2_vals = [test2_results['auroc'], test2_results['auprc'], 
                  test2_results['accuracy'], test2_results['f1_score']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[1, 2].bar(x - width/2, test1_vals, width, label='Test1', color='blue', alpha=0.7)
    axes[1, 2].bar(x + width/2, test2_vals, width, label='Test2', color='red', alpha=0.7)
    axes[1, 2].set_title('Test Set Metrics Comparison')
    axes[1, 2].set_ylabel('Score')
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(metrics, rotation=45)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(results_dir, f'{model_name}_evaluation_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   📊 Saved evaluation plots: {plot_path}")


def generate_comparison_report(all_results, results_dir):
    """Generate comprehensive comparison report"""
    
    print("\n" + "="*80)
    print("📋 FINAL COMPARISON REPORT")
    print("="*80)
    
    # Create summary table
    summary_data = []
    
    for model_name, results in all_results.items():
        if 'error' in results:
            print(f"❌ {model_name}: {results['error']}")
            continue
            
        training = results['training_results']
        test1 = results['test1_results']
        test2 = results['test2_results']
        
        summary_data.append({
            'Model': model_name,
            'Parameters': f"{training['num_parameters']:,}",
            'Best Val AUC': f"{training['best_val_auc']:.4f}",
            'Test1 AUROC': f"{test1['auroc']:.4f}",
            'Test1 AUPRC': f"{test1['auprc']:.4f}",
            'Test2 AUROC': f"{test2['auroc']:.4f}",
            'Test2 AUPRC': f"{test2['auprc']:.4f}",
            'Stable Training': "✅" if training['stable_training'] else "❌"
        })
        
        print(f"\n🔹 {model_name}:")
        print(f"   Parameters: {training['num_parameters']:,}")
        print(f"   Best Val AUC: {training['best_val_auc']:.4f} (epoch {training['best_epoch']})")
        print(f"   Test1 - AUROC: {test1['auroc']:.4f}, AUPRC: {test1['auprc']:.4f}")
        print(f"   Test2 - AUROC: {test2['auroc']:.4f}, AUPRC: {test2['auprc']:.4f}")
        print(f"   Stable Training: {'Yes' if training['stable_training'] else 'No'}")
    
    # Save detailed results
    detailed_report_path = os.path.join(results_dir, 'detailed_comparison_report.json')
    with open(detailed_report_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Save summary table
    import pandas as pd
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = os.path.join(results_dir, 'model_comparison_summary.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    
    print(f"\n💾 Detailed report saved: {detailed_report_path}")
    print(f"💾 Summary table saved: {summary_csv_path}")
    
    # Print best models
    if summary_data:
        best_test1_auroc = max(summary_data, key=lambda x: float(x['Test1 AUROC'].split(':')[-1] if ':' in x['Test1 AUROC'] else x['Test1 AUROC']))
        best_test2_auroc = max(summary_data, key=lambda x: float(x['Test2 AUROC'].split(':')[-1] if ':' in x['Test2 AUROC'] else x['Test2 AUROC']))
        
        print(f"\n🏆 BEST PERFORMING MODELS:")
        print(f"   Test1 AUROC: {best_test1_auroc['Model']} ({best_test1_auroc['Test1 AUROC']})")
        print(f"   Test2 AUROC: {best_test2_auroc['Model']} ({best_test2_auroc['Test2 AUROC']})")


if __name__ == "__main__":
    main()