#!/usr/bin/env python3
"""
Test script for the fixed protein interaction model.
Evaluates models/fixed_model_best.pth on test1 and test2 datasets.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Import from existing modules
from v4 import load_data, ProteinPairDataset, collate_fn

class SimplifiedProteinClassifier(nn.Module):
    """
    Simplified model for testing - must match the trained model architecture exactly
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

def load_trained_model(model_path, device):
    """Load the trained fixed model"""
    print(f"Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Create model with same architecture as trained
    model = SimplifiedProteinClassifier(hidden_dim=256)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    print(f"   - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   - Training epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"   - Validation AUC: {checkpoint.get('val_auc', 'unknown'):.4f}")
    
    return model

def evaluate_model_on_dataset(model, dataset_name, test_data, embeddings_dict, device, batch_size=64):
    """Evaluate model on a test dataset"""
    print(f"\n🧪 EVALUATING ON {dataset_name.upper()}")
    print("=" * 50)
    
    # Create dataset and loader
    test_dataset = ProteinPairDataset(test_data, embeddings_dict)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=2
    )
    
    print(f"Dataset: {len(test_dataset)} samples")
    print(f"Batches: {len(test_loader)}")
    
    # Evaluation
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (emb_a, emb_b, lengths_a, lengths_b, interactions) in enumerate(test_loader):
            # Move to device
            emb_a = emb_a.to(device).float()
            emb_b = emb_b.to(device).float()
            lengths_a = lengths_a.to(device)
            lengths_b = lengths_b.to(device)
            
            # Forward pass
            logits = model(emb_a, emb_b, lengths_a, lengths_b)
            if logits.dim() > 1:
                logits = logits.squeeze(-1)
            
            # Get predictions and probabilities
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            # Store results
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(interactions.numpy())
            
            if batch_idx % 20 == 0:
                print(f"  Processed batch {batch_idx}/{len(test_loader)}")
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Calculate comprehensive metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    
    # Additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Same as recall
    
    # Class distribution
    class_0_count = np.sum(all_labels == 0)
    class_1_count = np.sum(all_labels == 1)
    
    # Print results
    print(f"\n📊 {dataset_name.upper()} RESULTS:")
    print(f"{'Metric':<15} {'Value':<10}")
    print("-" * 25)
    print(f"{'Accuracy':<15} {accuracy:.4f}")
    print(f"{'Precision':<15} {precision:.4f}")
    print(f"{'Recall':<15} {recall:.4f}")
    print(f"{'F1 Score':<15} {f1:.4f}")
    print(f"{'ROC AUC':<15} {auc:.4f}")
    print(f"{'Specificity':<15} {specificity:.4f}")
    print(f"{'Sensitivity':<15} {sensitivity:.4f}")
    
    print(f"\n🔢 Class Distribution:")
    print(f"  Class 0 (No Interaction): {class_0_count:,} ({class_0_count/len(all_labels)*100:.1f}%)")
    print(f"  Class 1 (Interaction):    {class_1_count:,} ({class_1_count/len(all_labels)*100:.1f}%)")
    
    print(f"\n📋 Confusion Matrix:")
    print(f"              Predicted")
    print(f"             0     1")
    print(f"Actual  0  {tn:4d}  {fp:4d}")
    print(f"        1  {fn:4d}  {tp:4d}")
    
    # Detailed classification report
    print(f"\n📈 Detailed Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=['No Interaction', 'Interaction']))
    
    return {
        'dataset_name': dataset_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'confusion_matrix': cm.tolist(),
        'class_distribution': {
            'class_0': int(class_0_count),
            'class_1': int(class_1_count)
        },
        'predictions': all_preds.tolist(),
        'probabilities': all_probs.tolist(),
        'labels': all_labels.tolist()
    }

def create_visualizations(results_test1, results_test2):
    """Create comparison visualizations"""
    print(f"\n📊 CREATING VISUALIZATIONS")
    print("=" * 30)
    
    # Ensure plots directory exists
    os.makedirs('plots', exist_ok=True)
    
    # 1. ROC Curves Comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for results, label, color in [(results_test1, 'Test1', 'blue'), (results_test2, 'Test2', 'red')]:
        fpr, tpr, _ = roc_curve(results['labels'], results['probabilities'])
        auc_score = results['auc']
        plt.plot(fpr, tpr, color=color, label=f'{label} (AUC = {auc_score:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Precision-Recall Curves
    plt.subplot(1, 2, 2)
    for results, label, color in [(results_test1, 'Test1', 'blue'), (results_test2, 'Test2', 'red')]:
        precision_vals, recall_vals, _ = precision_recall_curve(results['labels'], results['probabilities'])
        f1_score_val = results['f1']
        plt.plot(recall_vals, precision_vals, color=color, label=f'{label} (F1 = {f1_score_val:.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/fixed_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Metrics Comparison Bar Chart
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'specificity']
    test1_values = [results_test1[m] for m in metrics]
    test2_values = [results_test2[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, test1_values, width, label='Test1', alpha=0.8, color='blue')
    plt.bar(x + width/2, test2_values, width, label='Test2', alpha=0.8, color='red')
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Performance Comparison: Test1 vs Test2')
    plt.xticks(x, [m.capitalize() for m in metrics])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for i, (v1, v2) in enumerate(zip(test1_values, test2_values)):
        plt.text(i - width/2, v1 + 0.01, f'{v1:.3f}', ha='center', va='bottom', fontsize=9)
        plt.text(i + width/2, v2 + 0.01, f'{v2:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('plots/fixed_model_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Visualizations saved:")
    print("   - plots/fixed_model_comparison.png")
    print("   - plots/fixed_model_metrics_comparison.png")

def main():
    """Main testing function"""
    print("🧪 TESTING FIXED PROTEIN INTERACTION MODEL")
    print("=" * 60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs('test_results', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    try:
        # Load data
        print("\n📁 Loading data...")
        train_data, val_data, test1_data, test2_data, protein_embeddings = load_data()
        
        print(f"✅ Data loaded:")
        print(f"   - Test1: {len(test1_data)} samples")
        print(f"   - Test2: {len(test2_data)} samples") 
        print(f"   - Protein embeddings: {len(protein_embeddings)}")
        
        # Load trained model
        model_path = 'models/fixed_model_best.pth'
        model = load_trained_model(model_path, device)
        
        # Evaluate on Test1
        results_test1 = evaluate_model_on_dataset(
            model, 'test1', test1_data, protein_embeddings, device
        )
        
        # Evaluate on Test2
        results_test2 = evaluate_model_on_dataset(
            model, 'test2', test2_data, protein_embeddings, device
        )
        
        # Create comparison summary
        print(f"\n🏆 FINAL COMPARISON SUMMARY")
        print("=" * 60)
        print(f"{'Metric':<15} {'Test1':<10} {'Test2':<10} {'Difference':<12}")
        print("-" * 50)
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'specificity']
        for metric in metrics:
            val1 = results_test1[metric]
            val2 = results_test2[metric]
            diff = val1 - val2
            print(f"{metric.capitalize():<15} {val1:<10.4f} {val2:<10.4f} {diff:<+12.4f}")
        
        # Determine better performance
        test1_better = sum(1 for m in metrics if results_test1[m] > results_test2[m])
        print(f"\nTest1 performs better on {test1_better}/{len(metrics)} metrics")
        print(f"Test2 performs better on {len(metrics)-test1_better}/{len(metrics)} metrics")
        
        # Create visualizations
        create_visualizations(results_test1, results_test2)
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_summary = {
            'timestamp': timestamp,
            'model_path': model_path,
            'test1_results': results_test1,
            'test2_results': results_test2,
            'comparison': {
                'test1_better_metrics': test1_better,
                'total_metrics': len(metrics)
            }
        }
        
        results_file = f'test_results/fixed_model_evaluation_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"\n✅ TESTING COMPLETED!")
        print(f"📁 Detailed results saved to: {results_file}")
        print(f"📊 Visualizations saved to: plots/")
        
        # Print key takeaways
        avg_auc = (results_test1['auc'] + results_test2['auc']) / 2
        print(f"\n🎯 KEY TAKEAWAYS:")
        print(f"   - Average AUC across test sets: {avg_auc:.4f}")
        print(f"   - Model shows {'consistent' if abs(results_test1['auc'] - results_test2['auc']) < 0.05 else 'variable'} performance")
        print(f"   - Best performing dataset: {'Test1' if results_test1['auc'] > results_test2['auc'] else 'Test2'}")
        
        if avg_auc > 0.75:
            print("   ✅ EXCELLENT: Model performs very well on both test sets!")
        elif avg_auc > 0.65:
            print("   ✅ GOOD: Model shows solid performance")
        else:
            print("   ⚠️  Model may need further improvement")
            
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()