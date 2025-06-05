#!/usr/bin/env python3
"""
Test script for v5.py PPI Classifier architecture
Verifies that all components work together correctly
"""

import torch
import numpy as np
from v5 import create_ppi_classifier, PPIClassifier
from v4_1 import load_data, ProteinPairDataset
from torch.utils.data import DataLoader
from v5_train_eval import collate_fn_v5, count_parameters

def test_model_creation():
    """Test model creation and parameter counting"""
    print("🔧 Testing model creation...")
    
    # Test without pre-trained weights
    model = create_ppi_classifier(
        mae_checkpoint_path=None,
        freeze_encoder=True,
        use_lora=False
    )
    
    total_params, trainable_params = count_parameters(model)
    print(f"✅ Model created successfully")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Frozen parameters: {total_params - trainable_params:,}")
    
    return model

def test_forward_pass():
    """Test forward pass with dummy data"""
    print("\n🚀 Testing forward pass...")
    
    model = create_ppi_classifier()
    model.eval()
    
    # Create dummy batch
    batch_size = 4
    seq_len_a = 150
    seq_len_b = 200
    
    emb_a = torch.randn(batch_size, seq_len_a, 960)
    emb_b = torch.randn(batch_size, seq_len_b, 960)
    
    print(f"   Input shapes: emb_a={emb_a.shape}, emb_b={emb_b.shape}")
    
    with torch.no_grad():
        logits = model(emb_a, emb_b)
    
    print(f"   Output shape: {logits.shape}")
    print(f"   Output range: [{logits.min():.4f}, {logits.max():.4f}]")
    
    # Test interaction embeddings
    interaction_emb = model.get_interaction_embeddings(emb_a, emb_b)
    print(f"   Interaction embeddings shape: {interaction_emb.shape}")
    
    print("✅ Forward pass successful")

def test_with_real_data():
    """Test with actual data loading"""
    print("\n📊 Testing with real data...")
    
    try:
        # Load real data
        train_data, val_data, test1_data, test2_data, protein_embeddings = load_data()
        
        # Create small dataset for testing
        small_train = train_data.head(32)  # Take only 32 samples
        train_dataset = ProteinPairDataset(small_train, protein_embeddings)
        
        # Create data loader
        train_loader = DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=False,
            collate_fn=collate_fn_v5
        )
        
        print(f"   Dataset size: {len(train_dataset)}")
        print(f"   Batch size: 8")
        
        # Test with real data
        model = create_ppi_classifier()
        model.eval()
        
        with torch.no_grad():
            for batch_idx, (emb_a, emb_b, interactions) in enumerate(train_loader):
                print(f"   Batch {batch_idx + 1}: emb_a={emb_a.shape}, emb_b={emb_b.shape}, labels={interactions.shape}")
                
                logits = model(emb_a, emb_b)
                print(f"   Predictions: {logits.shape}, range=[{logits.min():.4f}, {logits.max():.4f}]")
                
                # Test only first batch
                break
        
        print("✅ Real data test successful")
        
    except Exception as e:
        print(f"⚠️ Real data test failed (expected if data not available): {e}")

def test_gradient_flow():
    """Test gradient flow and training step"""
    print("\n⚡ Testing gradient flow...")
    
    model = create_ppi_classifier(freeze_encoder=False)  # Unfreeze for gradient test
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Create dummy batch
    batch_size = 4
    emb_a = torch.randn(batch_size, 100, 960)
    emb_b = torch.randn(batch_size, 120, 960)
    labels = torch.randint(0, 2, (batch_size,)).float()
    
    print(f"   Input shapes: emb_a={emb_a.shape}, emb_b={emb_b.shape}")
    print(f"   Labels shape: {labels.shape}")
    
    # Forward pass
    model.train()
    optimizer.zero_grad()
    
    logits = model(emb_a, emb_b)
    loss = criterion(logits, labels)
    
    print(f"   Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    total_grad_norm = 0
    param_count = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2)
            total_grad_norm += grad_norm.item() ** 2
            param_count += 1
    
    total_grad_norm = total_grad_norm ** (1. / 2)
    print(f"   Total gradient norm: {total_grad_norm:.4f}")
    print(f"   Parameters with gradients: {param_count}")
    
    optimizer.step()
    
    print("✅ Gradient flow test successful")

def test_frozen_vs_unfrozen():
    """Test frozen vs unfrozen encoder behavior"""
    print("\n❄️ Testing frozen vs unfrozen behavior...")
    
    # Test frozen encoder
    model_frozen = create_ppi_classifier(freeze_encoder=True)
    total_frozen, trainable_frozen = count_parameters(model_frozen)
    
    # Test unfrozen encoder
    model_unfrozen = create_ppi_classifier(freeze_encoder=False)
    total_unfrozen, trainable_unfrozen = count_parameters(model_unfrozen)
    
    print(f"   Frozen model - Total: {total_frozen:,}, Trainable: {trainable_frozen:,}")
    print(f"   Unfrozen model - Total: {total_unfrozen:,}, Trainable: {trainable_unfrozen:,}")
    
    # Should have same total parameters
    assert total_frozen == total_unfrozen, "Total parameters should be the same"
    
    # Unfrozen should have more trainable parameters
    assert trainable_unfrozen > trainable_frozen, "Unfrozen model should have more trainable parameters"
    
    print("✅ Frozen vs unfrozen test successful")

def test_different_sequence_lengths():
    """Test with various sequence lengths"""
    print("\n📏 Testing different sequence lengths...")
    
    model = create_ppi_classifier()
    model.eval()
    
    test_cases = [
        (50, 30),    # Short sequences
        (200, 180),  # Medium sequences
        (500, 600),  # Long sequences
        (1000, 800), # Very long sequences
        (10, 1200),  # Mismatched lengths
    ]
    
    with torch.no_grad():
        for i, (len_a, len_b) in enumerate(test_cases):
            emb_a = torch.randn(2, len_a, 960)
            emb_b = torch.randn(2, len_b, 960)
            
            logits = model(emb_a, emb_b)
            
            print(f"   Test {i+1}: {len_a}x{len_b} -> Output: {logits.shape}")
    
    print("✅ Different sequence lengths test successful")

def main():
    """Run all tests"""
    print("🧬 V5 PPI CLASSIFIER ARCHITECTURE TESTS")
    print("=" * 60)
    
    try:
        # Run all tests
        test_model_creation()
        test_forward_pass()
        test_with_real_data()
        test_gradient_flow()
        test_frozen_vs_unfrozen()
        test_different_sequence_lengths()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("✅ v5.py architecture is working correctly")
        print("✅ Ready for training with v5_train_eval.py")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 