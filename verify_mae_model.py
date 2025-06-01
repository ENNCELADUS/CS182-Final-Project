#!/usr/bin/env python3
"""
Simple script to verify the trained MAE model can be loaded and inspect its architecture
"""
import torch
from src.mask_autoencoder.v3 import TransformerMAE

def verify_mae_model():
    """Verify the trained MAE model can be loaded correctly"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    model_path = "experiments/v3/MAE_v3/mae_pairs_best_20250530-151013.pth"
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    print("✅ Model file loaded successfully!")
    print(f"📊 Training Information:")
    print(f"   - Epoch: {checkpoint['epoch']}")
    print(f"   - Training Loss: {checkpoint['train_loss']:.4f}")
    print(f"   - Validation Loss: {checkpoint['val_loss']:.4f}")
    
    # Initialize model with correct parameters
    model = TransformerMAE(
        input_dim=960,
        embed_dim=256,      # Same as training
        mask_ratio=0.5,
        num_layers=2,       # Same as training
        nhead=8,           # Same as training
        ff_dim=512,        # Same as training
        max_len=2000       # Same as training
    ).to(device)
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("✅ Model weights loaded successfully!")
    
    # Display model architecture
    print(f"\n🏗️ Model Architecture:")
    print(f"   - Input dimension: 960 (ESM embeddings)")
    print(f"   - Embedding dimension: 256")
    print(f"   - Number of transformer layers: 2")
    print(f"   - Number of attention heads: 8")
    print(f"   - Feed-forward dimension: 512")
    print(f"   - Maximum sequence length: 2000")
    print(f"   - Mask ratio: 50%")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📈 Model Statistics:")
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Trainable parameters: {trainable_params:,}")
    print(f"   - Model size: ~{total_params * 4 / (1024**2):.1f} MB (float32)")
    
    # Test forward pass with dummy data
    print(f"\n🧪 Testing forward pass...")
    batch_size = 2
    seq_len = 2000
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, seq_len, 960).to(device)
    dummy_lengths = torch.tensor([1500, 1800]).to(device)  # Valid sequence lengths
    
    with torch.no_grad():
        recon, compressed, mask_bool = model(dummy_input, dummy_lengths)
    
    print(f"✅ Forward pass successful!")
    print(f"   - Input shape: {dummy_input.shape}")
    print(f"   - Reconstruction shape: {recon.shape}")
    print(f"   - Compressed embedding shape: {compressed.shape}")
    print(f"   - Mask shape: {mask_bool.shape}")
    print(f"   - Masked positions: {mask_bool.sum().item()}/{mask_bool.numel()}")
    
    return model, checkpoint

if __name__ == "__main__":
    model, checkpoint = verify_mae_model()
    print(f"\n🎉 Model verification completed successfully!")
    print(f"Your MAE model is ready to use for embedding extraction!") 