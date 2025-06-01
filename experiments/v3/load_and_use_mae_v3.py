#!/usr/bin/env python3
"""
Script to load and use the trained MAE model for protein pair embedding extraction
"""
import torch
import pickle
import numpy as np
import pandas as pd
from src.mask_autoencoder.v3 import TransformerMAE, ProteinPairDataset, extract_embeddings_for_classification
from torch.utils.data import DataLoader

def load_trained_mae_model(model_path, device='cuda'):
    """
    Load the trained MAE model from checkpoint
    
    Args:
        model_path: Path to the .pth model file
        device: Device to load the model on ('cuda' or 'cpu')
    
    Returns:
        model: Loaded TransformerMAE model
        checkpoint_info: Dictionary with training information
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Initialize model with same parameters as training
    model = TransformerMAE(
        input_dim=960,
        embed_dim=256,      # Same as in training script
        mask_ratio=0.5,
        num_layers=2,       # Same as in training script  
        nhead=8,           # Same as in training script
        ff_dim=512,        # Same as in training script
        max_len=2000       # Same as in training script
    ).to(device)
    
    # Load the trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode
    
    # Extract checkpoint information
    checkpoint_info = {
        'epoch': checkpoint['epoch'],
        'train_loss': checkpoint['train_loss'],
        'val_loss': checkpoint['val_loss']
    }
    
    print(f"Model loaded successfully!")
    print(f"Training epoch: {checkpoint_info['epoch']}")
    print(f"Training loss: {checkpoint_info['train_loss']:.4f}")
    print(f"Validation loss: {checkpoint_info['val_loss']:.4f}")
    
    return model, checkpoint_info

def extract_embeddings_from_data(model, pairs_df, embeddings_dict, device='cuda', max_len=2000):
    """
    Extract MAE embeddings for protein pairs
    
    Args:
        model: Trained TransformerMAE model
        pairs_df: DataFrame with protein pairs
        embeddings_dict: Dictionary with protein embeddings
        device: Device for computation
        max_len: Maximum sequence length
    
    Returns:
        mae_embeddings: Numpy array of shape (n_samples, 960)
        interactions: Numpy array of interaction labels
    """
    # Create dataset
    dataset = ProteinPairDataset(pairs_df, embeddings_dict, max_len=max_len)
    
    # Extract embeddings
    mae_embeddings, interactions = extract_embeddings_for_classification(
        model, dataset, device, batch_size=8
    )
    
    return mae_embeddings, interactions

def example_usage():
    """Example of how to use the trained MAE model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load the trained model
    model_path = "experiments/v3/MAE_v3/mae_pairs_best_20250530-151013.pth"
    model, info = load_trained_mae_model(model_path, device)
    
    # 2. Load your data (you can modify these paths as needed)
    print("\nLoading data...")
    try:
        # Load validation data for example
        with open('data/validation_data.pkl', 'rb') as f:
            val_data = pickle.load(f)
        
        # Load embeddings
        with open('../ESM/embeddings_standardized.pkl', 'rb') as f:
            embeddings_dict = pickle.load(f)
        
        print(f"Loaded {len(val_data)} validation pairs")
        print(f"Loaded embeddings for {len(embeddings_dict)} proteins")
        
        # 3. Extract MAE embeddings for a subset (first 100 samples for example)
        subset_data = val_data.head(100)
        print(f"\nExtracting MAE embeddings for {len(subset_data)} protein pairs...")
        
        mae_embeddings, interactions = extract_embeddings_from_data(
            model, subset_data, embeddings_dict, device
        )
        
        print(f"Extracted embeddings shape: {mae_embeddings.shape}")
        print(f"Embedding dimension: {mae_embeddings.shape[1]}")
        print(f"First embedding sample: {mae_embeddings[0][:10]}...")  # Show first 10 values
        
        # 4. Save the extracted embeddings (optional)
        output_path = "mae_embeddings_sample.npz"
        np.savez(output_path, 
                embeddings=mae_embeddings, 
                interactions=interactions)
        print(f"Embeddings saved to {output_path}")
        
    except FileNotFoundError as e:
        print(f"Data files not found: {e}")
        print("Make sure the data files exist or modify the paths accordingly")

def get_embedding_for_single_pair(model, protein_a_id, protein_b_id, embeddings_dict, device='cuda', max_len=2000):
    """
    Get MAE embedding for a single protein pair
    
    Args:
        model: Trained TransformerMAE model
        protein_a_id: UniProt ID for protein A
        protein_b_id: UniProt ID for protein B
        embeddings_dict: Dictionary with protein embeddings
        device: Device for computation
        max_len: Maximum sequence length
    
    Returns:
        mae_embedding: Numpy array of shape (960,) - compressed embedding
    """
    # Create a single-row DataFrame
    single_pair = pd.DataFrame({
        'uniprotID_A': [protein_a_id],
        'uniprotID_B': [protein_b_id],
        'isInteraction': [0]  # Dummy value
    })
    
    # Extract embedding
    mae_embeddings, _ = extract_embeddings_from_data(
        model, single_pair, embeddings_dict, device, max_len
    )
    
    return mae_embeddings[0]  # Return single embedding

if __name__ == "__main__":
    example_usage() 