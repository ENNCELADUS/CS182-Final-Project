import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import pickle
from sklearn.metrics import make_scorer, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import seaborn as sns
import torch
import torch.nn as nn
import gc
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from torch.utils.data import DataLoader, Dataset
import sys

# Add the project root to Python path for imports
project_root = os.path.abspath('.')
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.mask_autoencoder.v3 import TransformerMAE, extract_embeddings_for_classification

# Set CS182-Final-Project as the project root
# Since the script is in src/data_process/, we need to ensure we're at project root
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))  # Go up 3 levels from src/data_process/file.py
os.chdir(project_root)

print("Setup complete. Current directory:", os.getcwd())
print("Project root set to:", project_root)

def analyze_protein_embeddings(protein_embeddings):
    """
    Analyze protein embeddings to find the maximum length
    
    Args:
        protein_embeddings: Dictionary of protein embeddings
    
    Returns:
        max_protein_len: Maximum length of individual protein embeddings
        recommended_max_len: Recommended max_len for paired embeddings (2*max + buffer)
    """
    print("Analyzing protein embedding dimensions...")
    
    lengths = []
    embedding_shapes = []
    embedding_types = {"1d_960": 0, "1d_flattened": 0, "2d_sequence": 0, "other": 0}
    sample_size = min(500, len(protein_embeddings))  # Analyze more samples for better statistics
    sample_proteins = list(protein_embeddings.keys())[:sample_size]
    
    for protein_id in tqdm(sample_proteins, desc="Analyzing protein lengths"):
        embedding = protein_embeddings[protein_id]
        
        # Handle different embedding formats
        if isinstance(embedding, np.ndarray):
            embedding_shapes.append(embedding.shape)
            
            if embedding.ndim == 1:
                if len(embedding) == 960:
                    # This seems to be an averaged/pooled embedding, not sequence-level
                    lengths.append(1)
                    embedding_types["1d_960"] += 1
                elif len(embedding) % 960 == 0:
                    # Assume it's flattened (seq_len * embed_dim)
                    seq_len = len(embedding) // 960
                    lengths.append(seq_len)
                    embedding_types["1d_flattened"] += 1
                else:
                    # Unknown format, assume it's a single embedding
                    lengths.append(1)
                    embedding_types["other"] += 1
                    
            elif embedding.ndim == 2:
                # 2D format: (seq_len, embed_dim)
                lengths.append(embedding.shape[0])
                embedding_types["2d_sequence"] += 1
            else:
                print(f"Warning: Unexpected embedding shape for {protein_id}: {embedding.shape}")
                embedding_types["other"] += 1
                continue
        else:
            print(f"Warning: Unexpected embedding type for {protein_id}: {type(embedding)}")
            embedding_types["other"] += 1
            continue
    
    if not lengths:
        print("Error: Could not determine protein lengths")
        return None, None
    
    # Calculate statistics
    max_protein_len = max(lengths)
    min_protein_len = min(lengths)
    avg_protein_len = np.mean(lengths)
    median_protein_len = np.median(lengths)
    percentile_95 = np.percentile(lengths, 95)
    percentile_99 = np.percentile(lengths, 99)
    
    print(f"Protein length statistics (from {len(lengths)} proteins):")
    print(f"  Min length: {min_protein_len}")
    print(f"  Max length: {max_protein_len}")
    print(f"  Average length: {avg_protein_len:.2f}")
    print(f"  Median length: {median_protein_len:.2f}")
    print(f"  95th percentile: {percentile_95:.2f}")
    print(f"  99th percentile: {percentile_99:.2f}")
    
    # Show embedding type distribution
    print(f"Embedding type distribution:")
    for etype, count in embedding_types.items():
        if count > 0:
            print(f"  {etype}: {count}")
    
    # Show some example embedding shapes
    unique_shapes = list(set(embedding_shapes[:10]))
    print(f"Example embedding shapes: {unique_shapes[:5]}")
    
    # Handle different cases for max_len recommendation
    if max_protein_len == 1 and embedding_types["1d_960"] > 0:
        # Most embeddings are averaged/pooled - these are not sequence-level
        print("Detected averaged/pooled protein embeddings (not sequence-level)")
        print("For paired embeddings, a moderate max_len should be sufficient")
        recommended_max_len = 100  # Conservative for averaged embeddings
        print(f"Recommended max_len for averaged embeddings: {recommended_max_len}")
    else:
        # For paired embeddings, we need to accommodate concatenation of two proteins
        # Use 99th percentile instead of max to avoid outliers, then add buffer
        conservative_max = int(percentile_99)
        recommended_max_len = int((2 * conservative_max) * 1.3)  # 30% buffer
        absolute_max_len = int((2 * max_protein_len) * 1.1)    # 10% buffer for absolute max
        
        print(f"Recommended max_len options:")
        print(f"  Conservative (99th percentile): {recommended_max_len}")
        print(f"  Absolute maximum: {absolute_max_len}")
    
    return max_protein_len, recommended_max_len

def load_trained_mae_model(model_path, max_len=2000, device='cuda'):
    """
    Load the trained MAE model from checkpoint
    
    Args:
        model_path: Path to the .pth model file
        max_len: Maximum length for protein pair sequences
        device: Device to load the model on ('cuda' or 'cpu')
    
    Returns:
        model: Loaded TransformerMAE model
        checkpoint_info: Dictionary with training information
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Try to get the original max_len from checkpoint if available
    original_max_len = 2000  # Default fallback
    if 'model_config' in checkpoint and 'max_len' in checkpoint['model_config']:
        original_max_len = checkpoint['model_config']['max_len']
        print(f"Found original model max_len in checkpoint: {original_max_len}")
    else:
        print(f"Original model max_len not found in checkpoint, using default: {original_max_len}")
    
    # Decide which max_len to use
    if max_len > original_max_len:
        print(f"Warning: Requested max_len ({max_len}) > original model max_len ({original_max_len})")
        print(f"This may cause issues with positional encodings. Using original max_len: {original_max_len}")
        final_max_len = original_max_len
    else:
        print(f"Using requested max_len: {max_len} (within original model capacity)")
        final_max_len = max_len
    
    # Initialize model with original max_len first to match checkpoint
    model = TransformerMAE(
        input_dim=960,
        embed_dim=256,      # Same as in training script
        mask_ratio=0.5,
        num_layers=2,       # Same as in training script  
        nhead=8,           # Same as in training script
        ff_dim=512,        # Same as in training script
        max_len=original_max_len     # Use original max_len to match checkpoint
    ).to(device)
    
    # Load the trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # If we want a smaller max_len, truncate the positional embeddings
    if final_max_len < original_max_len:
        print(f"Truncating positional embeddings from {original_max_len} to {final_max_len}")
        with torch.no_grad():
            # Truncate positional embeddings
            if hasattr(model, 'pos_embed') and model.pos_embed is not None:
                model.pos_embed = nn.Parameter(model.pos_embed[:, :final_max_len, :])
            
            # Update the model's max_len attribute if it exists
            if hasattr(model, 'max_len'):
                model.max_len = final_max_len
    
    model.eval()  # Set to evaluation mode
    
    # Extract checkpoint information
    checkpoint_info = {
        'epoch': checkpoint['epoch'],
        'train_loss': checkpoint['train_loss'],
        'val_loss': checkpoint['val_loss'],
        'original_max_len': original_max_len,
        'final_max_len': final_max_len
    }
    
    print(f"Model loaded successfully!")
    print(f"Training epoch: {checkpoint_info['epoch']}")
    print(f"Training loss: {checkpoint_info['train_loss']:.4f}")
    print(f"Validation loss: {checkpoint_info['val_loss']:.4f}")
    print(f"Model max_len set to: {final_max_len}")
    
    return model, checkpoint_info

# Load test datasets
print("Loading test datasets...")
test1_data = pickle.load(open('data/full_dataset/test1_data.pkl', 'rb'))
test2_data = pickle.load(open('data/full_dataset/test2_data.pkl', 'rb'))

# Load protein embeddings (used by MAE model)
print("Loading protein embeddings...")
protein_embeddings = pickle.load(open('data/full_dataset/embeddings/compressed_protein_features_v2.pkl', 'rb'))

print(f"Test1 data: {len(test1_data)} protein pairs")
print(f"Test2 data: {len(test2_data)} protein pairs")
print(f"Loaded embeddings for {len(protein_embeddings)} proteins")
print(f"Test1 columns: {test1_data.columns.tolist()}")
print(f"Test2 columns: {test2_data.columns.tolist()}")

# Analyze protein embeddings to determine optimal max_len
max_protein_len, recommended_max_len = analyze_protein_embeddings(protein_embeddings)

# Use the calculated max_len or fallback to 2000 if analysis fails
if recommended_max_len is not None:
    optimal_max_len = recommended_max_len
    print(f"Using calculated max_len: {optimal_max_len}")
else:
    optimal_max_len = 2000
    print(f"Falling back to default max_len: {optimal_max_len}")

# Additional safety check - ensure we don't go beyond reasonable limits
if optimal_max_len > 4000:
    print(f"Warning: Calculated max_len ({optimal_max_len}) is very large.")
    print(f"This may cause memory issues. Using conservative limit of 4000.")
    optimal_max_len = 4000
elif optimal_max_len > 6000:
    print(f"Warning: Calculated max_len ({optimal_max_len}) is extremely large.")
    print(f"This will likely cause out-of-memory errors. Using safe limit of 2000.")
    optimal_max_len = 2000

print(f"Final max_len decision: {optimal_max_len}")

# Load the trained MAE model with optimal max_len
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model_path = "experiments/v3/mae_pairs_best_20250530-151013.pth"
mae_model, info = load_trained_mae_model(model_path, optimal_max_len, device)

class ProteinPairDataset1D(Dataset):
    """
    Custom Dataset for protein pairs with 1D averaged embeddings
    This handles the case where embeddings are shape (960,) instead of (seq_len, 960)
    """
    def __init__(self, pairs_df, embeddings_dict, max_len=100):
        self.pairs_df = pairs_df
        self.embeddings_dict = embeddings_dict
        self.max_len = max_len
        
        # Filter valid pairs and store the actual rows
        self.valid_pairs = []
        for i, row in pairs_df.iterrows():
            if row['uniprotID_A'] in embeddings_dict and row['uniprotID_B'] in embeddings_dict:
                self.valid_pairs.append(row)
        
        print(f"Dataset: {len(self.valid_pairs)} valid pairs out of {len(pairs_df)} total pairs")
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        row = self.valid_pairs[idx]
        
        # Get embeddings for both proteins
        emb_A = torch.from_numpy(self.embeddings_dict[row['uniprotID_A']])  # (960,)
        emb_B = torch.from_numpy(self.embeddings_dict[row['uniprotID_B']])  # (960,)
        
        # Handle 1D embeddings by converting to 2D
        if emb_A.dim() == 1:
            emb_A = emb_A.unsqueeze(0)  # (1, 960)
        if emb_B.dim() == 1:
            emb_B = emb_B.unsqueeze(0)  # (1, 960)
        
        # Combine the embeddings
        combined_emb = torch.cat([emb_A, emb_B], dim=0)  # (2, 960)
        
        # Pad to max_len if needed
        seq_len = combined_emb.shape[0]
        if seq_len < self.max_len:
            pad_size = (self.max_len - seq_len, 960)
            padding = torch.zeros(pad_size, dtype=combined_emb.dtype)
            combined_emb = torch.cat([combined_emb, padding], dim=0)
        elif seq_len > self.max_len:
            combined_emb = combined_emb[:self.max_len]
            seq_len = self.max_len
        
        return {
            "seq": combined_emb.clone(),        # (max_len, 960)
            "padding_start": seq_len,           # int - actual length
            "uniprotID_A": row['uniprotID_A'],  # string
            "uniprotID_B": row['uniprotID_B'],  # string
            "isInteraction": row['isInteraction'] if 'isInteraction' in row else -1  # int
        }

def collate_fn_1d(batch):
    """Custom collate function for 1D embeddings"""
    seqs = torch.stack([item["seq"] for item in batch], dim=0)              # (B, L, 960)
    lengths = torch.tensor([item["padding_start"] for item in batch])        # (B,)
    interactions = torch.tensor([item["isInteraction"] for item in batch])   # (B,)
    return seqs, lengths, interactions

def extract_embeddings_from_data(model, pairs_df, embeddings_dict, max_len, device='cuda'):
    """Extract MAE embeddings for protein pairs using 1D averaged embeddings"""
    # Create dataset with 1D embedding support
    dataset = ProteinPairDataset1D(pairs_df, embeddings_dict, max_len=max_len)
    
    # Extract embeddings using custom dataloader
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn_1d)
    
    mae_embeddings = []
    interactions = []
    
    model.eval()
    with torch.no_grad():
        for batch_seqs, batch_lengths, batch_interactions in tqdm(dataloader, desc="Extracting embeddings"):
            batch_seqs = batch_seqs.to(device).float()
            batch_lengths = batch_lengths.to(device)
            
            # Forward pass through MAE model
            _, compressed, _ = model(batch_seqs, batch_lengths)
            
            mae_embeddings.append(compressed.cpu())
            interactions.append(batch_interactions)
    
    # Concatenate all batches
    mae_embeddings = torch.cat(mae_embeddings, dim=0).numpy()
    interactions = torch.cat(interactions, dim=0).numpy()
    
    return mae_embeddings, interactions

# Extract embeddings for test1_data
print("Extracting MAE embeddings for test1_data...")
test1_embeddings, test1_interactions = extract_embeddings_from_data(
    mae_model, test1_data, protein_embeddings, optimal_max_len, device
)

print(f"Test1 embeddings shape: {test1_embeddings.shape}")
print(f"Test1 interactions shape: {test1_interactions.shape}")

# Extract embeddings for test2_data  
print("\nExtracting MAE embeddings for test2_data...")
test2_embeddings, test2_interactions = extract_embeddings_from_data(
    mae_model, test2_data, protein_embeddings, optimal_max_len, device
)

print(f"Test2 embeddings shape: {test2_embeddings.shape}")
print(f"Test2 interactions shape: {test2_interactions.shape}")

# Save test1 embeddings
test1_data_with_mae = {
    'embeddings': test1_embeddings,
    'interactions': test1_interactions,
    'original_data': test1_data
}

test1_output_path = 'data/full_dataset/pair_embeddings/test1_data_with_mae_embeddings.pkl'
print(f"Saving test1 data with MAE embeddings to {test1_output_path}")
with open(test1_output_path, 'wb') as f:
    pickle.dump(test1_data_with_mae, f)

# Save test2 embeddings
test2_data_with_mae = {
    'embeddings': test2_embeddings,
    'interactions': test2_interactions,
    'original_data': test2_data
}

test2_output_path = 'data/full_dataset/pair_embeddings/test2_data_with_mae_embeddings.pkl'
print(f"Saving test2 data with MAE embeddings to {test2_output_path}")
with open(test2_output_path, 'wb') as f:
    pickle.dump(test2_data_with_mae, f)

print("Test embeddings saved successfully!")

# Clean up GPU memory
del mae_model
torch.cuda.empty_cache()
gc.collect()