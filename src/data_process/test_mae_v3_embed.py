#!/usr/bin/env python3
"""
Test script for mae_v3_embed.py functionality
This script tests the embedding analysis and max_len calculation
"""

import pickle
import numpy as np
import os
import sys
from tqdm.auto import tqdm

# Add the project root to Python path
sys.path.append(os.path.abspath('.'))

def create_test_protein_embeddings():
    """Create test protein embeddings with various formats and lengths"""
    
    test_embeddings = {}
    
    # Case 1: Short proteins (1D format, 960 dim - averaged)
    for i in range(10):
        test_embeddings[f'short_protein_{i}'] = np.random.randn(960)
    
    # Case 2: Medium proteins (2D format)
    for i in range(20):
        seq_len = np.random.randint(50, 200)
        test_embeddings[f'medium_protein_{i}'] = np.random.randn(seq_len, 960)
    
    # Case 3: Long proteins (2D format)
    for i in range(10):
        seq_len = np.random.randint(300, 800)
        test_embeddings[f'long_protein_{i}'] = np.random.randn(seq_len, 960)
    
    # Case 4: Very long proteins (edge case)
    for i in range(3):
        seq_len = np.random.randint(1000, 1500)
        test_embeddings[f'very_long_protein_{i}'] = np.random.randn(seq_len, 960)
    
    # Case 5: Flattened embeddings (1D but not 960)
    for i in range(5):
        seq_len = np.random.randint(50, 100)
        flattened_size = seq_len * 960
        test_embeddings[f'flattened_protein_{i}'] = np.random.randn(flattened_size)
    
    return test_embeddings

def analyze_protein_embeddings(protein_embeddings):
    """
    Test version of the analyze_protein_embeddings function
    """
    print("Analyzing protein embedding dimensions...")
    
    lengths = []
    embedding_shapes = []
    embedding_types = {"1d_960": 0, "1d_flattened": 0, "2d_sequence": 0, "other": 0}
    sample_size = min(500, len(protein_embeddings))
    sample_proteins = list(protein_embeddings.keys())[:sample_size]
    
    for protein_id in tqdm(sample_proteins, desc="Analyzing protein lengths"):
        embedding = protein_embeddings[protein_id]
        
        if isinstance(embedding, np.ndarray):
            embedding_shapes.append(embedding.shape)
            
            if embedding.ndim == 1:
                if len(embedding) == 960:
                    lengths.append(1)
                    embedding_types["1d_960"] += 1
                elif len(embedding) % 960 == 0:
                    seq_len = len(embedding) // 960
                    lengths.append(seq_len)
                    embedding_types["1d_flattened"] += 1
                else:
                    lengths.append(1)
                    embedding_types["other"] += 1
                    
            elif embedding.ndim == 2:
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
        conservative_max = int(percentile_99)
        recommended_max_len = int((2 * conservative_max) * 1.3)  # 30% buffer
        absolute_max_len = int((2 * max_protein_len) * 1.1)    # 10% buffer for absolute max
        
        print(f"Recommended max_len options:")
        print(f"  Conservative (99th percentile): {recommended_max_len}")
        print(f"  Absolute maximum: {absolute_max_len}")
    
    return max_protein_len, recommended_max_len

def test_max_len_calculation():
    """Test the max_len calculation with various scenarios"""
    
    print("Creating test protein embeddings...")
    test_embeddings = create_test_protein_embeddings()
    
    print(f"Created {len(test_embeddings)} test protein embeddings")
    
    # Test the analysis function
    max_protein_len, recommended_max_len = analyze_protein_embeddings(test_embeddings)
    
    print(f"\nTest Results:")
    print(f"Max individual protein length: {max_protein_len}")
    print(f"Recommended max_len for pairs: {recommended_max_len}")
    
    # Apply safety checks
    optimal_max_len = recommended_max_len
    
    if optimal_max_len > 4000:
        print(f"Warning: Calculated max_len ({optimal_max_len}) is very large.")
        print(f"This may cause memory issues. Using conservative limit of 4000.")
        optimal_max_len = 4000
    elif optimal_max_len > 6000:
        print(f"Warning: Calculated max_len ({optimal_max_len}) is extremely large.")
        print(f"This will likely cause out-of-memory errors. Using safe limit of 2000.")
        optimal_max_len = 2000
    
    print(f"Final max_len decision: {optimal_max_len}")
    
    return optimal_max_len

if __name__ == "__main__":
    print("Testing MAE v3 embedding analysis...")
    
    # Test with synthetic data
    test_max_len = test_max_len_calculation()
    
    print(f"\nTest completed successfully!")
    print(f"Recommended max_len: {test_max_len}")
    
    # If real data exists, test with it too
    real_embeddings_path = 'data/full_dataset/embeddings/compressed_protein_features_v2.pkl'
    if os.path.exists(real_embeddings_path):
        print(f"\nTesting with real protein embeddings...")
        try:
            real_embeddings = pickle.load(open(real_embeddings_path, 'rb'))
            print(f"Loaded {len(real_embeddings)} real protein embeddings")
            
            real_max_len, real_recommended = analyze_protein_embeddings(real_embeddings)
            print(f"Real data max protein length: {real_max_len}")
            print(f"Real data recommended max_len: {real_recommended}")
            
        except Exception as e:
            print(f"Could not load real embeddings: {e}")
    else:
        print(f"Real embeddings file not found at {real_embeddings_path}") 