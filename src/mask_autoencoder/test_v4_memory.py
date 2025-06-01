#!/usr/bin/env python3
"""
Memory and Performance Test for Enhanced Protein Interaction Prediction v4
Tests memory usage, data loading, and training readiness for server deployment
"""

import os
import sys
import torch
import psutil
import gc
import tracemalloc
import traceback
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import json
import time
from torch.utils.data import DataLoader

# Add parent directory to path to import v4 modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from v4 import (
        ProteinPairDataset, ProteinInteractionClassifier, 
        collate_fn, load_data, detect_column_names
    )
except ImportError as e:
    print(f"Error importing v4 modules: {e}")
    print("Make sure v4.py is in the same directory")
    sys.exit(1)


def create_mock_data():
    """Create mock data for testing when real data is not available"""
    print("📝 Creating mock data for testing...")
    
    # Create mock protein embeddings
    protein_ids = [f"P{i:05d}" for i in range(100)]  # 100 mock proteins
    protein_embeddings = {}
    
    np.random.seed(42)  # For reproducible results
    for protein_id in protein_ids:
        # Random sequence length between 50 and 500
        seq_len = np.random.randint(50, 501)
        # Random embedding (seq_len, 960) - ESM-C dimension
        embedding = np.random.randn(seq_len, 960).astype(np.float32)
        protein_embeddings[protein_id] = embedding
    
    # Create mock interaction data
    def create_mock_pairs(n_pairs, protein_ids):
        data = []
        for i in range(n_pairs):
            protein_a = np.random.choice(protein_ids)
            protein_b = np.random.choice(protein_ids)
            # Random interaction label
            interaction = np.random.randint(0, 2)
            data.append({
                'protein_id_a': protein_a,
                'protein_id_b': protein_b, 
                'isInteraction': interaction
            })
        return pd.DataFrame(data)
    
    # Create datasets
    train_data = create_mock_pairs(1000, protein_ids)
    cv_data = create_mock_pairs(200, protein_ids)
    test1_data = create_mock_pairs(200, protein_ids)
    test2_data = create_mock_pairs(200, protein_ids)
    
    print(f"✅ Created mock data:")
    print(f"   - {len(protein_embeddings)} proteins")
    print(f"   - {len(train_data)} training pairs")
    print(f"   - {len(cv_data)} validation pairs")
    print(f"   - {len(test1_data)} test1 pairs")
    print(f"   - {len(test2_data)} test2 pairs")
    
    return train_data, cv_data, test1_data, test2_data, protein_embeddings


def get_memory_info():
    """Get current memory usage information"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
        'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
        'percent': process.memory_percent(),       # Percentage of total memory
        'available_mb': psutil.virtual_memory().available / 1024 / 1024
    }


def get_gpu_memory_info():
    """Get GPU memory usage if CUDA is available"""
    if not torch.cuda.is_available():
        return None
    
    gpu_info = {}
    for i in range(torch.cuda.device_count()):
        gpu_info[f'gpu_{i}'] = {
            'allocated_mb': torch.cuda.memory_allocated(i) / 1024 / 1024,
            'cached_mb': torch.cuda.memory_reserved(i) / 1024 / 1024,
            'max_allocated_mb': torch.cuda.max_memory_allocated(i) / 1024 / 1024
        }
    return gpu_info


def print_memory_stats(stage_name, start_time=None):
    """Print comprehensive memory statistics"""
    memory_info = get_memory_info()
    gpu_info = get_gpu_memory_info()
    
    print(f"\n{'='*60}")
    print(f"MEMORY STATS - {stage_name}")
    print(f"{'='*60}")
    print(f"CPU Memory (RSS): {memory_info['rss_mb']:.1f} MB")
    print(f"CPU Memory (VMS): {memory_info['vms_mb']:.1f} MB")
    print(f"Memory Percentage: {memory_info['percent']:.1f}%")
    print(f"Available Memory: {memory_info['available_mb']:.1f} MB")
    
    if gpu_info:
        for gpu_id, info in gpu_info.items():
            print(f"{gpu_id.upper()} Allocated: {info['allocated_mb']:.1f} MB")
            print(f"{gpu_id.upper()} Cached: {info['cached_mb']:.1f} MB")
            print(f"{gpu_id.upper()} Max Used: {info['max_allocated_mb']:.1f} MB")
    
    if start_time:
        elapsed = time.time() - start_time
        print(f"Elapsed Time: {elapsed:.2f} seconds")
    
    print("="*60)


def test_data_loading():
    """Test data loading and memory usage"""
    print("\n🔍 TESTING DATA LOADING...")
    start_time = time.time()
    
    try:
        # Start memory tracking
        tracemalloc.start()
        initial_memory = get_memory_info()
        
        print("Loading datasets...")
        
        # Try to load real data first
        try:
            train_data, cv_data, test1_data, test2_data, protein_embeddings = load_data()
            print("✅ Successfully loaded real data")
        except (FileNotFoundError, OSError) as e:
            print(f"⚠️ Real data not found: {e}")
            print("Using mock data for testing...")
            train_data, cv_data, test1_data, test2_data, protein_embeddings = create_mock_data()
        
        print_memory_stats("After Data Loading", start_time)
        
        # Analyze embedding sizes
        print(f"\n📊 DATA ANALYSIS:")
        print(f"Training pairs: {len(train_data):,}")
        print(f"Validation pairs: {len(cv_data):,}")
        print(f"Test1 pairs: {len(test1_data):,}")
        print(f"Test2 pairs: {len(test2_data):,}")
        print(f"Protein embeddings: {len(protein_embeddings):,}")
        
        # Analyze embedding memory usage
        total_embedding_size = 0
        max_seq_len = 0
        min_seq_len = float('inf')
        seq_lengths = []
        
        print("\nAnalyzing embedding sizes (first 100 proteins)...")
        for i, (protein_id, embedding) in enumerate(protein_embeddings.items()):
            if i >= 100:  # Only check first 100 to avoid memory issues
                break
            
            if isinstance(embedding, np.ndarray):
                size_bytes = embedding.nbytes
                seq_len = embedding.shape[0]
            else:  # torch tensor
                size_bytes = embedding.numel() * embedding.element_size()
                seq_len = embedding.shape[0]
                
            total_embedding_size += size_bytes
            seq_lengths.append(seq_len)
            max_seq_len = max(max_seq_len, seq_len)
            min_seq_len = min(min_seq_len, seq_len)
        
        avg_seq_len = np.mean(seq_lengths) if seq_lengths else 0
        estimated_total_size = (total_embedding_size / len(seq_lengths)) * len(protein_embeddings) if seq_lengths else 0
        
        print(f"\n📈 EMBEDDING STATISTICS:")
        print(f"Sample size analyzed: {len(seq_lengths)} proteins")
        print(f"Average sequence length: {avg_seq_len:.1f}")
        print(f"Max sequence length: {max_seq_len}")
        print(f"Min sequence length: {min_seq_len}")
        print(f"Estimated total embedding size: {estimated_total_size / (1024**3):.2f} GB")
        print(f"Average embedding size per protein: {(total_embedding_size / len(seq_lengths)) / (1024**2):.2f} MB" if seq_lengths else "N/A")
        
        # Memory tracking
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        print(f"\n🧠 MEMORY TRACKING:")
        print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
        print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
        
        return train_data, cv_data, test1_data, test2_data, protein_embeddings
        
    except Exception as e:
        print(f"❌ Data loading failed: {str(e)}")
        traceback.print_exc()
        return None, None, None, None, None


def test_dataset_creation(train_data, protein_embeddings):
    """Test dataset creation and memory usage"""
    print("\n🔍 TESTING DATASET CREATION...")
    start_time = time.time()
    
    try:
        print("Creating training dataset...")
        train_dataset = ProteinPairDataset(train_data, protein_embeddings)
        
        print_memory_stats("After Dataset Creation", start_time)
        
        # Test dataset access
        print(f"\n📊 DATASET STATISTICS:")
        print(f"Valid pairs: {len(train_dataset)}")
        
        # Test a few samples
        print("Testing dataset access...")
        for i in range(min(3, len(train_dataset))):
            sample = train_dataset[i]
            seq_len_a = sample['emb_a'].shape[0]
            seq_len_b = sample['emb_b'].shape[0]
            print(f"Sample {i}: Protein A length={seq_len_a}, Protein B length={seq_len_b}, Interaction={sample['interaction']}")
        
        return train_dataset
        
    except Exception as e:
        print(f"❌ Dataset creation failed: {str(e)}")
        traceback.print_exc()
        return None


def test_dataloader_memory(train_dataset, batch_sizes=[4, 8, 16, 32]):
    """Test dataloader with different batch sizes"""
    print("\n🔍 TESTING DATALOADER MEMORY USAGE...")
    
    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
        start_time = time.time()
        
        try:
            # Clear cache before each test
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            dataloader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=False,  # Don't shuffle for memory test
                collate_fn=collate_fn,
                num_workers=0,  # No multiprocessing for memory test
                pin_memory=False
            )
            
            # Test one batch
            batch = next(iter(dataloader))
            emb_a, emb_b, lengths_a, lengths_b, interactions = batch
            
            print(f"  Batch shapes: A={emb_a.shape}, B={emb_b.shape}")
            print(f"  Max lengths: A={lengths_a.max().item()}, B={lengths_b.max().item()}")
            print(f"  Memory per batch: ~{(emb_a.numel() + emb_b.numel()) * 4 / (1024**2):.1f} MB")
            
            print_memory_stats(f"Batch Size {batch_size}", start_time)
            
            # Clean up
            del batch, emb_a, emb_b, lengths_a, lengths_b, interactions, dataloader
            
        except Exception as e:
            print(f"❌ Batch size {batch_size} failed: {str(e)}")
            continue


def test_model_memory(device='cpu'):
    """Test model creation and memory usage"""
    print(f"\n🔍 TESTING MODEL MEMORY USAGE (device: {device})...")
    start_time = time.time()
    
    try:
        # Test different configurations
        configs = [
            {'name': 'Lightweight', 'encoder_layers': 8, 'encoder_embed_dim': 256},
            {'name': 'Standard', 'encoder_layers': 16, 'encoder_embed_dim': 512},
            {'name': 'Heavy', 'encoder_layers': 24, 'encoder_embed_dim': 768}
        ]
        
        for config in configs:
            print(f"\nTesting {config['name']} configuration...")
            
            # Clear memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Create model
            model = ProteinInteractionClassifier(
                encoder_embed_dim=config['encoder_embed_dim'],
                encoder_layers=config['encoder_layers'],
                encoder_heads=16,
                use_variable_length=True,
                decoder_hidden_dims=[512, 256, 128],
                dropout=0.2
            )
            
            # Move to device
            if device != 'cpu':
                model = model.to(device)
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"  Total parameters: {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")
            print(f"  Model size: ~{total_params * 4 / (1024**2):.1f} MB")
            
            print_memory_stats(f"Model {config['name']}", start_time)
            
            # Clean up
            del model
            
    except Exception as e:
        print(f"❌ Model testing failed: {str(e)}")
        traceback.print_exc()


def test_forward_pass(train_dataset, device='cpu'):
    """Test forward pass with different batch sizes"""
    print(f"\n🔍 TESTING FORWARD PASS (device: {device})...")
    
    try:
        # Create a lightweight model for testing
        model = ProteinInteractionClassifier(
            encoder_embed_dim=256,  # Smaller for testing
            encoder_layers=4,       # Fewer layers for testing
            encoder_heads=8,
            use_variable_length=True,
            decoder_hidden_dims=[256, 128, 64],
            dropout=0.2
        )
        
        if device != 'cpu':
            model = model.to(device)
        
        # Test with small batches
        for batch_size in [1, 2, 4]:
            print(f"\nTesting forward pass with batch size {batch_size}...")
            start_time = time.time()
            
            # Clear memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Create dataloader
            dataloader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0
            )
            
            # Get one batch
            batch = next(iter(dataloader))
            emb_a, emb_b, lengths_a, lengths_b, interactions = batch
            
            if device != 'cpu':
                emb_a = emb_a.to(device)
                emb_b = emb_b.to(device)
                lengths_a = lengths_a.to(device)
                lengths_b = lengths_b.to(device)
            
            # Forward pass
            with torch.no_grad():
                logits = model(emb_a, emb_b, lengths_a, lengths_b)
                print(f"  Output shape: {logits.shape}")
            
            print_memory_stats(f"Forward Pass Batch {batch_size}", start_time)
            
            # Clean up
            del batch, emb_a, emb_b, lengths_a, lengths_b, interactions, dataloader, logits
        
        del model
        
    except Exception as e:
        print(f"❌ Forward pass testing failed: {str(e)}")
        traceback.print_exc()


def test_training_step(train_dataset, device='cpu'):
    """Test a single training step"""
    print(f"\n🔍 TESTING TRAINING STEP (device: {device})...")
    
    try:
        # Create a very lightweight model for testing
        model = ProteinInteractionClassifier(
            encoder_embed_dim=128,  # Very small for testing
            encoder_layers=2,       # Very few layers for testing
            encoder_heads=4,
            use_variable_length=True,
            decoder_hidden_dims=[128, 64],
            dropout=0.2
        )
        
        if device != 'cpu':
            model = model.to(device)
        
        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        criterion = torch.nn.BCEWithLogitsLoss()
        
        # Create small dataloader
        dataloader = DataLoader(
            train_dataset, 
            batch_size=2,  # Very small batch
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        print("Testing training step...")
        start_time = time.time()
        
        # Get one batch
        batch = next(iter(dataloader))
        emb_a, emb_b, lengths_a, lengths_b, interactions = batch
        
        if device != 'cpu':
            emb_a = emb_a.to(device)
            emb_b = emb_b.to(device)
            lengths_a = lengths_a.to(device)
            lengths_b = lengths_b.to(device)
            interactions = interactions.to(device).float()
        else:
            interactions = interactions.float()
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        logits = model(emb_a, emb_b, lengths_a, lengths_b).squeeze()
        loss = criterion(logits, interactions)
        
        loss.backward()
        optimizer.step()
        
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
        
        print_memory_stats("Training Step", start_time)
        
        # Clean up
        del model, optimizer, criterion, dataloader, batch
        del emb_a, emb_b, lengths_a, lengths_b, interactions, logits, loss
        
    except Exception as e:
        print(f"❌ Training step testing failed: {str(e)}")
        traceback.print_exc()


def generate_memory_report():
    """Generate memory optimization recommendations"""
    memory_info = get_memory_info()
    gpu_info = get_gpu_memory_info()
    
    print(f"\n{'='*60}")
    print("MEMORY OPTIMIZATION RECOMMENDATIONS")
    print(f"{'='*60}")
    
    print(f"💡 GENERAL RECOMMENDATIONS:")
    print(f"1. Current available memory: {memory_info['available_mb']:.1f} MB")
    
    if memory_info['available_mb'] < 8000:  # Less than 8GB
        print(f"⚠️  WARNING: Low available memory!")
        print(f"   - Consider using fixed-length embeddings")
        print(f"   - Reduce batch size to 4-8")
        print(f"   - Use gradient accumulation")
        print(f"   - Enable mixed precision training")
    elif memory_info['available_mb'] < 16000:  # Less than 16GB
        print(f"✅ Moderate memory available")
        print(f"   - Batch size 8-16 should work")
        print(f"   - Variable-length embeddings possible")
        print(f"   - Consider gradient checkpointing")
    else:
        print(f"✅ Good memory available")
        print(f"   - Batch size 16-32 should work")
        print(f"   - Full model configuration supported")
    
    print(f"\n💡 MODEL CONFIGURATION RECOMMENDATIONS:")
    
    if memory_info['available_mb'] < 8000:
        print(f"   - encoder_layers: 8-12")
        print(f"   - encoder_embed_dim: 256-384")
        print(f"   - batch_size: 4-8")
        print(f"   - use_variable_length: False")
    elif memory_info['available_mb'] < 16000:
        print(f"   - encoder_layers: 12-16")
        print(f"   - encoder_embed_dim: 384-512")
        print(f"   - batch_size: 8-16")
        print(f"   - use_variable_length: True")
    else:
        print(f"   - encoder_layers: 16-24")
        print(f"   - encoder_embed_dim: 512-768")
        print(f"   - batch_size: 16-32")
        print(f"   - use_variable_length: True")
    
    if gpu_info:
        print(f"\n💡 GPU RECOMMENDATIONS:")
        for gpu_id, info in gpu_info.items():
            total_gpu_memory = torch.cuda.get_device_properties(int(gpu_id.split('_')[1])).total_memory / (1024**3)
            print(f"   {gpu_id.upper()}: {total_gpu_memory:.1f} GB total")
            
            if total_gpu_memory < 8:
                print(f"     - Use CPU training or very small batch sizes")
            elif total_gpu_memory < 16:
                print(f"     - Batch size 4-8, consider gradient accumulation")
            else:
                print(f"     - Batch size 8-16+ supported")
    
    print(f"\n💡 OPTIMIZATION TECHNIQUES:")
    print(f"   - Use gradient_checkpointing=True in model")
    print(f"   - Enable torch.cuda.amp for mixed precision")
    print(f"   - Use gradient accumulation for effective larger batches")
    print(f"   - Clear cache between epochs: torch.cuda.empty_cache()")
    print(f"   - Use num_workers=0 or 1 for DataLoader to reduce memory")
    print(f"   - Consider using pin_memory=False")


def main():
    """Run comprehensive memory and performance tests"""
    print("🚀 ENHANCED PROTEIN INTERACTION PREDICTION v4 - MEMORY TEST")
    print("=" * 80)
    
    # Initial memory state
    print_memory_stats("Initial State")
    
    # Detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Using device: {device}")
    if device == 'cuda':
        print(f"   GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"   GPU {i}: {props.name} ({props.total_memory / (1024**3):.1f} GB)")
    
    # Test 1: Data Loading
    train_data, cv_data, test1_data, test2_data, protein_embeddings = test_data_loading()
    
    if train_data is None:
        print("❌ Data loading failed. Cannot continue tests.")
        return
    
    # Test 2: Dataset Creation
    train_dataset = test_dataset_creation(train_data, protein_embeddings)
    
    if train_dataset is None:
        print("❌ Dataset creation failed. Cannot continue tests.")
        return
    
    # Test 3: DataLoader Memory
    test_dataloader_memory(train_dataset)
    
    # Test 4: Model Memory (CPU)
    test_model_memory('cpu')
    
    # Test 5: Model Memory (GPU if available)
    if device == 'cuda':
        test_model_memory(device)
    
    # Test 6: Forward Pass (CPU)
    test_forward_pass(train_dataset, 'cpu')
    
    # Test 7: Forward Pass (GPU if available)
    if device == 'cuda':
        test_forward_pass(train_dataset, device)
    
    # Test 8: Training Step (CPU)
    test_training_step(train_dataset, 'cpu')
    
    # Test 9: Training Step (GPU if available)
    if device == 'cuda':
        test_training_step(train_dataset, device)
    
    # Final memory report
    generate_memory_report()
    
    print(f"\n✅ MEMORY TESTING COMPLETED!")
    print(f"Check the recommendations above before starting full training.")


if __name__ == '__main__':
    main() 