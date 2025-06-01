#!/usr/bin/env python3
"""
Script to analyze the data structure of the MAE embeddings pickle file
"""
import pickle
import pandas as pd
import numpy as np

def analyze_data_structure():
    """Analyze the structure of the MAE embeddings data file"""
    
    file_path = 'data/full_dataset/pair_embeddings/train_data_with_mae_embeddings_20250530-151013.pkl'
    
    try:
        # Load the pickle file
        print(f"Loading data from: {file_path}")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print('\n=== DATA STRUCTURE ANALYSIS ===')
        print(f'Data type: {type(data)}')
        
        if isinstance(data, pd.DataFrame):
            print(f'DataFrame shape: {data.shape}')
            print(f'Columns: {list(data.columns)}')
            print(f'Memory usage: {data.memory_usage(deep=True).sum() / (1024**2):.2f} MB')
            
            print(f'\n=== FIRST FEW ROWS ===')
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            print(data.head())
            
            # Check MAE embeddings specifically
            if 'mae_embeddings' in data.columns:
                print(f'\n=== MAE EMBEDDINGS ANALYSIS ===')
                mae_emb = data['mae_embeddings'].iloc[0]
                print(f'MAE embedding type: {type(mae_emb)}')
                print(f'MAE embedding shape: {mae_emb.shape if hasattr(mae_emb, "shape") else "No shape attribute"}')
                print(f'MAE embedding dtype: {mae_emb.dtype if hasattr(mae_emb, "dtype") else "No dtype"}')
                print(f'MAE embedding sample (first 10 values): {mae_emb[:10] if hasattr(mae_emb, "__getitem__") else mae_emb}')
                
                # Check if all embeddings have the same shape
                shapes = [emb.shape for emb in data['mae_embeddings'][:10]]  # Check first 10
                print(f'First 10 embedding shapes: {shapes}')
                
                # Statistical info
                if hasattr(mae_emb, 'shape') and len(mae_emb.shape) == 1:
                    print(f'Embedding statistics:')
                    print(f'  - Min: {mae_emb.min():.4f}')
                    print(f'  - Max: {mae_emb.max():.4f}')
                    print(f'  - Mean: {mae_emb.mean():.4f}')
                    print(f'  - Std: {mae_emb.std():.4f}')
            
            # Check data types of all columns
            print(f'\n=== COLUMN DATA TYPES ===')
            for col in data.columns:
                col_type = data[col].dtype
                sample_val = data[col].iloc[0]
                print(f'{col}: {col_type} (sample: {type(sample_val)})')
            
            # Check for missing values
            print(f'\n=== MISSING VALUES ===')
            missing = data.isnull().sum()
            print(missing[missing > 0] if missing.sum() > 0 else "No missing values")
            
        elif isinstance(data, dict):
            print(f'Dictionary keys: {list(data.keys())}')
            for key, value in data.items():
                print(f'  {key}: {type(value)}, shape: {getattr(value, "shape", "no shape")}')
                
        elif isinstance(data, list):
            print(f'List length: {len(data)}')
            print(f'First element type: {type(data[0])}')
            if hasattr(data[0], 'shape'):
                print(f'First element shape: {data[0].shape}')
                
        else:
            print(f'Other data type: {type(data)}')
            print(f'Data preview: {str(data)[:200]}...')
            
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        print("Please check if the file exists and the path is correct.")
    except Exception as e:
        print(f"❌ Error loading file: {e}")

if __name__ == "__main__":
    analyze_data_structure()