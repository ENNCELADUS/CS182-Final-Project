import pickle
import pandas as pd

# Check validation data
with open('data/full_dataset/pair_embeddings/val_data_with_mae_embeddings_20250530-151013.pkl', 'rb') as f:
    val_data = pickle.load(f)

print('=== VALIDATION DATA ===')
print(f'Type: {type(val_data)}')
print(f'Shape: {val_data.shape}')
print(f'Columns: {list(val_data.columns)}')
print(f'MAE embedding shape: {val_data["mae_embeddings"].iloc[0].shape}')
print(f'Memory usage: {val_data.memory_usage(deep=True).sum() / (1024**2):.2f} MB')
print(f'Sample interaction labels: {val_data["isInteraction"].value_counts()}') 