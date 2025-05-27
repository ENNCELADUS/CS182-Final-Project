import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import numpy as np


def collate_single_protein(batch):
    """
    Collate function for single protein tasks.
    
    Args:
        batch: List of tuples (embeddings, label)
            embeddings: (L_i, 960) tensor
            label: scalar or tensor
    
    Returns:
        embeddings: (B, L_max, 960) padded tensor
        mask: (B, L_max) boolean mask
        labels: (B,) or (B, n_classes) tensor
    """
    embeddings, labels = zip(*batch)
    
    # Convert to tensors if needed
    embeddings = [torch.tensor(emb, dtype=torch.float32) if not isinstance(emb, torch.Tensor) else emb 
                  for emb in embeddings]
    
    # Pad sequences
    padded_emb = pad_sequence(embeddings, batch_first=True)  # (B, L_max, 960)
    
    # Create mask
    lengths = torch.tensor([len(emb) for emb in embeddings])
    mask = torch.arange(padded_emb.size(1))[None, :] < lengths[:, None]  # (B, L_max)
    
    # Stack labels
    if isinstance(labels[0], (int, float)):
        labels = torch.tensor(labels, dtype=torch.float32)
    else:
        labels = torch.stack([torch.tensor(label, dtype=torch.float32) for label in labels])
    
    return padded_emb, mask, labels


def collate_protein_pair(batch):
    """
    Collate function for protein pair tasks.
    
    Args:
        batch: List of tuples (embA, embB, label)
            embA: (L_A, 960) tensor for protein A
            embB: (L_B, 960) tensor for protein B
            label: scalar or tensor
    
    Returns:
        embA: (B, L_A_max, 960) padded tensor
        maskA: (B, L_A_max) boolean mask
        embB: (B, L_B_max, 960) padded tensor
        maskB: (B, L_B_max) boolean mask
        labels: (B,) or (B, n_classes) tensor
    """
    embA_list, embB_list, labels = zip(*batch)
    
    # Convert to tensors if needed
    embA_list = [torch.tensor(emb, dtype=torch.float32) if not isinstance(emb, torch.Tensor) else emb 
                 for emb in embA_list]
    embB_list = [torch.tensor(emb, dtype=torch.float32) if not isinstance(emb, torch.Tensor) else emb 
                 for emb in embB_list]
    
    # Pad sequences separately
    padded_embA = pad_sequence(embA_list, batch_first=True)  # (B, L_A_max, 960)
    padded_embB = pad_sequence(embB_list, batch_first=True)  # (B, L_B_max, 960)
    
    # Create masks
    lengthsA = torch.tensor([len(emb) for emb in embA_list])
    lengthsB = torch.tensor([len(emb) for emb in embB_list])
    maskA = torch.arange(padded_embA.size(1))[None, :] < lengthsA[:, None]  # (B, L_A_max)
    maskB = torch.arange(padded_embB.size(1))[None, :] < lengthsB[:, None]  # (B, L_B_max)
    
    # Stack labels
    if isinstance(labels[0], (int, float)):
        labels = torch.tensor(labels, dtype=torch.float32)
    else:
        labels = torch.stack([torch.tensor(label, dtype=torch.float32) for label in labels])
    
    return padded_embA, maskA, padded_embB, maskB, labels


class ProteinDataset(Dataset):
    """Generic protein dataset for single protein tasks."""
    
    def __init__(self, embeddings, labels, transform=None):
        """
        Args:
            embeddings: List of (L_i, 960) tensors or arrays
            labels: List of labels (scalars or arrays)
            transform: Optional transform to apply to embeddings
        """
        assert len(embeddings) == len(labels), "Embeddings and labels must have same length"
        self.embeddings = embeddings
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        emb = self.embeddings[idx]
        label = self.labels[idx]
        
        if self.transform:
            emb = self.transform(emb)
        
        return emb, label


class ProteinPairDataset(Dataset):
    """Generic protein pair dataset for PPI tasks."""
    
    def __init__(self, embeddingsA, embeddingsB, labels, transform=None):
        """
        Args:
            embeddingsA: List of (L_A_i, 960) tensors for protein A
            embeddingsB: List of (L_B_i, 960) tensors for protein B
            labels: List of labels (scalars or arrays)
            transform: Optional transform to apply to embeddings
        """
        assert len(embeddingsA) == len(embeddingsB) == len(labels), \
            "All inputs must have same length"
        self.embeddingsA = embeddingsA
        self.embeddingsB = embeddingsB
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.embeddingsA)
    
    def __getitem__(self, idx):
        embA = self.embeddingsA[idx]
        embB = self.embeddingsB[idx]
        label = self.labels[idx]
        
        if self.transform:
            embA = self.transform(embA)
            embB = self.transform(embB)
        
        return embA, embB, label


def create_dataloader(dataset, batch_size=32, shuffle=True, num_workers=0, task_type='single'):
    """
    Create a DataLoader with appropriate collate function.
    
    Args:
        dataset: Dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        task_type: 'single' or 'pair'
    
    Returns:
        DataLoader instance
    """
    if task_type == 'single':
        collate_fn = collate_single_protein
    elif task_type == 'pair':
        collate_fn = collate_protein_pair
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )


# Utility functions for data preprocessing
def normalize_embeddings(embeddings, method='layer_norm'):
    """
    Normalize embeddings.
    
    Args:
        embeddings: (L, 960) tensor
        method: 'layer_norm', 'batch_norm', or 'none'
    
    Returns:
        Normalized embeddings
    """
    if method == 'layer_norm':
        return F.layer_norm(embeddings, embeddings.shape[-1:])
    elif method == 'batch_norm':
        # Normalize across the feature dimension
        mean = embeddings.mean(dim=-1, keepdim=True)
        std = embeddings.std(dim=-1, keepdim=True)
        return (embeddings - mean) / (std + 1e-8)
    elif method == 'none':
        return embeddings
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def truncate_or_pad_sequence(embedding, max_length=1024, pad_value=0.0):
    """
    Truncate or pad a single sequence to fixed length.
    
    Args:
        embedding: (L, 960) tensor
        max_length: Maximum sequence length
        pad_value: Value to use for padding
    
    Returns:
        (max_length, 960) tensor
    """
    L, D = embedding.shape
    
    if L >= max_length:
        # Truncate
        return embedding[:max_length]
    else:
        # Pad
        padding = torch.full((max_length - L, D), pad_value, dtype=embedding.dtype)
        return torch.cat([embedding, padding], dim=0)


class ProteinTransform:
    """Composable transforms for protein embeddings."""
    
    def __init__(self, normalize=True, max_length=None, add_noise=False, noise_std=0.01):
        """
        Args:
            normalize: Whether to apply layer normalization
            max_length: If specified, truncate/pad to this length
            add_noise: Whether to add Gaussian noise (for data augmentation)
            noise_std: Standard deviation of noise
        """
        self.normalize = normalize
        self.max_length = max_length
        self.add_noise = add_noise
        self.noise_std = noise_std
    
    def __call__(self, embedding):
        """Apply transforms to embedding."""
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding, dtype=torch.float32)
        
        if self.normalize:
            embedding = normalize_embeddings(embedding, method='layer_norm')
        
        if self.max_length is not None:
            embedding = truncate_or_pad_sequence(embedding, self.max_length)
        
        if self.add_noise:
            noise = torch.randn_like(embedding) * self.noise_std
            embedding = embedding + noise
        
        return embedding 