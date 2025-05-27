"""
Protein TCN Encoder - A dilated 1D Temporal Convolutional Network for protein sequence encoding.

This package implements a TCN-based encoder that converts ESM-C residue embeddings 
into fixed-length protein vectors using dilated convolutions and attention pooling.
"""

from .encoder import ProteinTCNEncoder, AttnPool, TCNBlock
from .model import SingleProteinModel, ProteinPairModel, MultiTaskModel
from .dataset import (
    ProteinDataset, 
    ProteinPairDataset, 
    collate_single_protein, 
    collate_protein_pair,
    create_dataloader,
    ProteinTransform
)

__version__ = "1.0.0"
__author__ = "CS182 Final Project"

__all__ = [
    # Core encoder components
    "ProteinTCNEncoder",
    "AttnPool", 
    "TCNBlock",
    
    # Model classes
    "SingleProteinModel",
    "ProteinPairModel", 
    "MultiTaskModel",
    
    # Dataset utilities
    "ProteinDataset",
    "ProteinPairDataset",
    "collate_single_protein",
    "collate_protein_pair", 
    "create_dataloader",
    "ProteinTransform",
] 