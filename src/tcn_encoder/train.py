import argparse
import os
import json
import math
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm

# Try to import PyTorch Lightning (optional)
try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    from pytorch_lightning.loggers import TensorBoardLogger
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False
    print("PyTorch Lightning not available. Using vanilla PyTorch training.")

from .model import SingleProteinModel, ProteinPairModel
from .dataset import ProteinDataset, ProteinPairDataset, create_dataloader


class ProteinLightningModule(pl.LightningModule):
    """PyTorch Lightning module for protein tasks."""
    
    def __init__(self, model, learning_rate=3e-4, weight_decay=1e-4, warmup_steps=1000, 
                 max_steps=10000, task_type='classification'):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.task_type = task_type
        
        # Loss function
        if task_type == 'binary_classification':
            self.criterion = nn.BCEWithLogitsLoss()
        elif task_type == 'multi_classification':
            self.criterion = nn.CrossEntropyLoss()
        elif task_type == 'regression':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unknown task_type: {task_type}")
        
        self.save_hyperparameters(ignore=['model'])
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def training_step(self, batch, batch_idx):
        if isinstance(self.model, ProteinPairModel):
            embA, maskA, embB, maskB, labels = batch
            logits = self.model(embA, maskA, embB, maskB)
        else:
            emb, mask, labels = batch
            logits = self.model(emb, mask)
        
        if self.task_type == 'binary_classification':
            labels = labels.float()
            loss = self.criterion(logits.squeeze(), labels)
        elif self.task_type == 'multi_classification':
            labels = labels.long()
            loss = self.criterion(logits, labels)
        elif self.task_type == 'regression':
            labels = labels.float()
            loss = self.criterion(logits.squeeze(), labels)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        if isinstance(self.model, ProteinPairModel):
            embA, maskA, embB, maskB, labels = batch
            logits = self.model(embA, maskA, embB, maskB)
        else:
            emb, mask, labels = batch
            logits = self.model(emb, mask)
        
        if self.task_type == 'binary_classification':
            labels = labels.float()
            loss = self.criterion(logits.squeeze(), labels)
            preds = torch.sigmoid(logits.squeeze()) > 0.5
            acc = (preds == labels).float().mean()
            self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        elif self.task_type == 'multi_classification':
            labels = labels.long()
            loss = self.criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)
            acc = (preds == labels).float().mean()
            self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        elif self.task_type == 'regression':
            labels = labels.float()
            loss = self.criterion(logits.squeeze(), labels)
            # For regression, log MAE as well
            mae = torch.abs(logits.squeeze() - labels).mean()
            self.log('val_mae', mae, on_epoch=True, prog_bar=True)
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Cosine annealing with warmup
        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / self.warmup_steps
            else:
                progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * progress))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }


class VanillaTrainer:
    """Vanilla PyTorch trainer."""
    
    def __init__(self, model, train_loader, val_loader, device='cuda', 
                 learning_rate=3e-4, weight_decay=1e-4, warmup_steps=1000,
                 max_epochs=50, task_type='classification', save_dir='./checkpoints'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_epochs = max_epochs
        self.task_type = task_type
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Loss function
        if task_type == 'binary_classification':
            self.criterion = nn.BCEWithLogitsLoss()
        elif task_type == 'multi_classification':
            self.criterion = nn.CrossEntropyLoss()
        elif task_type == 'regression':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unknown task_type: {task_type}")
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Scheduler
        total_steps = len(train_loader) * max_epochs
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * progress))
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
        
        # Tracking
        self.best_val_loss = float('inf')
        self.step = 0
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            # Move to device
            if isinstance(self.model, ProteinPairModel):
                embA, maskA, embB, maskB, labels = [x.to(self.device) for x in batch]
                batch_device = (embA, maskA, embB, maskB, labels)
            else:
                emb, mask, labels = [x.to(self.device) for x in batch]
                batch_device = (emb, mask, labels)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    loss = self._compute_loss(batch_device)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss = self._compute_loss(batch_device)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            self.step += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'})
        
        return total_loss / num_batches
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move to device
                if isinstance(self.model, ProteinPairModel):
                    embA, maskA, embB, maskB, labels = [x.to(self.device) for x in batch]
                    batch_device = (embA, maskA, embB, maskB, labels)
                    logits = self.model(embA, maskA, embB, maskB)
                else:
                    emb, mask, labels = [x.to(self.device) for x in batch]
                    batch_device = (emb, mask, labels)
                    logits = self.model(emb, mask)
                
                loss = self._compute_loss(batch_device)
                total_loss += loss.item()
                
                # Compute accuracy for classification tasks
                if self.task_type in ['binary_classification', 'multi_classification']:
                    if self.task_type == 'binary_classification':
                        preds = (torch.sigmoid(logits.squeeze()) > 0.5).long()
                        labels = labels.long()
                    else:
                        preds = torch.argmax(logits, dim=1)
                        labels = labels.long()
                    
                    total_correct += (preds == labels).sum().item()
                    total_samples += labels.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        
        return avg_loss, accuracy
    
    def _compute_loss(self, batch):
        if isinstance(self.model, ProteinPairModel):
            embA, maskA, embB, maskB, labels = batch
            logits = self.model(embA, maskA, embB, maskB)
        else:
            emb, mask, labels = batch
            logits = self.model(emb, mask)
        
        if self.task_type == 'binary_classification':
            labels = labels.float()
            return self.criterion(logits.squeeze(), labels)
        elif self.task_type == 'multi_classification':
            labels = labels.long()
            return self.criterion(logits, labels)
        elif self.task_type == 'regression':
            labels = labels.float()
            return self.criterion(logits.squeeze(), labels)
    
    def train(self):
        print(f"Training for {self.max_epochs} epochs...")
        
        for epoch in range(self.max_epochs):
            print(f"\nEpoch {epoch + 1}/{self.max_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                }, self.save_dir / 'best_model.pt')
                print(f"Saved best model with val_loss: {val_loss:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description='Train Protein TCN Encoder')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, choices=['single', 'pair'], default='single',
                        help='Type of model: single protein or protein pair')
    parser.add_argument('--task_type', type=str, 
                        choices=['binary_classification', 'multi_classification', 'regression'],
                        default='binary_classification', help='Type of task')
    parser.add_argument('--n_classes', type=int, default=2, help='Number of classes for classification')
    parser.add_argument('--d_encoder', type=int, default=512, help='Encoder output dimension')
    parser.add_argument('--d_hidden', type=int, default=256, help='Hidden dimension in task head')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--max_epochs', type=int, default=50, help='Maximum number of epochs')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Warmup steps')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True, help='Path to data')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    
    # Training setup
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--use_lightning', action='store_true', help='Use PyTorch Lightning')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile (PyTorch 2.0+)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data (this is a placeholder - you'll need to implement data loading)
    print("Loading data...")
    # TODO: Implement data loading based on your specific data format
    # For now, we'll create dummy data
    
    # Create dummy data for demonstration
    if args.model_type == 'single':
        # Dummy single protein data
        embeddings = [torch.randn(np.random.randint(50, 500), 960) for _ in range(1000)]
        labels = torch.randint(0, args.n_classes, (1000,))
        dataset = ProteinDataset(embeddings, labels)
        task_type = 'single'
    else:
        # Dummy protein pair data
        embeddingsA = [torch.randn(np.random.randint(50, 500), 960) for _ in range(1000)]
        embeddingsB = [torch.randn(np.random.randint(50, 500), 960) for _ in range(1000)]
        labels = torch.randint(0, args.n_classes, (1000,))
        dataset = ProteinPairDataset(embeddingsA, embeddingsB, labels)
        task_type = 'pair'
    
    # Split dataset
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = create_dataloader(
        train_dataset, args.batch_size, shuffle=True, 
        num_workers=args.num_workers, task_type=task_type
    )
    val_loader = create_dataloader(
        val_dataset, args.batch_size, shuffle=False,
        num_workers=args.num_workers, task_type=task_type
    )
    
    # Create model
    if args.model_type == 'single':
        model = SingleProteinModel(
            n_classes=args.n_classes if args.task_type != 'regression' else None,
            d_encoder=args.d_encoder,
            d_hidden=args.d_hidden,
            dropout=args.dropout,
            task_type=args.task_type
        )
    else:
        model = ProteinPairModel(
            n_classes=args.n_classes if args.task_type != 'regression' else 1,
            d_encoder=args.d_encoder,
            d_hidden=args.d_hidden,
            dropout=args.dropout,
            task_type=args.task_type
        )
    
    # Compile model if requested (PyTorch 2.0+)
    if args.compile:
        try:
            model = torch.compile(model)
            print("Model compiled successfully")
        except Exception as e:
            print(f"Failed to compile model: {e}")
    
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Train model
    if args.use_lightning and LIGHTNING_AVAILABLE:
        # PyTorch Lightning training
        lightning_model = ProteinLightningModule(
            model, args.learning_rate, args.weight_decay,
            args.warmup_steps, len(train_loader) * args.max_epochs,
            args.task_type
        )
        
        callbacks = [
            ModelCheckpoint(
                dirpath=args.save_dir,
                filename='best-{epoch:02d}-{val_loss:.2f}',
                monitor='val_loss',
                mode='min',
                save_top_k=1
            ),
            EarlyStopping(monitor='val_loss', patience=10, mode='min')
        ]
        
        logger = TensorBoardLogger(args.save_dir, name='protein_tcn')
        
        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            callbacks=callbacks,
            logger=logger,
            accelerator='gpu' if device.type == 'cuda' else 'cpu',
            devices=1,
            precision=16 if device.type == 'cuda' else 32,
            gradient_clip_val=1.0
        )
        
        trainer.fit(lightning_model, train_loader, val_loader)
    else:
        # Vanilla PyTorch training
        trainer = VanillaTrainer(
            model, train_loader, val_loader, device,
            args.learning_rate, args.weight_decay, args.warmup_steps,
            args.max_epochs, args.task_type, args.save_dir
        )
        trainer.train()
    
    print("Training completed!")


if __name__ == '__main__':
    main() 