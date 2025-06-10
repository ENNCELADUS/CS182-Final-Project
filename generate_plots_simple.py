#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Training data from 753381.out
train_loss = [0.7076, 0.6562, 0.6271, 0.6054, 0.5846, 0.5563, 0.5244, 0.4910, 0.4538, 0.4133, 0.3711, 0.3349, 0.2952, 0.2624, 0.2306, 0.2035, 0.1809, 0.1621, 0.1450]
val_loss = [0.6660, 0.6254, 0.6095, 0.6021, 0.5835, 0.5825, 0.5867, 0.5912, 0.6020, 0.6555, 0.6792, 0.6988, 0.7918, 0.8847, 0.9367, 1.0378, 1.0181, 1.1436, 1.1914]
train_auc = [0.5347, 0.6453, 0.6904, 0.7207, 0.7476, 0.7783, 0.8093, 0.8371, 0.8646, 0.8899, 0.9127, 0.9297, 0.9457, 0.9573, 0.9670, 0.9744, 0.9799, 0.9839, 0.9871]
val_auc = [0.6375, 0.6953, 0.7142, 0.7330, 0.7475, 0.7562, 0.7593, 0.7625, 0.7686, 0.7634, 0.7626, 0.7577, 0.7522, 0.7500, 0.7513, 0.7505, 0.7482, 0.7462, 0.7425]
train_acc = [0.5244, 0.5966, 0.6323, 0.6544, 0.6763, 0.7033, 0.7296, 0.7554, 0.7820, 0.8077, 0.8345, 0.8546, 0.8758, 0.8927, 0.9077, 0.9189, 0.9298, 0.9386, 0.9460]
val_acc = [0.5922, 0.6324, 0.6500, 0.6605, 0.6701, 0.6832, 0.6864, 0.6872, 0.6961, 0.6904, 0.6904, 0.6948, 0.6910, 0.6907, 0.6904, 0.6929, 0.6893, 0.6935, 0.6857]

epochs = range(1, len(train_loss) + 1)
best_epoch = 9

# Create plots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('V5.2 Training Progress (753381.out)', fontsize=16, fontweight='bold')

# Loss curves
axes[0, 0].plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
axes[0, 0].plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
axes[0, 0].axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label='Best Model')
axes[0, 0].set_title('Loss Curves')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# AUC curves
axes[0, 1].plot(epochs, train_auc, 'b-', label='Train AUC', linewidth=2)
axes[0, 1].plot(epochs, val_auc, 'r-', label='Val AUC', linewidth=2)
axes[0, 1].axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label='Best Model')
axes[0, 1].set_title('AUC Curves')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('AUC')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Accuracy curves
axes[1, 0].plot(epochs, train_acc, 'b-', label='Train Accuracy', linewidth=2)
axes[1, 0].plot(epochs, val_acc, 'r-', label='Val Accuracy', linewidth=2)
axes[1, 0].axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label='Best Model')
axes[1, 0].set_title('Accuracy Curves')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Accuracy')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Overfitting analysis
train_val_gap = [train - val for train, val in zip(train_auc, val_auc)]
axes[1, 1].plot(epochs, train_val_gap, 'purple', linewidth=2, label='Train-Val AUC Gap')
axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
axes[1, 1].axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label='Best Model')
axes[1, 1].set_title('Overfitting Analysis')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('AUC Gap')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()

# Save plots
plt.savefig('logs/v5_2_training_20250610_020129/training_curves.png', dpi=300, bbox_inches='tight')
plt.savefig('v5_2_training_curves_753381.png', dpi=300, bbox_inches='tight')

print("Training curves generated successfully!")
best_auc = max(val_auc)
best_epoch_idx = val_auc.index(best_auc) + 1
overfitting_gap = train_auc[-1] - val_auc[-1]
print("Best Validation AUC: %.4f at epoch %d" % (best_auc, best_epoch_idx))
print("Final overfitting gap: %.4f" % overfitting_gap)

plt.close() 