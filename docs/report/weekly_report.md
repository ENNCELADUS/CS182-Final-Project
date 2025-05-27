## PPI Prediction Work Report

### 1. Summary of Work Done

In this period, we have finished the following work:

- Stage 1(Extract features): Successfully use ESM C as encoder to embed every protein in the dataset and get embeddings for each protein with shape of [L, 960], where L is the sequence length and 960 is the embedding dimension.

- Stage 2(Pooling): Transform the embeddings from [L, 960] to [960]
  - First we try mean-pooling, which performs not so well
<img src="../../results/img/xgb_mean.png" alt="XGBoost Mean Pooling Results" width="600">
  - Then we implement a masked autoencoder, which performs better(and we are still finetuning it)
<img src="../../results/img/xgb_mae_v1.png" alt="XGBoost MAE Results" width="600">

- The following plot is  
  
