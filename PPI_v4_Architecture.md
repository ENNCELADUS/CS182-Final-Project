## 🧬 PPI Architecture Evolution (v2 → v4)

## 🔁 Key Architectural Updates

### 📋 v2 → v4 Architecture Comparison

| Aspect                         | v2 (Masked Autoencoder)                | v4 (Enhanced PPI Classifier)                      |
| ------------------------------ | -------------------------------------- | ------------------------------------------------- |
| **Purpose**              | Self-supervised protein representation | Supervised protein-protein interaction prediction |
| **Input**                | Single protein sequences               | Protein pairs (A + B)                             |
| **Task**                 | Reconstruction (autoencoder)           | Binary classification                             |
| **Positional Encoding**  | Learnable parameters                   | RoPE (Rotary Position Embedding)                  |
| **Transformer Layers**   | Standard (4 layers)                    | Enhanced transformer with RoPE (8–16 layers)     |
| **Interaction Modeling** | None                                   | Bidirectional cross-attention                     |
| **Pooling Strategy**     | Simple mean pooling                    | Hierarchical: Global + Local pooling              |
| **Decoder**              | MLP (embed → ff → input)             | Residual MLP                                      |
| **Variable Length**      | Fixed padding                          | Dynamic length processing                         |
| **Architecture Depth**   | Shallow (4 layers)                     | Deep (8–16 layers)                               |
| **Multi-GPU Support**    | None                                   | `torch.nn.DataParallel`                         |

---

## 🗺️ Architecture Roadmap

### Phase 1: Foundation (v2) — *Masked Autoencoder*

Input: Single Protein (B, L, 960)
    ↓
Learnable Positional Encoding
    ↓
Standard Transformer Encoder (4 layers)
    ↓
Mean Pooling
    ↓
Simple MLP Decoder
    ↓
Output: Reconstructed Protein (B, L, 960)

### Phase 2: Evolution (v4) - Enhanced PPI Classifier


Input: Protein Pair (emb_a, emb_b) each of shape (B, L, 960)

↓

┌─────────────────────────────────────────────────┐

│           ENHANCED PROTEIN ENCODER              │

│  ┌─────────────────────────────────────────────┐│

│  │   RoPE Positional Encoding                  ││

│  │   Enhanced Transformer Layers (8–16):       ││

│  │   - RoPE, GELU, Norm                        ││

│  └─────────────────────────────────────────────┘│

│  Hierarchical Attention Pooling:               │

│  ┌───────────────┬────────────────────────────┐│

│  │ Global Attn   │ Local Attn Pooling         ││

│  └───────────────┴────────────────────────────┘│

│  Compression Head (→ 960)                      │

└─────────────────────────────────────────────────┘

↓                            ↓

Refined emb_a (B, 960)     Refined emb_b (B, 960)

↓                            ↓

┌─────────────────────────────────────────────────┐

│       CROSS-ATTENTION INTERACTION               │

│  ┌─────────────────────────────────────────────┐│

│  │ A attends to B, B attends to A              ││

│  │ Interaction FFN + Residual                  ││

└─────────────────────────────────────────────────┘

↓

interaction_emb (B, 960)

↓

┌─────────────────────────────────────────────────┐

│           ENHANCED MLP DECODER                  │

│  [960 → 512 → 256 → 128 → 1]                    │

│  Each Layer: Linear → Norm → GELU → Dropout → Residual │

└─────────────────────────────────────────────────┘

↓

Output: Interaction Logits (B, 1)


### Phase 3: Advanced Features (v4)

#### 🔧 Technical Enhancements

- **Variable-Length Support**: Dynamic batching
- **Multi-GPU**: Parallel training with detection
- **Advanced Training**: AUC-based model selection, cosine annealing
- **Checkpointing**: Best model + resume capability
- **Memory Optimization**: Efficient padding, accumulation, caching

#### 🧠 Architectural Innovations

- **RoPE**: Rotary positional embeddings
- **Cross-Attention**: Bidirectional interaction
- **Hierarchical Pooling**: Global + local context
- **Residuals**: Across encoder and decoder
- **Enhanced Norming**: LayerNorm + dropout
