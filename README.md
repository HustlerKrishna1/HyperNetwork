# 🧠 Hypernetwork (Tiny Model → Full Model Weights)

This project builds a **small model (hypernetwork)** that generates the weights of a larger model.

Instead of storing full weight matrices, we store **shared bases** and only generate a few values per layer → huge memory savings.

---

## 🚀 What This Does

- Replaces large model weights with a tiny generator
- Uses **SVD-style factorization**:
  W = U @ diag(s) @ V^T
  
- Only generates:
- `s` (small vector, e.g. 32 values)
- Stores:
- `U` and `V` (shared, learned once)

---

## 📦 Result

| Component        | Size     |
|-----------------|----------|
| Hypernetwork     | ~1.3 MB  |
| Original model   | ~28 MB   |
| Compression      | **~21× smaller** |

---

## 🏗️ Structure
hypernetwork/
├── main.py
├── models/
├── heads/
├── training/
├── utils/
└── experiments/

---

## ⚡ How to Run

### 1. Quick check (no GPU)
### 2. Train the system

python main.py --mode train


### 3. Run experiments

python main.py --mode experiments


### 4. See size breakdown

python main.py --mode budget


### 5. Compare methods

python main.py --mode strategies

🧩 Key Idea

Instead of generating full weight matrices:

❌ Old way:

backbone → full weights (huge)


✅ New way:

backbone → small vector (s)

↓


W = U @ diag(s) @ V^T


This keeps the model:
- small
- fast
- efficient

---

## 🏋️ Training Process

1. Train a normal model (teacher)
2. Train hypernetwork to copy its weights
3. Fine-tune using:
   - task loss
   - distillation

---

## 🔧 Variants Included

- **SVD Hypernetwork** (main)
