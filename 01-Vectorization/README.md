# 1: Vectorization and Linear Regression

## 🎯 Objective
Today's focus was on computational efficiency and foundational modeling. I learned how to eliminate slow Python `for` loops leveraging PyTorch's vectorized operations, and applied these skills to build a Linear Regression model from scratch.

## 📝 Exercises Completed
1. **Vectorization Basics:** Transitioned basic iterative logic into optimized PyTorch tensor operations.
2. **Vectorization Advanced:** Implemented complex operations like matrix multiplication, dot products, and broadcasting.
3. **Extra Practice:** Completed additional vectorization challenges.
4. **Linear Regression:** Built a custom linear regression model optimized with PyTorch vectorization.

## 🛠️ PyTorch Functions Mastered
During these exercises (including the comprehensive `Practice.ipynb`), I utilized a wide array of PyTorch functions to eliminate loops and optimize complex operations:
* **Tensor Manipulation & Reshaping:** `torch.cat()`, `tensor.unsqueeze()`, `tensor.expand()`, `tensor.view()`, `tensor.permute()`, `tensor.transpose()`
* **Mathematical & Logical Operations:** `torch.sqrt()`, `torch.relu()`, `torch.clamp()`, `tensor.any()`
* **Linear Algebra:** Matrix multiplication (`@`, `torch.matmul()`), `torch.einsum()` (for advanced tensor contractions like bilinear products), `tensor.norm()`
* **Reductions & Statistics:** `torch.sum()`, `torch.mean()`, `torch.max()`, `torch.min()`, `torch.argmin()`, `torch.topk()`
* **Advanced Scatter & Indexing:** `tensor.scatter_()`, `tensor.scatter_add_()`, `tensor.scatter_reduce_()`, `tensor.masked_fill()`, Advanced Indexing (`tensor[batch_idx, seq_idx]`)
* **Computer Vision / Sequence Ops:** `F.unfold()` / `tensor.unfold()` (for rolling windows and ViT patching), `torch.triu()` (causal masking)

## 🚀 Key Takeaways
* **Speed:** Vectorized PyTorch operations are highly optimized (and can run on GPUs), making them magnitudes faster than standard Python loops.
* **Broadcasting:** Learned how PyTorch handles operations on tensors of different shapes without needing to duplicate data in memory, specifically using `.unsqueeze()`.
