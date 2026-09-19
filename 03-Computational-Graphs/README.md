# 3: Computational Graphs and Backpropagation

## 🎯 Objective
Today's focus was on understanding the exact mathematics powering PyTorch's `autograd` engine. I manually constructed computational graphs and calculated both the forward and backward passes (gradients) for scalar and matrix operations, proving an understanding of reverse-mode automatic differentiation.

## 📝 Exercises Completed
1. **Computational Graphs (`03a-computational-graphs`):** 
   - Constructed Directed Acyclic Graphs (DAGs) representing mathematical functions.
   - Manually calculated and hardcoded the **forward pass** activations for complex functions.
   - Manually calculated the **backward pass** (local gradients and chain rule) to compute derivatives with respect to inputs, handling both scalars and matrix/tensor operations (like `Wx + b`, `ReLU`, and `sigma`).

## 🚀 Key Takeaways
* **Reverse-Mode Autodiff:** By tracing the chain rule backwards from the loss to the inputs, I demonstrated exactly how deep learning frameworks compute gradients efficiently without redundant calculations.
* **Matrix Calculus:** Extending the chain rule from scalars to matrices/tensors is the core of training Neural Networks, requiring careful attention to dimensions and transpositions during the backward pass.
