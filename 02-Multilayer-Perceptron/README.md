# 2: Multilayer Perceptrons (MLP)

## 🎯 Objective
Today's focus was on understanding the architecture of Neural Networks. I built Multi-Layer Perceptrons (MLPs) using two distinct approaches: a low-level approach from scratch using raw PyTorch tensors, and a high-level approach leveraging PyTorch's powerful `torch.nn` module.

## 📝 Exercises Completed
1. **MLP Low-Level (`02b-mlp-low-level`):** Implemented a neural network manually. This involved manually defining weight matrices and bias vectors, and writing the mathematical forward pass equations.
2. **MLP High-Level (`02c-mlp-high-level`):** Transitioned to PyTorch's production-ready tools. Built the same network architecture using `torch.nn.Sequential`, `nn.Linear`, and built-in activation functions.

## 🚀 Key Takeaways
* **Under the Hood:** Building an MLP from scratch solidified my understanding of how linear transformations ($XW + b$) and non-linear activations interact.
* **PyTorch `nn` Module:** Discovered how much boilerplate code PyTorch's `nn.Module` and `nn.Linear` save you when constructing complex deep learning architectures.
