# 🧠 Programming Deep Learning

This repository documents my comprehensive, hands-on journey through deep learning concepts, algorithms, and advanced architectures. The focus of this project is to implement deep learning mechanisms efficiently from scratch using **PyTorch**, moving from foundational matrix mathematics all the way up to Large Language Models (LLMs).

## 🛠️ Tech Stack
* **Framework:** PyTorch (Tensors, Autograd, Neural Network modules)
* **Environment:** Python 3.12+, Jupyter Notebooks
* **Package Management:** `uv` (for fast, reproducible virtual environments)

## 🚀 Learning Journey & Progress Tracker
This project is broken down into 13 core topics. Each folder contains my vectorized PyTorch implementations, Jupyter notebooks detailing the concepts, and summaries of related research papers.

- [x] **01. Vectorization**: Mastering PyTorch tensor manipulations, broadcasting, and eliminating slow Python loops.
- [x] **02. Multilayer Perceptrons**: Implementing MLPs from scratch using low-level tensor mathematics and PyTorch's `nn` module.
- [x] **03. Computational Graphs**: Manually calculating forward and backward passes to understand reverse-mode automatic differentiation.
- [x] **04. Optimizers**: Implementing SGD, Momentum, RMSProp, Adam, and AdamW from scratch using PyTorch math.
- [x] **05. Convolutional Neural Networks**: Building Conv2D, Max-Pooling, and Dropout layers from scratch (forward and backward passes).
- [x] **06. Normalization and CNN Design**: Implementing LayerNorm and BatchNorm from scratch, and architecting a deep CNN on CIFAR-10.
- [x] **07. Weight Initialization and ResNets**: Kaiming initialization and implementing Residual Networks with skip connections from scratch.
- [x] **08. Recurrent Neural Networks (LSTMs)**: Building LSTM gates from scratch and training a character-level Shakespeare language model.


## 💻 Local Setup & Reproduction
If you want to run these notebooks locally, this repository uses `uv` for lightning-fast dependency management.

1. Clone the repository:
   ```bash
   git clone https://github.com/vatsalv101/Programming-Deep-Learning.git
   cd Programming-Deep-Learning