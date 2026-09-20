# 4: Optimizers from Scratch

## 🎯 Objective
Today's focus was on understanding exactly how deep learning models update their weights. Instead of relying on PyTorch's built-in `torch.optim` modules, I implemented the mathematics behind the industry's most popular gradient descent optimization algorithms completely from scratch.

## 📝 Optimizers Implemented
I successfully wrote the initialization and step functions for the following optimizers:
1. **SGD (Stochastic Gradient Descent):** The foundational algorithm.
2. **SGD with Momentum:** Added velocity tracking to accelerate gradients in the right directions.
3. **AdaGrad:** Implemented adaptive learning rates based on accumulated squared gradients.
4. **RMSProp:** Solved AdaGrad's vanishing learning rate problem using a moving average.
5. **Adam:** Combined the benefits of Momentum (first moments) and RMSProp (second moments) with bias correction.
6. **AdamW:** Implemented decoupled weight decay for better generalization, matching the current state-of-the-art approach for training Transformers.

## 🚀 Key Takeaways
* **Optimization States:** Learned how advanced optimizers like Adam carry internal states (`m`, `v`, `t`) across training steps, unlike basic SGD.
* **Bias Correction:** Understood exactly why Adam requires bias correction (`1 - beta^t`) in the early stages of training to prevent moments from heavily biasing towards zero.
* **Weight Decay vs. L2 Regularization:** Saw first-hand the mathematical difference between standard L2 regularization and decoupled weight decay (AdamW).
