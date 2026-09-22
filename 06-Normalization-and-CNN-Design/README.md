# 6: Normalization and CNN Design

## 🎯 Objective
Today's focus was on stabilizing deep neural networks and scaling up architectures. I learned how internal covariate shift slows down training and implemented normalization techniques from scratch to fix it. Finally, I designed and trained a deeper Convolutional Neural Network on CIFAR-10.

## 📝 Exercises Completed
1. **Layer Normalization (`06a-layernorm-forward`):** Implemented the forward pass of LayerNorm from scratch, primarily used in Transformers.
2. **Batch Normalization (`06b-batchnorm-forward`):** Implemented the forward pass of BatchNorm from scratch, the standard for stabilizing and accelerating CNNs, including the tracking of running means and variances for inference.
3. **CNN Architecture Design (`06c-cnn-design`):** 
   - Designed a deep CNN using `torch.nn.Sequential`.
   - Combined convolutional layers, `nn.BatchNorm2d`, ReLU activations, and Dropout into a robust, modern architecture.
   - Wrote a full PyTorch training loop integrating the AdamW optimizer and CrossEntropy loss.

## 🚀 Key Takeaways
* **Stabilization:** Normalization techniques (Batch and Layer Norm) are critical for training deep networks by keeping intermediate activations on a similar scale, allowing for higher learning rates.
* **Architecture Engineering:** Successfully combined the individual components I've been building (convolutions, pooling, dropout, optimization) into a cohesive, production-ready CNN pipeline.
