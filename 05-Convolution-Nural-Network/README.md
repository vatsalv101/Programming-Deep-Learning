# 5: Convolutional Neural Networks (CNNs)

## 🎯 Objective
Today's focus was on understanding the core building blocks of modern computer vision. Instead of just stacking `nn.Conv2d` layers, I implemented the forward and backward passes for Convolutions, Dropout, and Max-Pooling completely from scratch.

## 📝 Exercises Completed
1. **Conv2D Forward & Backward (`05a`, `05b`):** 
   - Implemented the cross-correlation operation for the forward pass of a 2D convolutional layer.
   - Calculated the complex gradient equations for the backward pass, routing gradients back to the inputs and the convolution kernels.
2. **Dropout & Max-Pooling (`05c-dropout-maxpool`):** 
   - Built an **Inverted Dropout** layer, handling the difference between `train` and `eval` modes.
   - Implemented a **Max-Pooling** layer from scratch, routing gradients backwards only to the pixels that triggered the maximum values.

## 🚀 Key Takeaways & Results
* **Performance Gain:** I trained two small CNN architectures side-by-side on CIFAR-10. Adding my custom Dropout and Max-Pooling layers increased the baseline test accuracy from **45.8% to 48.5%**.
* **Regularization:** Witnessed how Dropout prevents complex co-adaptations between features, acting as a powerful regularizer to prevent overfitting.
* **Downsampling:** Max-pooling provided translation invariance and drastically reduced the spatial dimensions, saving memory while keeping the most critical feature activations.
