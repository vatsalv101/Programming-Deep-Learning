# 7: Weight Initialization and ResNets

## 🎯 Objective
Today's focus was on solving the degradation problem in extremely deep networks. I explored how proper weight initialization mathematically prevents exploding/vanishing gradients, and then I implemented one of the most influential architectures in deep learning history: Residual Networks (ResNet).

## 📝 Exercises Completed
1. **Weight Initialization (`07a-weight-init`):** 
   - Studied the mathematical variance of activations during forward and backward passes.
   - Implemented Kaiming (He) initialization to keep variance stable across deep layers, preventing gradients from vanishing.
2. **ResNet Paper Reading (`07b-resnet-paper-reading`):** 
   - Analyzed the original 2015 ResNet paper by Kaiming He et al. to deeply understand why adding layers to plain networks hurts training, and how skip connections solve this.
3. **ResNet Implementation (`07c-resnet-cifar10`):** 
   - Built the fundamental building blocks of a ResNet (`blocks.py` and `shortcuts.py`), including the standard Residual Block with identity skip connections and projection shortcuts (1x1 convolutions).
   - Assembled the blocks into a complete, scalable ResNet architecture tailored for the CIFAR-10 dataset (`resnet.py`).
   - Trained the model from scratch to observe the performance benefits of deep residual learning.

## 🚀 Key Takeaways
* **The Degradation Problem:** Learned that simply stacking more layers in standard CNNs causes training errors to increase (not just test errors due to overfitting), and skip connections perfectly mitigate this by allowing gradients to flow unimpeded.
* **Initialization Matters:** Even with skip connections or batch norm, initializing weights correctly (like Kaiming Normal) is crucial for ensuring the network starts training effectively from epoch 1.
