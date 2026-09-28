# 11: Generative Models (VAEs & GANs)

## 🎯 Objective
Today's focus shifted from discriminative tasks (predicting labels) to generative tasks (creating new data). I explored how to model complex data distributions and implemented three foundational generative architectures from scratch to generate images.

## 📝 Exercises Completed
1. **Variational Autoencoder (VAE) (`11a-vae`):** 
   - Implemented the reparameterization trick to allow backpropagation through a stochastic sampling node.
   - Built the Encoder and Decoder networks, optimizing both Reconstruction Loss (MSE/BCE) and KL-Divergence to enforce a continuous, structured latent space.
2. **Generative Adversarial Network (GAN) (`11b-gan`):** 
   - Built two competing neural networks: a Generator and a Discriminator.
   - Implemented the adversarial training loop, where the Generator learns to map random noise into realistic images to fool the Discriminator.

## 🚀 Key Takeaways
* **Latent Spaces:** Learned the mathematical difference between a standard autoencoder (which memorizes data) and a VAE (which structures the latent space into a Gaussian distribution for smooth sampling).
* **Adversarial Training:** Discovered the power and instability of GANs—how two networks playing a zero-sum game can result in incredibly sharp, realistic generated images compared to the blurry outputs of standard VAEs.
