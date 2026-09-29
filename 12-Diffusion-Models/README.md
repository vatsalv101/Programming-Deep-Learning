# 12: Diffusion Models

## 🎯 Objective
Today's focus was on the most powerful generative architecture of the current era: Diffusion Models. I learned how to slowly corrupt data with noise and train a U-Net to reverse this process, allowing for the generation of incredibly high-fidelity images.

## 📝 Exercises Completed
1. **U-Net Architecture (`12a-unet`):** 
   - Built the classic U-Net architecture from scratch, utilizing skip connections between the encoder (downsampling) and decoder (upsampling) paths to preserve high-frequency spatial details.
2. **Denoising Diffusion Probabilistic Models (`12b-ddpm`):** 
   - Implemented the forward diffusion process (adding Gaussian noise over $T$ timesteps).
   - Trained the U-Net to predict the added noise at any given timestep (the backward denoising process).
3. **Stochastic Differential Equations (`12c-sde`):** 
   - Explored the continuous-time mathematical generalization of diffusion models using SDEs (Score-Based Generative Modeling).
4. **Denoising Diffusion Implicit Models (`12d-ddim`):** 
   - Implemented DDIM sampling to massively accelerate the generation process. By changing the backward generation process to be non-Markovian, I was able to skip timesteps and generate images exponentially faster than standard DDPMs.

## 🚀 Key Takeaways
* **Generative Superiority:** Discovered why Diffusion models have largely replaced GANs for image generation—they offer much more stable training (no adversarial min-max game) and better mode coverage.
* **Sampling Speed:** Understood the biggest drawback of standard DDPMs (slow generation requiring 1000+ neural network passes) and practically solved it by implementing DDIM sampling.
