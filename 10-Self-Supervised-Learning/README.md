# 10: Self-Supervised Learning (SSL)

## 🎯 Objective
Today's focus was on breaking away from the reliance on massive labeled datasets. I explored Self-Supervised Learning (SSL) to train neural networks to learn rich, meaningful feature representations from completely unlabeled data by solving "pretext" tasks and using non-contrastive methods.

## 📝 Exercises Completed
1. **RotNet (`10a-rotnet`):** 
   - Implemented a classic SSL pretext task where the network learns features simply by trying to predict the random rotation applied to an image (0°, 90°, 180°, 270°).
2. **BYOL - Bootstrap Your Own Latent (`10b-byol`):** 
   - Built a state-of-the-art non-contrastive SSL architecture (`models.py`, `pretrain.py`).
   - Implemented the Online and Target networks, using an Exponential Moving Average (EMA) to update the Target network.
   - Designed the data augmentation pipeline (`data.py`) required to generate two different views of the same image.
   - Evaluated the learned representations by freezing the network and training a simple linear classifier on top for a downstream task (`downstream.py`).

## 🚀 Key Takeaways
* **Pretext Tasks:** Learned that networks forced to understand the structure of an image (e.g., its orientation) naturally learn representations that transfer incredibly well to actual classification tasks.
* **Non-Contrastive Learning:** Discovered how BYOL prevents representation collapse (where the network just outputs a constant vector) without needing negative samples, relying strictly on the EMA target network and an asymmetric predictor.
