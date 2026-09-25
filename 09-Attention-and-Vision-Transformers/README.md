# 9: Attention and Vision Transformers

## 🎯 Objective
Today's focus was on the mechanism that revolutionized modern AI: Self-Attention. I learned how to process sequences without recurrence and applied this to computer vision by building a Vision Transformer (ViT) from scratch to classify images on CIFAR-10.

## 📝 Exercises Completed
1. **Data Preparation (`09a-data-preparation`):** 
   - Preprocessed and tokenized image data by splitting images into patches, preparing them for the Transformer architecture.
2. **Attention Mechanism (`09b-attention`):** 
   - Implemented the core Scaled Dot-Product Attention (Q, K, V) from scratch.
   - Built the Multi-Head Attention mechanism to allow the model to focus on different representation subspaces simultaneously.
3. **Vision Transformer (`09c-vit-cifar10`):** 
   - Assembled the Transformer Encoder blocks (LayerNorm, Multi-Head Attention, MLP).
   - Added positional embeddings to the image patches so the model retains spatial awareness.
   - Built the full ViT architecture and trained it on CIFAR-10, bridging the gap between NLP and Computer Vision architectures.

## 🚀 Key Takeaways
* **Self-Attention:** Discovered why attention mechanisms scale so well computationally compared to LSTMs—they process all tokens in parallel rather than sequentially.
* **Transformers for Vision:** Witnessed firsthand how an architecture initially designed for text translation (Transformers) can be elegantly adapted to process image patches, competing directly with CNNs.
