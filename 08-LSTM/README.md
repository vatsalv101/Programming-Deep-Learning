# 8: Recurrent Neural Networks (LSTMs)

## 🎯 Objective
Today's focus shifted from computer vision to sequence modeling. Standard RNNs suffer heavily from the vanishing gradient problem when processing long sequences. To solve this, I implemented a Long Short-Term Memory (LSTM) network from scratch and trained it to generate Shakespearean text.

## 📝 Exercises Completed
1. **LSTM Implementation (`08a-lstm`):**
   - Implemented the core math of an LSTM Cell (`lstm_cells.py`), including the Forget Gate, Input Gate, Output Gate, and Cell State updates.
   - Built an autoregressive character-level Language Model (`model.py`) that uses the LSTM cells to predict the next character in a sequence.
   - Trained the model on a dataset of Shakespeare's works (`shakespeare.txt`), allowing it to learn the statistical distribution of the text and generate novel, Shakespeare-like text character by character.

## 🚀 Key Takeaways
* **Gating Mechanisms:** Learned how the internal gates (sigmoid and tanh) of an LSTM allow it to selectively remember and forget information over long time steps, effectively bypassing the vanishing gradient problem that plagues vanilla RNNs.
* **Autoregressive Generation:** Gained hands-on experience with how language models generate text: predicting a probability distribution for the next token, sampling it, and feeding it back as input for the next step.
