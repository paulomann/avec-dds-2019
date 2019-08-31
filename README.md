# AVEC2019 Challenge

Some code I wrote for The Audio/Visual Emotion Challenge and Workshop (AVEC 2019) challenge to solve problems related to state-of-mind, depression, and cross-cultural affect.

This code is related to the Detecting Depression with AI Sub-Challenge (DDS), where we need to correlate audio-visual recordings of US Army veterans interacting with a virtual agent with the PHQ-8 questionnaire result (a questionnaire that can measure the severity of depression).

For the approach, in this code, I have tried a vanilla LSTM using OpenFace and ResNet50 features. However, due to the nature of the dataset, where videos can be huge, LSTM suffers from vanishing gradients. However, if we could first reduce the size of the sequence (using a Conv1D for example, similar to how we do for text classification [here](https://www.aclweb.org/anthology/D14-1181)), we can manage to alleviate the vanishing gradients problem. However, we can combine ths Conv1D approach with a Transformer or Transformer-XL architecture.

# TODO list

- [x] LSTM
- [ ] LSTM with self-attention
- [ ] Conv1D first, LSTM with attention after
- [ ] Conv1D first, Transformer
- [ ] Conv1D first, Transformer-XL
