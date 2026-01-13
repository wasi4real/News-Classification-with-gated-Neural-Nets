# AG News Classification: Bidirectional RNN Analysis

## Overview

This project implements a classification system for the AG News dataset, categorizing articles into four domains: World, Sports, Business, and Sci/Tech. The primary focus is a comparative performance analysis between Bidirectional Long Short-Term Memory (BiLSTM) and Bidirectional Gated Recurrent Unit (BiGRU) architectures.

## Dataset Characteristics

The AG News dataset comprises news articles across four balanced categories:

1. World
2. Sports
3. Business
4. Sci/Tech

The training set contains 120,000 samples, while the test set contains 7,600 samples.

## Preprocessing Pipeline

The text processing sequence is standardized to ensure optimal neural network convergence:

* **Normalization**: Conversion of all text to lowercase and removal of non-alphanumeric characters.
* **Tokenization**: Segmenting text into individual word units using NLTK’s `RegexpTokenizer`.
* **Stopword Removal**: Eliminating high-frequency words with low semantic value.
* **Lemmatization**: Reducing words to their base linguistic forms.
* **Vectorization**: Integer encoding via the Keras `Tokenizer`.
* **Sequence Formatting**: Truncating and padding all inputs to a fixed length of 100 tokens (`maxlen=100`).

## Model Architectures

The system evaluates two specific bidirectional recurrent neural network (RNN) configurations:

### 1. Bidirectional LSTM (BiLSTM)

* **Mechanism**: Employs a cell state and three gates (input, forget, output) to regulate information flow.
* **Directionality**: Processes sequences in both forward and backward directions to capture global context.

### 2. Bidirectional GRU (BiGRU)

* **Mechanism**: Uses two gates (reset and update) and merges the cell state with the hidden state.
* **Directionality**: Simultaneous bidirectional processing similar to the BiLSTM.

## Experimental Results

Performance metrics were obtained through 10 training epochs with a 0.2 dropout rate to mitigate overfitting.

| Metric | BiLSTM | BiGRU |
| --- | --- | --- |
| **Test Accuracy** | 90.88% | **91.22%** |
| **Training Time** | 186.42s | **183.99s** |
| **Precision (Weighted)** | 0.91 | 0.91 |
| **Recall (Weighted)** | 0.91 | 0.91 |

## Performance Insights

The BiGRU model demonstrated a marginal but consistent superiority in both accuracy and computational efficiency.

1. **Complexity**: The GRU architecture has fewer parameters compared to the LSTM, reducing the computational overhead during training.
2. **Overfitting Mitigation**: For the specific sequence length of 100 tokens, the simpler GRU structure proved more robust against overfitting than the more complex LSTM.
3. **Efficiency**: The BiGRU achieved a faster training cycle (183.99 seconds) while maintaining higher precision in the "Sci/Tech" and "Business" categories compared to the BiLSTM.

