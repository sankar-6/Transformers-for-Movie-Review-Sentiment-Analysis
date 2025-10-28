# SentimentScope: Transformer-Based Sentiment Analysis

A comprehensive implementation of sentiment analysis using transformer models trained from scratch on the IMDB movie review dataset. This project demonstrates how to build, train, and evaluate a transformer-based classifier for binary sentiment classification.

## 🎯 Project Overview

SentimentScope is a sentiment analysis project developed for Cinescope, an entertainment company looking to enhance their recommendation system. The project implements a custom transformer architecture to classify movie reviews as positive or negative, achieving **77.36% accuracy** on the test dataset.

### Key Features

- 🏗️ **Custom Transformer Architecture**: Built from scratch using PyTorch
- 📊 **Comprehensive Data Analysis**: Detailed exploration of the IMDB dataset
- 🔧 **Modular Design**: Well-structured, reusable components
- 🚀 **GPU Acceleration**: CUDA-enabled training for faster processing
- 📈 **Performance Monitoring**: Real-time training progress and validation metrics

## 📋 Learning Objectives

By completing this project, you will demonstrate competency in:

- Loading, exploring, and preparing text datasets for transformer model training
- Customizing transformer architecture for classification tasks
- Implementing efficient data loading with PyTorch DataLoader
- Training and evaluating transformer models on real-world datasets
- Achieving target performance metrics (>75% accuracy)

## 🗂️ Dataset

### IMDB Movie Review Dataset

This project uses the [IMDB dataset](http://ai.stanford.edu/~amaas/data/sentiment/) by Maas et al., which contains:

- **50,000 movie reviews** (25,000 training, 25,000 testing)
- **Binary labels**: Positive (1) and Negative (0) sentiment
- **Balanced distribution**: Equal number of positive and negative reviews
- **Average review length**: ~234 words per review

### Download Instructions

1. **Download the dataset**:
   ```bash
   wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
   ```

2. **Extract the dataset**:
   ```bash
   tar -xzf aclImdb_v1.tar.gz
   ```

3. **Dataset structure**:
   ```
   aclImdb/
   ├── train/
   │   ├── pos/    # Positive reviews for training
   │   ├── neg/    # Negative reviews for training
   │   └── unsup/  # Unsupervised data (not used)
   └── test/
       ├── pos/    # Positive reviews for testing
       └── neg/    # Negative reviews for testing
   ```

## 🏗️ Architecture

### Model Configuration

```python
config = {
    "vocabulary_size": 30522,    # BERT-base-uncased vocabulary
    "num_classes": 2,            # Binary classification
    "d_embed": 128,              # Embedding dimension
    "context_size": 128,         # Maximum sequence length
    "layers_num": 4,             # Number of transformer layers
    "heads_num": 4,              # Number of attention heads
    "head_size": 32,             # Dimension per attention head
    "dropout_rate": 0.1,         # Dropout probability
    "use_bias": True             # Linear layer bias
}
```

### Key Components

1. **Token Embedding Layer**: Maps token indices to dense vectors
2. **Positional Embedding**: Adds positional information to tokens
3. **Multi-Head Attention**: 4 attention heads with 32 dimensions each
4. **Feed-Forward Networks**: GELU-activated linear transformations
5. **Layer Normalization**: Stabilizes training across layers
6. **Classification Head**: Mean pooling + linear layer for binary classification

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch torchvision torchaudio
pip install transformers
pip install pandas matplotlib numpy
pip install tqdm
```

### Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/sentiment-scope.git
   cd sentiment-scope
   ```

2. **Download and prepare the dataset**:
   ```bash
   wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
   tar -xzf aclImdb_v1.tar.gz
   ```

3. **Run the notebook**:
   ```bash
   jupyter notebook SentimentScope_starter.ipynb
   ```

### Training Parameters

- **Batch Size**: 32
- **Learning Rate**: 3e-4
- **Epochs**: 3 (can be increased for better performance)
- **Optimizer**: AdamW
- **Loss Function**: Cross-Entropy Loss
- **Max Sequence Length**: 128 tokens

## 📊 Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 77.36% |
| **Validation Accuracy** | 77.88% |
| **Training Loss** | 0.41 (final epoch) |

### Training Progress

```
Epoch 1 - Validation Accuracy: 70.84%
Epoch 2 - Validation Accuracy: 76.16%
Epoch 3 - Validation Accuracy: 77.88%
```

## 🔧 Project Structure

```
sentiment-scope/
├── README.md                    # This file
├── SentimentScope_starter.ipynb # Main notebook
├── aclImdb_v1.tar.gz           # Dataset (download separately)
└── aclImdb/                    # Extracted dataset
    ├── train/
    │   ├── pos/
    │   └── neg/
    └── test/
        ├── pos/
        └── neg/
```

## 📚 Implementation Details

### Data Processing Pipeline

1. **Text Loading**: Custom `load_dataset()` function reads all `.txt` files
2. **Tokenization**: BERT-base-uncased tokenizer with subword tokenization
3. **Padding/Truncation**: All sequences padded/truncated to 128 tokens
4. **Data Splitting**: 90/10 train/validation split with shuffling

### Model Architecture

The transformer model consists of:

- **4 Transformer Blocks**: Each with multi-head attention and feed-forward layers
- **Mean Pooling**: Aggregates token-level representations
- **Classification Head**: Linear layer for binary classification
- **Residual Connections**: Skip connections for gradient flow
- **Layer Normalization**: Applied before each sub-layer

### Training Strategy

- **Gradient Clipping**: Prevents exploding gradients
- **Dropout**: 0.1 dropout rate for regularization
- **Learning Rate Scheduling**: AdamW optimizer with warmup
- **Early Stopping**: Based on validation accuracy

## 🎯 Future Improvements

- [ ] **Extended Training**: Increase epochs for better performance
- [ ] **Larger Model**: Experiment with more layers/heads
- [ ] **Hyperparameter Tuning**: Grid search for optimal parameters
- [ ] **Advanced Tokenization**: Try different tokenizers
- [ ] **Data Augmentation**: Text augmentation techniques
- [ ] **Ensemble Methods**: Combine multiple models
- [ ] **Visualization**: Add training curves and confusion matrix

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [IMDB Dataset](http://ai.stanford.edu/~amaas/data/sentiment/) by Maas et al.
- [Hugging Face Transformers](https://huggingface.co/transformers/) for tokenization
- [PyTorch](https://pytorch.org/) for deep learning framework
- [Udacity](https://www.udacity.com/) for project structure and guidance

## 📞 Contact

For questions or suggestions, please open an issue or contact [sankarambati1266@gmail.com].

---

**Note**: This project is part of the AWS Machine Learning Engineer Nanodegree program and demonstrates practical implementation of transformer models for sentiment analysis.
