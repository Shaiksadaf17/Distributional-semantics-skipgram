# Distributional Semantics with Skip-gram Word Embeddings

This repository implements a distributional semantics model using the Skip-gram
architecture to learn dense word embeddings from text data. The project
demonstrates how semantic similarity between words can emerge purely from
contextual co-occurrence statistics.

---

## Motivation

Understanding how words acquire meaning from context is a core problem in
Natural Language Processing (NLP). Distributional semantics is based on the
principle that words appearing in similar contexts tend to have similar
meanings.

This project builds word embeddings from scratch, without relying on
pre-trained models, to provide a transparent and educational implementation
of the Skip-gram model.

---

## Approach

The project follows a standard Skip-gram pipeline:

1. Text preprocessing and tokenization  
2. Vocabulary construction  
3. Skip-gram neural network training  
4. Embedding extraction  
5. Semantic similarity evaluation using cosine similarity  

The model is trained in an unsupervised manner on raw text data.

---

## Dataset Structure

The dataset is split into training, validation, and test sets to support
reproducibility and fair evaluation:
data/
├── train.csv
├── val.csv
└── test.csv

Each CSV file contains a single column:

csv
text
this is an example sentence
distributional semantics captures meaning

---
Purpose of Splits
Training set: used to learn word embeddings
Validation set: used for hyperparameter tuning
Test set: used for qualitative evaluation of embedding quality

---
Model Architecture

Input layer: one-hot encoded target word
Hidden layer: dense embedding layer
Output layer: context word probability distribution

The model is trained using negative sampling and optimized with stochastic
gradient descent.

 ---
Evaluation

Since Skip-gram is an unsupervised model, evaluation focuses on semantic quality
rather than classification accuracy. Evaluation methods include:
Cosine similarity between word vectors
Nearest-neighbor word analysis
Qualitative inspection of learned embeddings
All evaluations are performed on held-out test data.

---
Technologies Used

1. Python

2. PyTorch

3. NumPy

4. Pandas

5. Matplotlib

---

How to Run
1. Clone the repository
git clone https://github.com/your-username/distributional-semantics-skipgram.git
cd distributional-semantics-skipgram

2. Install dependencies
pip install -r requirements.txt

3. Run the notebook
jupyter notebook

Open distributional_semantics_skipgram.ipynb to train and evaluate the model.
---
Results

The trained embeddings capture meaningful semantic relationships between words,
demonstrating the effectiveness of the Skip-gram model for learning
distributional representations from text.
---
Future Work:

Train on larger and more diverse corpora

Add intrinsic evaluation benchmarks

Extend to CBOW and GloVe models

Visualize embeddings using PCA or t-SNE
