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

