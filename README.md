# Image Similarity Retrieval using Deep Embeddings

An end-to-end deep learning system for **visual similarity search**, built using  
**metric learning, cosine similarity, and FastAPI deployment**.

---

## Problem Overview

The objective of this project is to design an **image similarity retrieval pipeline** that:

- Converts images into **dense vector embeddings**
- Retrieves **visually similar images** instead of predicting class labels
- Ensures **similar instances lie close in embedding space**

This focuses on **representation learning + nearest-neighbor search**,  
not traditional image classification.

 

---

## End-to-End Pipeline

Complete workflow:

1. Feature extraction using pretrained **ResNet-50**
2. Embedding learning via **Triplet Loss**
3. Saving **model, embeddings, and labels**
4. Retrieval using **cosine similarity**
5. Deployment through **FastAPI**

 

---


## Technology Stack 🛠️
### Core Deep Learning
PyTorch — model training & inference
Torchvision — pretrained ResNet-50 and transforms
NumPy — numerical operations

### Data Processing & Visualization
Pandas — metadata handling
Pillow (PIL) — image loading
Matplotlib — retrieval visualization
tqdm — progress tracking

### Similarity Search & Deployment
Scikit-learn — cosine similarity computation
FastAPI — REST API deployment
Uvicorn — ASGI server

### Development Environment
Python 3.13
Jupyter Notebook — experimentation & training
PyCharm — modular project structure & API development
Git & GitHub — version control and submission 

---

## Training Strategies Explored

### Option A — Full Triplet Training (Slowest)

**Approach**

- Train entire ResNet-50 + embedding head end-to-end  
- Maximum flexibility but extremely compute-heavy  

**Observations**

- Training time: **many hours on CPU**
- High resource usage
- Not practical for CPU-only setup

 

---

### Option B — Light Fine-Tuning (Selected)

**Key Insight**

The task is **embedding + similarity retrieval**, not classification.

Therefore:

- Freeze pretrained **backbone**
- Train only the **embedding projection layer**

This preserves learned visual features while adapting to **similarity learning**.

**Performance**

- Training time: **~30–60 minutes on CPU**
- Stable convergence
- Strong retrieval accuracy

<img width="910" height="672" alt="image" src="https://github.com/user-attachments/assets/9a5844b6-8a54-4240-9218-a57f2b61999f" />


---

## Model Architecture

- **Backbone:** ResNet-50 (ImageNet pretrained)
- **Frozen convolutional layers**
- **Trainable embedding head**
  - Linear → 512-dimensional vector
  - ReLU activation
  - Batch Normalization
  - L2 normalization for cosine similarity

 <img width="910" height="639" alt="image" src="https://github.com/user-attachments/assets/2a6e6e39-72ca-4a44-9a8b-61a049d294ee" />


---



## Similarity Metric

We use **cosine similarity** because it:

- Works best with **normalized embeddings**
- Measures **angular similarity**
- Is standard for **metric learning & retrieval systems**

 
---

## Evaluation Protocol

Performance is reported using:

- **Top-1 Accuracy**
- **Top-5 Accuracy**
- **Top-10 Accuracy**

These metrics verify whether a **correct instance appears within Top-K retrieved results**.

<img width="270" height="88" alt="image" src="https://github.com/user-attachments/assets/c09b9c58-060f-4214-8cb2-dd7a0100641d" />


---

## Sample Retrieval Visualization ⭐

Demonstrates **qualitative correctness**:

- Query image  
- Top-K retrieved images  
- Correct matches highlighted  

<img width="929" height="414" alt="image" src="https://github.com/user-attachments/assets/06a12f2a-6d39-4a86-b258-3c1fab02d09f" />


> This is the **most important proof** for evaluation.

---

## Deployment via FastAPI

The trained model is exposed through an **API endpoint**:

- Upload query image
- Generate embedding
- Perform similarity search
- Return **Top-K matches as JSON**

### Run locally

```bash
uvicorn src.api.main:app --reload
```


This makes the system robust, scalable, and suitable for real-world similarity search applications.
