# MNIST Digit Classification — ANN from Scratch

## Project Overview
This project implements a fully connected Artificial Neural Network (ANN) from scratch using only NumPy, without any deep learning frameworks like TensorFlow or PyTorch. The model is trained on the MNIST dataset to classify handwritten digits (0–9).

---

## Dataset
- **Name:** MNIST (Modified National Institute of Standards and Technology)
- **Total Samples:** 42,000 images
- **Training Set:** 80% (~33,600 samples)
- **Validation Set:** 20% (~8,400 samples)
- **Input Size:** 28×28 grayscale images → flattened to 784 pixels
- **Output Classes:** 10 (digits 0 to 9)

---

## Architecture

```
Input Layer        →  784 neurons  (28×28 pixels flattened)
Hidden Layer 1     →  128 neurons  (ReLU activation)
Hidden Layer 2     →  64 neurons   (ReLU activation)
Output Layer       →  10 neurons   (Softmax activation)
```

### Layer Details:
| Layer | Input Size | Output Size | Activation |
|---|---|---|---|
| Hidden Layer 1 | 784 | 128 | ReLU |
| Hidden Layer 2 | 128 | 64 | ReLU |
| Output Layer | 64 | 10 | Softmax |

---

## Implementation Details

### 1. Parameter Initialization
- Weights initialized using **He Initialization** to prevent vanishing/exploding gradients
- Biases initialized to **zeros**

```python
W1 = np.random.randn(128, 784) * np.sqrt(2/784)
W2 = np.random.randn(64, 128)  * np.sqrt(2/128)
W3 = np.random.randn(10, 64)   * np.sqrt(2/64)
B1 = np.zeros((128, 1))
B2 = np.zeros((64, 1))
B3 = np.zeros((10, 1))
```

### 2. Activation Functions
- **ReLU** (Rectified Linear Unit) — Hidden layers
```python
def ReLU(X):
    return np.maximum(X, 0)
```

- **Softmax** — Output layer (with numerical stability fix)
```python
def softmax(Z):
    Z = Z - np.max(Z, axis=0, keepdims=True)
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)
    return A
```

### 3. Forward Propagation
```
Z1 = W1·X + B1  →  A1 = ReLU(Z1)
Z2 = W2·A1 + B2 →  A2 = ReLU(Z2)
Z3 = W3·A2 + B3 →  A3 = Softmax(Z3)
```

### 4. Loss Function
- **Cross Entropy Loss** (implicit in backpropagation via one-hot encoding)

### 5. Backward Propagation
- Gradients computed manually using chain rule
- **ReLU derivative:** `(Z > 0)` — 1 if positive, 0 if negative
- **Softmax + Cross Entropy derivative:** `A - Y_one_hot`

### 6. Parameter Update — Gradient Descent
```python
W = W - learning_rate * dW
B = B - learning_rate * dB
```

---

## Training Configuration
| Parameter | Value |
|---|---|
| Learning Rate | 0.30 |
| Iterations | 500 |
| Optimizer | Vanilla Gradient Descent |
| Batch Type | Full Batch |

---

## Data Preprocessing
- **Normalization:** Pixel values divided by 255.0 (range 0–1)
- **Shuffling:** Data shuffled before train/val split
- **One-Hot Encoding:** Labels converted to one-hot vectors for backpropagation

---

## Results
| Metric | Value |
|---|---|
| Training Accuracy | ~91–95% |
| Validation Accuracy | ~91–95% |

---

## Bugs Fixed During Development
1. **Return inside loop** — `return` statement was inside the training loop causing only 1 iteration
2. **Global `m` variable** — Used total dataset size instead of batch size in gradients
3. **Softmax axis** — `np.sum` was summing entire matrix instead of per sample (`axis=0`)
4. **Dying ReLU** — Weights initialized with `-0.5` shift causing all neurons to output 0
5. **NaN values** — Caused by unnormalized pixel values (0–255) exploding gradients
6. **`dB1` shape** — Missing `axis=1, keepdims=True` causing shape mismatch

---

## Key Learnings
- **He Initialization** is critical for deep networks with ReLU
- **Normalization** (dividing by 255) is essential — without it gradients explode
- **Softmax stability** — Subtracting max before exp prevents overflow
- **More layers ≠ better** — Overfitting and vanishing gradients occur beyond a sweet spot
- This is an **ANN, not a CNN** — No convolutional layers, spatial structure of image is not utilized

---

## Limitations
- Flat pixel input — spatial relationships between pixels ignored
- Sensitive to handwriting style (European 8 vs MNIST 8)
- No data augmentation — rotation, shift not handled
- Vanilla gradient descent — no Adam/momentum optimization

---

## Tech Stack
- **Language:** Python
- **Libraries:** NumPy, Matplotlib, PIL
- **Platform:** Google Colab
- **Dataset:** MNIST (via Kaggle CSV format)

---

*Built from scratch without any deep learning framework — pure NumPy implementation.*
