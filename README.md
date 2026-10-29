# 📝 Sentiment140 Twitter Sentiment Analysis

## 📌 Project Overview

This project explores **sentiment analysis**  using the **Sentiment140 dataset** from Kaggle. The goal is to classify tweets as **positive** or **negative** using **Deep Learning models**.

We experimented with two architectures:

* 🌀 **Bi-directional LSTM (BiLSTM)** for sequential word dependencies
* 🔲 **Convolutional Neural Network (CNN)** for feature extraction from word embeddings

👉 Both models performed better than traditional approaches (like VADER), but BiLSTM captured context more effectively.

---

## 📂 Dataset

The dataset is available on Kaggle:
🔗 [Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)

* **1.6M tweets** 📊
* Labeled as:

  * `0` → Negative 😡
  * `4` → Positive 😊

---

## ⚙️ Setup & How to Run

This project was developed in **Kaggle Notebook** 💻. To run it:

1. Open Kaggle and create a new notebook.
2. Attach the Sentiment140 dataset.
3. Copy the code cells from this repo (or adapt snippets below).
4. Run all cells 🚀.

---

## 🔑 Key Code Snippets

### 1️⃣ Tokenization & Padding

```python
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

tokenizer = Tokenizer(num_words=50000)
tokenizer.fit_on_texts(X_train)

X_train_padded = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=100)
X_test_padded  = pad_sequences(tokenizer.texts_to_sequences(X_test),  maxlen=100)
```

### 2️⃣ BiLSTM Model

```python
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout

model = Sequential([
    Embedding(50000, 128, input_length=100),
    Bidirectional(LSTM(64)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
```

### 3️⃣ CNN Model

```python
from keras.layers import Conv1D, GlobalMaxPooling1D

model = Sequential([
    Embedding(50000, 128, input_length=100),
    Conv1D(128, 5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
```

---

## 📊 Results

Test-set results (VADER is scored on the full 1.6M tweets):

| Model     | Loss | Accuracy | Observation                             |
| --------- | ---- | -------- | --------------------------------------- |
| 🌀 BiLSTM | 0.41 | 82.1%    | Captures context, better generalization |
| 🔲 CNN    | 0.48 | 81.0%    | Faster, simpler, but less context-aware |
| 📊 VADER  | -    | 65.2%    | Good baseline, but weaker than DL       |

---

## 📌 Observations

* ✅ **CNN**: Achieved **81.0% accuracy** (loss 0.48). Better than VADER, but lacks deep context.
* ✅ **BiLSTM**: Achieved **82.1% accuracy** (loss 0.41). Outperforms CNN by capturing sequential dependencies.
* ⚠️ Both models start **overfitting after ~2–3 epochs** 📉 (validation loss rises while training loss keeps falling). Early stopping and stronger regularization would help.

---

## 🚀 Future Improvements

* Add **attention mechanisms** for better interpretability.
* Try **transformer-based models** like BERT 🤖.
* Use **data augmentation** to reduce overfitting.

---

✨ **This project shows how deep learning outperforms traditional sentiment analysis tools on real-world Twitter data.**
