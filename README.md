---

# 🧠 Twitter Sentiment Analysis with Deep Learning

This project focuses on **Sentiment Analysis** 📝 of Twitter data using the [Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140).
We build and evaluate models ranging from **traditional methods (VADER)** to **deep learning (CNN, LSTM, BiLSTM, GRU)** to predict whether a tweet is **positive 😀** or **negative 😡**.

---

## 📊 Dataset

* Source: [Sentiment140 Kaggle Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)
* **1.6M tweets** labeled for sentiment:

  * `0` → Negative 😠
  * `4` → Positive 😄
* Preprocessing included:

  * Removing stopwords & special characters
  * Tokenization ✂️
  * Padding for sequence length 📏

---

## ⚙️ Project Workflow

1. **Data Preprocessing** 🧹

   ```python
   import re
   from nltk.corpus import stopwords

   def clean_text(text):
       text = re.sub(r'http\S+', '', text)      # remove links
       text = re.sub(r'[^a-zA-Z]', ' ', text)   # remove special chars
       text = text.lower()                      # lowercase
       text = ' '.join([w for w in text.split() if w not in stopwords.words('english')])
       return text
   ```

   * Tweets cleaned and padded.

2. **Train-Test Split** ✂️

   ```python
   from sklearn.model_selection import train_test_split

   X_train, X_test, y_train, y_test = train_test_split(
       df['clean_text'], df['target'], test_size=0.2, random_state=42
   )
   ```

3. **Embedding & Tokenization** 🔤

   ```python
   from keras.preprocessing.text import Tokenizer
   from keras.utils import pad_sequences

   tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
   tokenizer.fit_on_texts(X_train)

   X_train_seq = tokenizer.texts_to_sequences(X_train)
   X_test_seq = tokenizer.texts_to_sequences(X_test)

   X_train_padded = pad_sequences(X_train_seq, maxlen=50, padding="post")
   X_test_padded = pad_sequences(X_test_seq, maxlen=50, padding="post")
   ```

4. **CNN Model** 🏗️

   ```python
   from keras.models import Sequential
   from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout

   model = Sequential([
       Embedding(input_dim=5000, output_dim=128, input_length=50),
       Conv1D(128, 5, activation="relu"),
       GlobalMaxPooling1D(),
       Dense(64, activation="relu"),
       Dropout(0.5),
       Dense(1, activation="sigmoid")  # Binary classification
   ])

   model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
   ```

5. **Training** 🏃

   ```python
   history = model.fit(
       X_train_padded, y_train,
       validation_split=0.1,
       epochs=10,
       batch_size=128,
       verbose=1
   )
   ```

---

## 📈 Results

* **CNN Model Performance**:

  * Loss: `0.4877`
  * Accuracy: `81%` ✅

* **Observation**:

  * The CNN outperformed **VADER**, but it still **isn’t ideal** for capturing the full context of tweets.
  * Overfitting started after **5 epochs** 📉.

---

## 🚀 How to Run

Since this is a **Kaggle Notebook**, you can:

1. Open the notebook in Kaggle 🧑‍💻.
2. Upload the dataset (or connect directly via Kaggle Datasets).
3. Run the notebook **cell by cell** ▶️.
4. Modify hyperparameters (embedding size, sequence length, model type).
5. Check outputs directly in Kaggle logs & plots.

---

## 🔮 Next Steps

* Try **LSTM / BiLSTM / GRU** for better sequence understanding.
* Use **Pre-trained embeddings (GloVe / Word2Vec / BERT)** for richer representations.
* Apply **regularization & dropout** to reduce overfitting.
* Explore **transformers (BERT, RoBERTa, DistilBERT)** for state-of-the-art results.

---

## 🤝 Contribution

Feel free to fork, experiment, and enhance!
Pull requests are welcome 🚀

---

Would you like me to also add **a comparison table** 📊 between **VADER vs CNN vs LSTM (planned)** in the README so it looks more professional?
