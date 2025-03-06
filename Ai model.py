import json
import random
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Load intents data
with open("intents.json", "r") as file:
    data = json.load(file)

# Initialize NLP tools
nltk.download("punkt")
nltk.download("wordnet")
lemmatizer = WordNetLemmatizer()
tokenizer = Tokenizer()

# Prepare data
patterns, tags, responses = [], [], {}
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokenized_words = word_tokenize(pattern.lower())  # Tokenize
        lemmatized_words = [lemmatizer.lemmatize(w) for w in tokenized_words]  # Lemmatize
        sentence = " ".join(lemmatized_words)
        patterns.append(sentence)
        tags.append(intent["tag"])
    responses[intent["tag"]] = intent["responses"]

# Tokenize words
tokenizer.fit_on_texts(patterns)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(patterns)
max_length = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen=max_length, padding="post")

# Encode tags
encoder = LabelEncoder()
y = encoder.fit_transform(tags)

# Build improved LSTM model
model = Sequential([
    Embedding(input_dim=len(word_index) + 1, output_dim=32, input_length=max_length),
    Bidirectional(LSTM(32, return_sequences=True)),
    Bidirectional(LSTM(32)),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(len(set(tags)), activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X, y, epochs=300, batch_size=8)

# Save the model
model.save("improved_chatbot_model.h5")
