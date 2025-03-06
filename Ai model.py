import json
import random
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load intents file
with open("intents.json", "r") as file:
    data = json.load(file)

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
tokenizer = Tokenizer()

# Prepare data
patterns = []
tags = []
responses = {}
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        tags.append(intent["tag"])
    responses[intent["tag"]] = intent["responses"]

# Tokenize words
tokenizer.fit_on_texts(patterns)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(patterns)
max_length = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen=max_length, padding="post")

# Encode tags
unique_tags = list(set(tags))
tag_index = {tag: i for i, tag in enumerate(unique_tags)}
y = np.array([tag_index[tag] for tag in tags])

# Build Model
model = Sequential([
    Embedding(input_dim=len(word_index) + 1, output_dim=16, input_length=max_length),
    LSTM(16, return_sequences=True),
    LSTM(16),
    Dense(16, activation="relu"),
    Dense(len(unique_tags), activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train Model
model.fit(X, y, epochs=200, batch_size=8)
model.save("chatbot_model.h5")
