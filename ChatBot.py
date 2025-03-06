import tensorflow as tf
import random

# Load trained model
model = tf.keras.models.load_model("improved_chatbot_model.h5")

def chatbot_response(text):
    tokenized = word_tokenize(text.lower())
    lemmatized = [lemmatizer.lemmatize(w) for w in tokenized]
    seq = tokenizer.texts_to_sequences([" ".join(lemmatized)])
    padded = pad_sequences(seq, maxlen=max_length, padding="post")
    
    prediction = model.predict(padded)
    tag = encoder.inverse_transform([np.argmax(prediction)])[0]
    
    return random.choice(responses[tag])

# Chat loop
print("Chatbot is ready! (Type 'quit' to exit)")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    print("Bot:", chatbot_response(user_input))
