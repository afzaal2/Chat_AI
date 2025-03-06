def chatbot_response(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_length, padding="post")
    prediction = model.predict(padded)
    tag = unique_tags[np.argmax(prediction)]
    return random.choice(responses[tag])

# Chat Loop
print("Chatbot is ready! (Type 'quit' to exit)")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    print("Bot:", chatbot_response(user_input))
