import tensorflow as tf
import numpy as np
import json
from tensorflow import tokenizer_from_json

# Load the tokenizer from the JSON file
with open('tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

# Categories
categories = ['Bathroom', 'Bedroom', 'Dining', 'Kitchen', 'Livingroom']

# Load the trained model
model = tf.keras.models.load_model('text_classification_model.keras')

# Function to predict the category of a given sentence
def predict_category(sentence):
    sequence = tokenizer.texts_to_sequences([sentence])
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=100)
    prediction = model.predict(padded_sequence)
    category_index = np.argmax(prediction, axis=1)[0]
    return categories[category_index]

# Main loop to get user input and predict category
while True:
    user_input = input("Enter a sentence describing a room (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    category = predict_category(user_input)
    print(f"The described room is: {category}")
