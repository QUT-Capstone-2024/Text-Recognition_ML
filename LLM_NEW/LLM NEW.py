import json
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import numpy as np

# Load the generated data
with open('synthetic_room_descriptions.json', 'r') as f:
    data = json.load(f)

# Prepare the data
texts = []
labels = []
categories = ['Bathroom', 'Bedroom', 'Dining', 'Kitchen', 'Livingroom']
label_map = {category: idx for idx, category in enumerate(categories)}

for category, sentences in data.items():
    texts.extend(sentences)
    labels.extend([label_map[category]] * len(sentences))

# Parameters
max_length = 100  # Maximum length of sequences
vocab_size = 10000  # Size of vocabulary
embedding_dim = 128  # Dimension of embedding vectors
batch_size = 256
epochs = 10

# Tokenize the text data
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length)

# Split the data into training, validation, and test sets
train_sequences, temp_sequences, train_labels, temp_labels = train_test_split(padded_sequences, labels, test_size=0.3, stratify=labels, random_state=42)
val_sequences, test_sequences, val_labels, test_labels = train_test_split(temp_sequences, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42)

print(f'Total samples: {len(texts)}')
print(f'Training set size: {len(train_sequences)}')
print(f'Validation set size: {len(val_sequences)}')
print(f'Test set size: {len(test_sequences)}')

# Create datasets
train_ds = tf.data.Dataset.from_tensor_slices((train_sequences, train_labels)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices((val_sequences, val_labels)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((test_sequences, test_labels)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

# Create a text classification model
def create_text_classification_model(vocab_size, embedding_dim, num_classes):
    model = tf.keras.Sequential([
        layers.Embedding(vocab_size, embedding_dim),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(32)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

num_classes = len(categories)

model = create_text_classification_model(vocab_size, embedding_dim, num_classes)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_ds,
    epochs=epochs,
    validation_data=val_ds,
    callbacks=[tf.keras.callbacks.ModelCheckpoint('best_model_text_classification.keras', monitor='val_accuracy', save_best_only=True, mode='max')]
)

# Evaluate on test data
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Save the model
model.save('text_classification_model.keras')
print("Model saved to 'text_classification_model.keras'")

# Save the tokenizer to a JSON file
tokenizer_json = tokenizer.to_json()
with open('tokenizer.json', 'w') as f:
    json.dump(tokenizer_json, f)

print("Tokenizer saved to 'tokenizer.json'")
