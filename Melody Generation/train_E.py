import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import os

# Setting GPU memory to grow dynamically
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except RuntimeError as e:
        print(e)

with open("mapping.json", "r") as f:
    mappings = json.load(f)
    OUTPUT_UNITS = len(mappings)

NUM_UNITS = [128, 256]
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
EPOCHS = 5
BATCH_SIZE = 64
SEQUENCE_LENGTH = 64
SAVE_MODEL_PATH = "model.h5"
DATASET_FILE = "file_dataset"


# generation data
def generate_training_sequences(seq_length, batch_size):
    with open(DATASET_FILE, "r") as f:
        songs = f.read()

    # mapping
    int_songs = [mappings[s] for s in songs.split() if s in mappings]

    inputs = []
    targets = []
    for i in range(len(int_songs) - seq_length):
        inputs.append(int_songs[i:i+seq_length])
        targets.append(int_songs[i+seq_length])

        if len(inputs) == batch_size:
            yield np.array(inputs, dtype=np.int32), np.array(targets, dtype=np.int32)
            inputs, targets = [], []

    if len(inputs) > 0:
        yield np.array(inputs, dtype=np.int32), np.array(targets, dtype=np.int32)


def generator_wrapper():
    return generate_training_sequences(SEQUENCE_LENGTH, BATCH_SIZE)


# neuron network
def build_model(sequence_length, vocabulary_size):
    model = keras.Sequential([
        # int --> vectors
        keras.layers.Embedding(
            input_dim=vocabulary_size,
            output_dim=64,
            input_length=sequence_length
            ),
        
        keras.layers.Bidirectional(keras.layers.LSTM(
            NUM_UNITS[0], 
            return_sequences=True),
            merge_mode='concat'    # forward + backward
        ),
        keras.layers.Dropout(0.2),
        
        keras.layers.LSTM(NUM_UNITS[1]),
        keras.layers.Dropout(0.3),
        
        # full connect
        keras.layers.Dense(vocabulary_size, activation='softmax')
    ])
    
    return model


def train():
    # Creating Data Sets
    dataset = tf.data.Dataset.from_generator(
        #  lambda: generate_training_sequences(SEQUENCE_LENGTH, BATCH_SIZE),
        generator_wrapper,    
        output_signature=( 
            tf.TensorSpec(shape=(None, SEQUENCE_LENGTH), dtype=tf.int32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )
    ).prefetch(tf.data.AUTOTUNE)

    model = build_model(SEQUENCE_LENGTH, OUTPUT_UNITS)

    callbacks = [
        keras.callbacks.EarlyStopping(patience=5),
        keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True)
    ]
    
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'] 
                  )

    # Train 
    model.fit(dataset, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks)

    # Save the model different with weight
    model.save(SAVE_MODEL_PATH)

if __name__ == "__main__":
    tf.keras.backend.clear_session()
    train()
