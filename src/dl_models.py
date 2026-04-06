import tensorflow as tf
from tensorflow.keras import layers, Model

def build_gru(vocab_size):
    inputs = layers.Input(shape=(None,))
    x = layers.Embedding(vocab_size, 128)(inputs)
    x = layers.GRU(64)(x)
    outputs = layers.Dense(3, activation="softmax")(x)
    return Model(inputs, outputs)

def build_bilstm(vocab_size):
    inputs = layers.Input(shape=(None,))
    x = layers.Embedding(vocab_size, 128)(inputs)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    outputs = layers.Dense(3, activation="softmax")(x)
    return Model(inputs, outputs)
