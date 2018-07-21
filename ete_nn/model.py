import keras
from keras.layers import Input, Dense, BatchNormalization, Dropout, PReLU, Conv1D
from keras.models import Model

def get_basic_nn(input_size=9):
    nn_list = [Input(shape=(input_size,))]
    for layer in [
        Dense(32), BatchNormalization(), PReLU(),
        Dense(64), BatchNormalization(), PReLU(),
        Dense(75), BatchNormalization(), PReLU(),
        Dense(100), BatchNormalization(), PReLU(),
        Dense(128), BatchNormalization(), PReLU(),
        Dense(128), BatchNormalization(), PReLU(),
        Dense(128), BatchNormalization(), PReLU(),
        Dense(128), BatchNormalization(), PReLU(),
        Dense(128), BatchNormalization(), PReLU(),
        Dense(110), BatchNormalization(), PReLU(),
        Dense(75), BatchNormalization(), PReLU(),
        Dense(64), BatchNormalization(), PReLU(),
    ]:
        nn_list.append(layer(nn_list[-1]))
    return nn_list

def MLP_with_dropout(input_size=9, rate=0.5):
    nn_list = [Input(shape=(input_size,))]
    for layer in[
        Dense(32), BatchNormalization(), PReLU(),
        Dropout(rate),
        Dense(64), BatchNormalization(), PReLU(),
        Dense(128), BatchNormalization(), PReLU(),
        Dropout(rate),
        Dense(256), BatchNormalization(), PReLU(),
        Dropout(rate),
        Dense(512), BatchNormalization(), PReLU(),
        Dense(1024), BatchNormalization(), PReLU(),
        Dropout(rate),
        Dense(2048), BatchNormalization(), PReLU(),
        Dropout(rate),
        Dense(4096), BatchNormalization(), PReLU(),
        Dense(4096), BatchNormalization(), PReLU(),
        Dropout(rate),
        Dense(2048), BatchNormalization(), PReLU(),
        Dense(1024), BatchNormalization(), PReLU(),
        Dropout(rate),
        Dense(512), BatchNormalization(), PReLU(),
        Dense(256), BatchNormalization(), PReLU(),
        Dropout(rate),
        Dense(128), BatchNormalization(), PReLU(),
        Dropout(rate),
        Dense(64), BatchNormalization(), PReLU(),
    ]:
      nn_list.append(layer(nn_list[-1]))
    return nn_list  

def basic_cnn(input_size=9):
    nn_list = [Input(shape=(input_size,1,))]
    for layer in[
        Conv1D(kernel_size=(1), filters=50, strides=1, padding='same',
        activation="relu", kernel_initializer="glorot_normal"),
        BatchNormalization(),
        Conv1D(kernel_size=(1), filters=100, strides=1, padding='same',
        activation="relu", kernel_initializer="glorot_normal"),
        BatchNormalization(),
        Conv1D(kernel_size=(1), filters=200, strides=1, padding='same',
        activation="relu", kernel_initializer="glorot_normal"),
        BatchNormalization(),
        Conv1D(kernel_size=(1), filters=400, strides=1, padding='same',
        activation="relu", kernel_initializer="glorot_normal"),
        BatchNormalization(),
        Conv1D(kernel_size=(1), filters=800, strides=1, padding='same',
        activation="relu", kernel_initializer="glorot_normal"),
        BatchNormalization(),
        Conv1D(kernel_size=(1), filters=1600, strides=1, padding='same',
        activation="relu", kernel_initializer="glorot_normal"),
        BatchNormalization(),
        Conv1D(kernel_size=(1), filters=3200, strides=1, padding='same',
        activation="relu", kernel_initializer="glorot_normal"),
        BatchNormalization(),
        Conv1D(kernel_size=(1), filters=6400, strides=1, padding='same',
        activation="relu", kernel_initializer="glorot_normal"),
        Conv1D(kernel_size=(1), filters=12800, strides=1, padding='same',
        activation="relu", kernel_initializer="glorot_normal"),
        BatchNormalization(),
        Conv1D(kernel_size=(1), filters=5000, strides=1, padding='same',
        activation="relu", kernel_initializer="glorot_normal"),
        BatchNormalization(),
        Conv1D(kernel_size=(1), filters=3000, strides=1, padding='same',
        activation="relu", kernel_initializer="glorot_normal"),
        Conv1D(kernel_size=(1), filters=1500, strides=1, padding='same',
        activation="relu", kernel_initializer="glorot_normal"),
        BatchNormalization(),
        Conv1D(kernel_size=(1), filters=64, strides=1, padding='same',
        activation="relu", kernel_initializer="glorot_normal"),
        BatchNormalization(),
    ]:
        nn_list.append(layer(nn_list[-1]))
    return nn_list