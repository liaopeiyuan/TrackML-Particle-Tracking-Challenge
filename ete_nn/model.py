import keras
from keras.layers import AveragePooling1D, Input, Dense, BatchNormalization, Dropout, PReLU, Conv1D, Reshape, Flatten
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
        Dropout(rate),
        Dense(512), BatchNormalization(), PReLU(),
        Dropout(rate),
        Dense(128), BatchNormalization(), PReLU(),
        Dropout(rate),
        Dense(64), BatchNormalization(), PReLU(),
    ]:
      nn_list.append(layer(nn_list[-1]))
    return nn_list  

def complex_cnn(input_size=9):
    nn_list = [Input(shape=(input_size,))]
    for layer in[
        Dense(16384), Reshape((128, 128)),
        Conv1D(kernel_size=(1), filters=100, strides=1, padding='same',
        activation="relu", kernel_initializer="glorot_normal"),
        Conv1D(kernel_size=(1), filters=200, strides=1, padding='same',
        activation="relu", kernel_initializer="glorot_normal"),
        BatchNormalization(), AveragePooling1D(2),
        Conv1D(kernel_size=(2), filters=200, strides=1, padding='same',
        activation="relu", kernel_initializer="glorot_normal"),
        Conv1D(kernel_size=(2), filters=300, strides=1, padding='same',
        activation="relu", kernel_initializer="glorot_normal"),
        BatchNormalization(),AveragePooling1D(2),
        Conv1D(kernel_size=(4), filters=300, strides=1, padding='same',
        activation="relu", kernel_initializer="glorot_normal"),
        Conv1D(kernel_size=(4), filters=400, strides=1, padding='same',
        activation="relu", kernel_initializer="glorot_normal"),
        Conv1D(kernel_size=(4), filters=400, strides=1, padding='same',
	activation='relu', kernel_initializer='glorot_normal')
        BatchNormalization(),AveragePooling1D(2),
        activation="relu", kernel_initializer="glorot_normal"),
        Conv1D(kernel_size=(8), filters=300, strides=1, padding='same',
        activation="relu", kernel_initializer="glorot_normal"),
        Conv1D(kernel_size=(8), filters=500, strides=1, padding='same',
        activation="relu", kernel_initializer="glorot_normal"),
        BatchNormalization(),AveragePooling1D(2),
        Conv1D(kernel_size=(16), filters=150, strides=1, padding='same',
        activation="relu", kernel_initializer="glorot_normal"),
        Conv1D(kernel_size=(16), filters=600, strides=1, padding='same',
        activation="relu", kernel_initializer="glorot_normal"),
        BatchNormalization(),AveragePooling1D(4),
        Conv1D(kernel_size=(64), filters=250, strides=1, padding='same',
        activation="relu", kernel_initializer="glorot_normal"),
        Conv1D(kernel_size=(64), filters=300, strides=1, padding='same',
        activation="relu", kernel_initializer="glorot_normal"),
        BatchNormalization(),AveragePooling1D(2),
        Conv1D(kernel_size=(128), filters=300, strides=1, padding='same',
        activation="relu", kernel_initializer="glorot_normal"),
        BatchNormalization(),
        Conv1D(kernel_size=(128), filters=300, strides=1, padding='same',
        activation="relu", kernel_initializer="glorot_normal"),
        Conv1D(kernel_size=(128), filters=600, strides=1, padding='same',
        activation="relu", kernel_initializer="glorot_normal"),
        BatchNormalization(), Flatten(),
        Dense(64)
    ]:
        nn_list.append(layer(nn_list[-1]))
    return nn_list

def basic_cnn(input_size=9):
    nn_list = [Input(shape=(input_size,))]
    for layer in[
        Dense(10000), Reshape((100, 100)),
        Conv1D(kernel_size=(20), filters=20, strides=1, padding='same',
        activation="relu", kernel_initializer="glorot_normal"),
        BatchNormalization(),
        Conv1D(kernel_size=(20), filters=40, strides=1, padding='same',
        activation="relu", kernel_initializer="glorot_normal"),
        BatchNormalization(),
        Conv1D(kernel_size=(40), filters=40, strides=1, padding='same',
        activation="relu", kernel_initializer="glorot_normal"),
        BatchNormalization(),
        Conv1D(kernel_size=(50), filters=50, strides=1, padding='same',
        activation="relu", kernel_initializer="glorot_normal"),
        BatchNormalization(),
        Conv1D(kernel_size=(20), filters=60, strides=1, padding='same',
        activation="relu", kernel_initializer="glorot_normal"),
        BatchNormalization(),
        Conv1D(kernel_size=(15), filters=70, strides=1, padding='same',
        activation="relu", kernel_initializer="glorot_normal"),
        BatchNormalization(),
        Conv1D(kernel_size=(10), filters=40, strides=1, padding='same',
        activation="relu", kernel_initializer="glorot_normal"),
        Conv1D(kernel_size=(15), filters=20, strides=1, padding='same',
        activation="relu", kernel_initializer="glorot_normal"),
        BatchNormalization(),
        Conv1D(kernel_size=(5), filters=10, strides=1, padding='same',
        activation="relu", kernel_initializer="glorot_normal"),
        BatchNormalization(), Flatten(),
        Dense(64)
    ]:
        nn_list.append(layer(nn_list[-1]))
    return nn_list


def MLP(input_size=9, rate=0.5):
    nn_list = [Input(shape=(input_size,))]
    for layer in[
        Dense(32, kernel_initializer='RandomUniform'), BatchNormalization(), PReLU(),
        #Dropout(rate),
        Dense(64, kernel_initializer='RandomUniform'), BatchNormalization(), PReLU(),
        Dense(128, kernel_initializer='RandomUniform'), BatchNormalization(), PReLU(),
        Dense(128, kernel_initializer='RandomUniform'), BatchNormalization(), PReLU(),
#        Dense(128, kernel_initializer='RandomUniform'), BatchNormalization(), PReLU(),
        Dense(128, kernel_initializer='RandomUniform'), BatchNormalization(), PReLU(),
        #Dropout(rate),
        Dense(256, kernel_initializer='RandomUniform'), BatchNormalization(), PReLU(),
        Dense(256, kernel_initializer='RandomUniform'), BatchNormalization(), PReLU(),
        Dense(256, kernel_initializer='RandomUniform'), BatchNormalization(), PReLU(),
        Dropout(rate),
        Dense(512, kernel_initializer='RandomUniform'), BatchNormalization(), PReLU(),
        Dense(512, kernel_initializer='RandomUniform'), BatchNormalization(), PReLU(),
        Dense(512, kernel_initializer='RandomUniform'), BatchNormalization(), PReLU(),
        Dense(512, kernel_initializer='RandomUniform'), BatchNormalization(), PReLU(),
        Dropout(rate),
        Dense(1024, kernel_initializer='RandomUniform'), BatchNormalization(), PReLU(),
        Dense(1024, kernel_initializer='RandomUniform'), BatchNormalization(), PReLU(),
        Dense(1024, kernel_initializer='RandomUniform'), BatchNormalization(), PReLU(),
#	Dropout(rate),
#	Dense(1500, kernel_initializer='RandomUniform'), BatchNormalization(), PReLU(),
#	Dense(1500, kernel_initializer='RandomUniform'), BatchNormalization(), PReLU(),
    ]:
      nn_list.append(layer(nn_list[-1]))
    return nn_list
