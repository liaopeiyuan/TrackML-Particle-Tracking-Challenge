from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential
from keras import regularizers
import numpy
from scipy.io import loadmat
import tensorflow as tf
from sklearn.preprocessing import normalize, scale
# fix random seed for reproducibility
numpy.random.seed(7)
X=numpy.loadtxt('feature.csv',dtype='float',delimiter=',')
Y=numpy.loadtxt('label.csv',dtype='float',delimiter=',')
X=X[1:100000,:]
Y=Y[1:100000,:]
X=normalize(X,axis=0)
X = scale( X, axis=0, with_mean=True, with_std=True, copy=True )
print(type(X))
print(type(Y))

def nn_1(input_length):
    input_layer = Input(shape=(input_length,))
    encoded = Dense(256, activation="tanh")(input_layer)

    encoded = Dense(128, activation="tanh",activity_regularizer=regularizers.l2(0.01))(encoded)
    encoded = Dropout(0.5, noise_shape=None, seed=None)(encoded)
    encoded = Dense(96, activation="tanh",activity_regularizer=regularizers.l2(0.01))(encoded)
    encoded = Dense(64, activation="tanh",activity_regularizer=regularizers.l2(0.01))(encoded)
    encoded = Dropout(0.5, noise_shape=None, seed=None)(encoded)
    encoded = Dense(96, activation="tanh",activity_regularizer=regularizers.l2(0.01))(encoded)

    #encoded = Dense(128, activation="tanh")(encoded)
    #encoded = Dense(96, activation="tanh")(encoded)
    #encoded = Dense(64, activation="tanh")(encoded)
    #encoded = Dense(96, activation="tanh")(encoded)

    #encoded = Dense(128, activation="tanh")(encoded)
    #encoded = Dense(96, activation="tanh")(encoded)
    #encoded = Dense(64, activation="tanh")(encoded)
    #encoded = Dense(96, activation="tanh")(encoded)

    decoded = Dense(64, activation="tanh",activity_regularizer=regularizers.l2(0.01))(encoded)
    encoded = Dropout(0.5, noise_shape=None, seed=None)(encoded)
    decoded = Dense(96, activation="tanh",activity_regularizer=regularizers.l2(0.01))(decoded)
    encoded = Dropout(0.5, noise_shape=None, seed=None)(encoded)
    decoded = Dense(128, activation="tanh",activity_regularizer=regularizers.l2(0.01))(decoded)

    decoded = Dense(3, activation="linear")(decoded)
    # encoder = Model(input_layer, encoded)
    nn_predictor = Model(input_layer, decoded)
    nn_predictor.compile(optimizer="RMSprop", loss="mean_squared_error")  # mean_absolute_error ?
    return nn_predictor

nn_predictor = nn_1(3)

with tf.device('/gpu:0'):
    nn_predictor.fit(X,Y, batch_size=512, epochs=3000, validation_split=0.2,verbose=1)
nn_predictor.save('my_model.h5')
