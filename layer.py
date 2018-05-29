"""
layer.py

experimental nn network

doesnt work at all

by Alexander Liao
"""

from keras.layers import Input, Dense, Dropout, BatchNormalization, PReLU
from keras.models import Model, Sequential
from keras import optimizers
from keras import regularizers
import numpy
from scipy.io import loadmat
import tensorflow as tf
from sklearn.preprocessing import normalize, scale
from sklearn.metrics import r2_score
from sklearn.preprocessing import minmax_scale
from datetime import datetime

# fix random seed for reproducibility
numpy.random.seed(7)
X=numpy.loadtxt('feature.csv',dtype='float',delimiter=',')
Y=numpy.loadtxt('label.csv',dtype='float',delimiter=',')
Xtrain=X[0:50000,:]
Ytrain=Y[0:50000,:]
Ｙtrain=minmax_scale(Ytrain, feature_range=(0, 1), axis=0, copy=True)
#Xtrain=normalize(Xtrain,axis=1)
Xtrain = scale( Xtrain, axis=0, with_mean=True, with_std=True, copy=True )
#Ytrain = scale( Ytrain, axis=0, with_mean=True, with_std=True, copy=True )
print(Xtrain)
Xtest=X[300000:310000,:]
Ytest=Y[300000:310000,:]
Ｙtest=minmax_scale(Ytest, feature_range=(0, 1), axis=0, copy=True)
#Xtest=normalize(Xtest,axis=1)
Xtest = scale( Xtest, axis=0, with_mean=True, with_std=True, copy=True )
#Ytest = scale( Ytest, axis=0, with_mean=True, with_std=True, copy=True )
print(type(X))
print(type(Y))

def nn_1(input_length):

    #PReLU()=PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)
    #BatchNormalization()=BatchNormalization()alization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
    #Dropout(0.5, noise_shape=None, seed=None)=Dropout(0.5, noise_shape=None, seed=None)(0.5, noise_shape=None, seed=None)
    model = Sequential()
    model.add(Dense(64, input_dim=input_length, init='RandomNormal'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.5))

    model.add(Dense(128, input_dim=input_length, init='RandomNormal'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.5))

    model.add(Dense(256, input_dim=input_length, init='RandomNormal'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.5))

    model.add(Dense(512, input_dim=input_length, init='RandomNormal'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.5))

    model.add(Dense(1024, input_dim=input_length, init='RandomNormal'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.5))

    model.add(Dense(2048, input_dim=input_length, init='RandomNormal'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.5))

    """
    model.add(Dense(2048, input_dim=input_length, init='RandomNormal'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.5))
    """

    model.add(Dense(1024, input_dim=input_length, init='RandomNormal'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.5))

    model.add(Dense(512, input_dim=input_length, init='RandomNormal'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.5))

    model.add(Dense(256, input_dim=input_length, init='RandomNormal'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.5))

    model.add(Dense(128, input_dim=input_length, init='RandomNormal'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.5))

    model.add(Dense(64, input_dim=input_length, init='RandomNormal'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.5))

    model.add(Dense(3, activation="linear"))
    """
    input_layer = Input(shape=(input_length,))
    encoded = BatchNormalization()(encoded)
    encoded = PReLU()(encoded)
    encoded = Dropout(0.5, noise_shape=None, seed=None)(encoded)

    encoded = Dense(64)(input_layer)
    encoded = BatchNormalization()(encoded)
    encoded = PReLU()(encoded)
    encoded = Dropout(0.5, noise_shape=None, seed=None)(encoded)

    encoded = Dense(128 )(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = PReLU()(encoded)
    encoded = Dropout(0.5, noise_shape=None, seed=None)(encoded)

    encoded = Dense(256 )(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = PReLU()(encoded)
    encoded = Dropout(0.5, noise_shape=None, seed=None)(encoded)

    encoded = Dense(512 )(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = PReLU()(encoded)
    encoded = Dropout(0.5, noise_shape=None, seed=None)(encoded)

    encoded = Dense(1024 )(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = PReLU()(encoded)
    encoded = Dropout(0.5, noise_shape=None, seed=None)(encoded)

    encoded = Dense(2048)(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = PReLU()(encoded)
    encoded = Dropout(0.5, noise_shape=None, seed=None)(encoded)

    decoded = Dense(1024 )(encoded)
    decoded = BatchNormalization()(decoded)
    decoded = PReLU()(decoded)
    decoded = Dropout(0.5, noise_shape=None, seed=None)(decoded)

    decoded = Dense(512 )(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = PReLU()(decoded)
    decoded = Dropout(0.5, noise_shape=None, seed=None)(decoded)

    decoded = Dense(256 )(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = PReLU()(decoded)
    decoded = Dropout(0.5, noise_shape=None, seed=None)(decoded)

    decoded = Dense(128 )(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = PReLU()(decoded)
    decoded = Dropout(0.5, noise_shape=None, seed=None)(decoded)

    decoded = Dense(64 )(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = PReLU()(decoded)
    decoded = Dropout(0.5, noise_shape=None, seed=None)(decoded)

    decoded = Dense(3, activation="linear")(decoded)
    # encoder = Model(input_layer, encoded)
    nn_predictor = Model(input_layer, decoded)
    opt = optimizers.SGD(lr=0.01, momentum=0.5, decay=0.5, nesterov=True)
    nn_predictor.compile(optimizer="Nadam", loss="mean_squared_error")  # mean_absolute_error ?

    return nn_predictor
    """
    opt = optimizers.SGD(lr=0.001, momentum=0.9, decay=0.0001, nesterov=False)

    model.compile(optimizer="adam", loss="binary_crossentropy")

    return model


nn_predictor = nn_1(3)

with tf.device('/gpu:0'):
    try:
        nn_predictor.fit(Xtrain,Ytrain, batch_size=512, epochs=25000, validation_split=0.1,verbose=1)
    except (KeyboardInterrupt, SystemExit):
        nn_predictor.save(str(datetime.now()))
        print(r2_score(Ytest,nn_predictor.predict(Xtest)))
nn_predictor.save(str(datetime.now()))
print(r2_score(Ytest,nn_predictor.predict(Xtest)))
