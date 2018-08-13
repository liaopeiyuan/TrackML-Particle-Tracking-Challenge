"""
try to imitate inception resnet v2
"""

from functools import partial

from keras.models import Model
from keras.layers import Activation, BatchNormalization, Concatenate, Dropout, Dense, Lambda, Input
from keras import backend as K


def preprocess_input(x):
    return x


def fc_bn(x, units, activation="relu", use_bias=False, name=None):
    """
    fully connected and batch normalization layers
    """
    x = Dense(units=units, use_bias=use_bias, name=name)(x)
    if not use_bias:
        bn_name = _generate_layer_name('BatchNorm', prefix=name)
        x = BatchNormalization(scale=False, name=bn_name)(x)
    if activation is not None:
        ac_name = _generate_layer_name('Activation', prefix=name)
        x = Activation(activation, name=ac_name)(x)
    return x


def _generate_layer_name(name, branch_idx=None, prefix=None):
    if prefix is None:
        return None
    if branch_idx is None:
        return '_'.join((prefix, name))
    return '_'.join((prefix, 'Branch', str(branch_idx), name))


def _inception_resnet_block(x, scale, block_type, block_idx, activation='relu'):
    if block_idx is None:
        prefix = None
    else:
        prefix = '_'.join((block_type, str(block_idx)))
    name_fmt = partial(_generate_layer_name, prefix=prefix)

    if block_type == 'Block35':
        branch_0 = fc_bn(x, 32, name=name_fmt('Conv2d_1x1', 0))
        branch_1 = fc_bn(x, 32, name=name_fmt('Conv2d_0a_1x1', 1))
        branch_1 = fc_bn(branch_1, 32, name=name_fmt('Conv2d_0b_3x3', 1))
        branch_2 = fc_bn(x, 32, name=name_fmt('Conv2d_0a_1x1', 2))
        branch_2 = fc_bn(branch_2, 48, name=name_fmt('Conv2d_0b_3x3', 2))
        branch_2 = fc_bn(branch_2, 64, name=name_fmt('Conv2d_0c_3x3', 2))
        branches = [branch_0, branch_1, branch_2]
    elif block_type == 'Block17':
        branch_0 = fc_bn(x, 192, name=name_fmt('Conv2d_1x1', 0))
        branch_1 = fc_bn(x, 128, name=name_fmt('Conv2d_0a_1x1', 1))
        branch_1 = fc_bn(branch_1, 160, name=name_fmt('Conv2d_0b_1x7', 1))
        branch_1 = fc_bn(branch_1, 192, name=name_fmt('Conv2d_0c_7x1', 1))
        branches = [branch_0, branch_1]
    elif block_type == 'Block8':
        branch_0 = fc_bn(x, 192, name=name_fmt('Conv2d_1x1', 0))
        branch_1 = fc_bn(x, 192, name=name_fmt('Conv2d_0a_1x1', 1))
        branch_1 = fc_bn(branch_1, 224, name=name_fmt('Conv2d_0b_1x3', 1))
        branch_1 = fc_bn(branch_1, 256,name=name_fmt('Conv2d_0c_3x1', 1))
        branches = [branch_0, branch_1]
    else:
        raise ValueError('Unknown Inception-ResNet block type. '
                         'Expects "Block35", "Block17" or "Block8", '
                         'but got: ' + str(block_type))

    mixed = Concatenate(name=name_fmt('Concatenate'))(branches)
    up = fc_bn(mixed, K.int_shape(x)[-1], activation=None, use_bias=True, name=name_fmt('Conv2d_1x1'))
    x = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
               output_shape=K.int_shape(x)[1:],
               arguments={'scale': scale},
               name=name_fmt('ScaleSum'))([x, up])
    if activation is not None:
        x = Activation(activation, name=name_fmt('Activation'))(x)
    return x


def InceptionResNetV2(input_shape=None, dropout_keep_prob=0.8):
    # Determine proper input shape
    input_layer = Input(shape=input_shape)
    
    # Stem block: 35 x 35 x 192
    x = fc_bn(input_layer, 32, name='Conv2d_1a_3x3')
    x = fc_bn(x, 32, name='Conv2d_2a_3x3')
    x = fc_bn(x, 64, name='Conv2d_2b_3x3')
    x = fc_bn(x, 80, name='Conv2d_3b_1x1')
    x = fc_bn(x, 192, name='Conv2d_4a_3x3')

    # Mixed 5b (Inception-A block): 35 x 35 x 320
    name_fmt = partial(_generate_layer_name, prefix='Mixed_5b')
    branch_0 = fc_bn(x, 96, name=name_fmt('Conv2d_1x1', 0))
    branch_1 = fc_bn(x, 48, name=name_fmt('Conv2d_0a_1x1', 1))
    branch_1 = fc_bn(branch_1, 64, name=name_fmt('Conv2d_0b_5x5', 1))
    branch_2 = fc_bn(x, 64, name=name_fmt('Conv2d_0a_1x1', 2))
    branch_2 = fc_bn(branch_2, 96, name=name_fmt('Conv2d_0b_3x3', 2))
    branch_2 = fc_bn(branch_2, 96, name=name_fmt('Conv2d_0c_3x3', 2))
    
    branch_pool = fc_bn(x, 64, name=name_fmt('Conv2d_0b_1x1', 3))
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = Concatenate(name='Mixed_5b')(branches)

    # 10x Block35 (Inception-ResNet-A block): 35 x 35 x 320
    for block_idx in range(1, 11):
        x = _inception_resnet_block(x, scale=0.17, block_type='Block35', block_idx=block_idx)
        
    # Mixed 6a (Reduction-A block): 17 x 17 x 1088
    name_fmt = partial(_generate_layer_name, prefix='Mixed_6a')
    branch_0 = fc_bn(x, 384, name=name_fmt('Conv2d_1a_3x3', 0))
    branch_1 = fc_bn(x, 256, name=name_fmt('Conv2d_0a_1x1', 1))
    branch_1 = fc_bn(branch_1, 256, name=name_fmt('Conv2d_0b_3x3', 1))
    branch_1 = fc_bn(branch_1, 384, name=name_fmt('Conv2d_1a_3x3', 1))
    branches = [branch_0, branch_1, x]
    x = Concatenate(name='Mixed_6a')(branches)
    
    # 20x Block17 (Inception-ResNet-B block): 17 x 17 x 1088
    for block_idx in range(1, 21):
        x = _inception_resnet_block(x, scale=0.1, block_type='Block17', block_idx=block_idx)
        
    # Mixed 7a (Reduction-B block): 8 x 8 x 2080
    name_fmt = partial(_generate_layer_name, prefix='Mixed_7a')
    branch_0 = fc_bn(x, 256, name=name_fmt('Conv2d_0a_1x1', 0))
    branch_0 = fc_bn(branch_0, 384, name=name_fmt('Conv2d_1a_3x3', 0))
    branch_1 = fc_bn(x, 256, name=name_fmt('Conv2d_0a_1x1', 1))
    branch_1 = fc_bn(branch_1, 288, name=name_fmt('Conv2d_1a_3x3', 1))
    branch_2 = fc_bn(x, 256, name=name_fmt('Conv2d_0a_1x1', 2))
    branch_2 = fc_bn(branch_2, 288, name=name_fmt('Conv2d_0b_3x3', 2))
    branch_2 = fc_bn(branch_2, 320, name=name_fmt('Conv2d_1a_3x3', 2))
    branches = [branch_0, branch_1, branch_2, x]
    x = Concatenate(name='Mixed_7a')(branches)

    # 10x Block8 (Inception-ResNet-C block): 8 x 8 x 2080
    for block_idx in range(1, 10):
        x = _inception_resnet_block(x, scale=0.2, block_type='Block8', block_idx=block_idx)
    x = _inception_resnet_block(x, scale=1., activation=None, block_type='Block8', block_idx=10)

    # Final convolution block
    x = fc_bn(x, 1536, name='Conv2d_7b_1x1')
    return input_layer, x

