from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import tensorflow as tf

class ResidualLayer(Layer):
  def __init__(self):
    super(ResidualLayer, self).__init__()

  def call(self, x):
    z = Conv2D(256, 3, padding = "same")(x)
    z = BatchNormalization()(z)
    z = ReLU()(z)
    z = Conv2D(256, 3, padding = "same")(z)
    z = BatchNormalization()(z)
    z = Add()([x, z])
    o = ReLU()(z)
    return o
    
class ConvolutionalLayer(Layer):
  def __init__(self):
    super(ConvolutionalLayer, self).__init__()

  def call(self, x):
    z = Conv2D(256, 3)(x)
    z = BatchNormalization()(z)
    o = ReLU()(z)
    return o
    
class PolicyHead(Layer):
  def __init__(self):
    super(PolicyHead, self).__init__()

  def call(self, x):
    z = Conv2D(2, 1)(x)
    z = BatchNormalization()(z)
    z = ReLU()(z)
    z = Flatten()(z)
    p = Dense(19 * 19 + 1, activation = "softmax")(z)
    p = tf.math.log(p / (tf.ones(19 * 19 + 1) - p))
    return p
    
class ValueHead(Layer):
  def __init__(self):
    super(ValueHead, self).__init__()

  def call(self, x):
    z = Conv2D(1, 1)(x)
    z = BatchNormalization()(z)
    z = ReLU()(z)
    z = Flatten()(z)
    z = Dense(256)(z)
    z = ReLU()(z)
    v = Dense(1, activation = 'tanh')(z)
    return v
    
class AlphaZero(Model):
  def __init__(self):
    super(AlphaZero, self).__init__()

  def call(self, x):
    z = ConvolutionalLayer()(x)
    for i in range(40):
      z = ResidualLayer()(z)

    v = ValueHead()(z)
    p = PolicyHead()(z)

    return v, p
