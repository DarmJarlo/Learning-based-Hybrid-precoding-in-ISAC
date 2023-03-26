"Here is the test environment"
import tensorflow as tf
from network import DL_method_NN
from config import antenna_size
model = DL_method_NN()
model.build(input_shape=(None, 9000, 1))
model = tf.saved_model.load('Keras_models/new_model')
output = model(channel_state)

digital_beamforming = output[0:antenna_size]







