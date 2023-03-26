import tensorflow.keras as keras
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
import tensorflow as tf

import config_parameter
from config_parameter import rf_size, antenna_size
class DL_method_NN(keras.Model)
    def __init__(self):
        super().__init__()
        act_func = "relu"
        init = keras.initializers.GlorotNormal() #Xavier initializer
        self.conv_layer1 = Conv2D(32, kernel_size=3, activation=act_func, input_shape=(10, 11, 11, 22), kernel_initializer=init, padding="same")
        self.bn1 = keras.layers.BatchNormalization()
        self.maxpool1 = MaxPooling2D()
        self.conv_layer2 = Conv2D(64, kernel_size=3, activation=act_func, kernel_initializer=init, padding="same")
        self.bn2 = keras.layers.BatchNormalization()
        self.maxpool2 = MaxPooling2D()
        self.conv_layer3 = Conv2D(128, kernel_size=3, activation=act_func, kernel_initializer=init, padding="same")
        self.bn3 = keras.layers.BatchNormalization()
        self.maxpool3 = MaxPooling2D()
        self.flatten = Flatten()
        self.dense_1 = Dense(1200, activation=act_func, kernel_initializer=init)
        self.dense_2 = Dense(1200, activation=act_func, kernel_initializer=init)
        self.dense_3 = Dense(600, activation=act_func, kernel_initializer=init)
        self.dense_4 = Dense(600, activation=act_func, kernel_initializer=init)
        #self.dense_5 = Dense(20, activation=act_func, kernel_initializer=init)
        #self.dense_6 = Dense(20, activation=act_func, kernel_initializer=init)
        parameter_size = config_parameter.rf_size*config_parameter.antenna_size + 2*config_parameter.rf_size*config_parameter.num_vehicle
        self.out = Dense(parameter_size, activation='softmax', kernel_initializer=init)


    def call(self, input):
        out = self.conv_layer1(input)
        out = self.maxpool1(out)
        out = self.bn1(out)
        out = self.conv_layer2(out)
        out = self.maxpool2(out)
        out = self.bn2(out)
        out = self.conv_layer3(out)
        out = self.maxpool3(out)
        out = self.bn3(out)
        out = self.flatten(out)

        out = self.dense_1(out)
        out = self.dense_2(out)
        out = self.dense_3(out)
        out = self.out(out)
        Parameter = tf.clip_by_value(out,1e-10,1-1e-10)


        return Parameter


