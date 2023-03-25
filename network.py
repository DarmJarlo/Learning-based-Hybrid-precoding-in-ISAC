import tensorflow.keras as keras
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
import tensorflow as tf

class DL_method_NN(keras.Model)
    def __init__(self, parameter_size):
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
        self.dense_1 = Dense(500, activation=act_func, kernel_initializer=init)
        self.dense_2 = Dense(500, activation=act_func, kernel_initializer=init)
        self.dense_3 = Dense(100, activation=act_func, kernel_initializer=init)
        self.dense_4 = Dense(100, activation=act_func, kernel_initializer=init)
        #self.dense_5 = Dense(20, activation=act_func, kernel_initializer=init)
        #self.dense_6 = Dense(20, activation=act_func, kernel_initializer=init)

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

        policy = self.dense_1(out)
        policy = self.dense_2(policy)
        policy = self.dense_3(policy)
        policy = self.out(policy)
        Parameter = tf.clip_by_value(policy,1e-10,1-1e-10)


        return Parameter


class CombinedNN(keras.Model):
    def __init__(self, action_space):
        super().__init__()
        act_func = "relu"
        init = keras.initializers.GlorotNormal() #Xavier initializer
        self.conv_layer1 = Conv2D(32, kernel_size=3, activation=act_func, input_shape=(1, 11, 11, 22), kernel_initializer=init, padding="same")
        self.bn1 = keras.layers.BatchNormalization()
        self.maxpool1 = MaxPooling2D()
        self.conv_layer2 = Conv2D(64, kernel_size=3, activation=act_func, kernel_initializer=init, padding="same")
        self.bn2 = keras.layers.BatchNormalization()
        self.maxpool2 = MaxPooling2D()
        self.conv_layer3 = Conv2D(128, kernel_size=3, activation=act_func, kernel_initializer=init, padding="same")
        self.bn3 = keras.layers.BatchNormalization()
        self.maxpool3 = MaxPooling2D()
        self.flatten = Flatten()
        self.dense_1 = Dense(500, activation=act_func, kernel_initializer=init)
        self.dense_2 = Dense(500, activation=act_func, kernel_initializer=init)
        self.dense_3 = Dense(100, activation=act_func, kernel_initializer=init)
        self.dense_4 = Dense(100, activation=act_func, kernel_initializer=init)
        self.dense_5 = Dense(20, activation=act_func, kernel_initializer=init)
        self.dense_6 = Dense(20, activation=act_func, kernel_initializer=init)

        self.out_policy = Dense(action_space, activation='softmax', kernel_initializer=init)
        self.out_value = Dense(1, kernel_initializer=init)

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

        policy = self.dense_1(out)
        policy = self.dense_3(policy)
        policy = self.dense_5(policy)
        policy = self.out_policy(policy)
        policy = tf.clip_by_value(policy,1e-10,1-1e-10)
        value = self.dense_1(out)
        value = self.dense_3(value)
        value = self.dense_5(value)
        value = self.out_value(value)

        return policy, value