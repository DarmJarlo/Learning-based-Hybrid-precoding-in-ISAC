'''
this is the NN for beamforming
input contains three factors maybe ( angle, distance, velocity)
think we need to add the velocity(not sure estimated or the real) as the input

the question remains is if we need to use real_angle and real_distance or the estimated one


2. need to consider the specific output should link to the specific input, to let every user has
linked element of matrix

ResNet code snippet partly refers to
MIT License

Copyright (c) 2019 calmisential

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''


import tensorflow.keras as keras
from keras.layers import Dense,LSTM,Concatenate
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
import tensorflow as tf
import numpy as np
import config_parameter
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, LSTM, Dense, BatchNormalization, Activation, Add,Reshape

class ResNet(tf.keras.Model):
    def __init__(self, layer_params=[0,0,1,0]):
        super(ResNet, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(3, 3),
                                            strides=2,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                               strides=2,
                                               padding="same")

        self.layer1 = make_bottleneck_layer(filter_num=64,
                                            blocks=layer_params[0])
        self.layer2 = make_bottleneck_layer(filter_num=128,
                                            blocks=layer_params[1],
                                            stride=2)
        self.layer3 = make_bottleneck_layer(filter_num=256,
                                            blocks=layer_params[2],
                                            stride=2)
        self.layer4 = make_bottleneck_layer(filter_num=512,
                                            blocks=layer_params[3],
                                            stride=2)

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar + config_parameter.num_horizoncar

        #parameter_size = config_parameter.rf_size * config_parameter.vehicle_antenna_size + 2 * config_parameter.rf_size * num_vehicle
        parameter_size = 2*config_parameter.vehicle_antenna_size*num_vehicle
        #parameter_size = config_parameter.rf_size * config_parameter.vehicle_antenna_size + 2 * config_parameter.rf_size * num_vehicle #e64777e4300db20e83b8d9ae1526e548fcb0d836
        self.lstm = tf.keras.layers.LSTM(units=64)
        self.fc = tf.keras.layers.Dense(units=parameter_size, activation=tf.keras.activations.softmax)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.leaky_relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        #x = self.lstm(x)
        x = self.avgpool(x)
        output = self.fc(x)

        return output

from config_parameter import rf_size, antenna_size
def make_bottleneck_layer(filter_num, blocks, stride=1):

    res_block = tf.keras.Sequential()
    #res_block.add(BottleNeck(filter_num, stride=stride))
    #res_block.add(OneD_BottleNeck(filter_num, stride=stride))
    res_block.add(BottleNeck(filter_num, stride=stride))
    for _ in range(1, blocks):
        res_block.add(BottleNeck(filter_num, stride=stride))
        #res_block.add(BottleNeck(filter_num, stride=stride))

        #res_block.add(OneD_BottleNeck(filter_num, stride=stride))
    return res_block





class DL_method_NN_for_v2x_mod(keras.Model):
    def __init__(self):
        super().__init__()
        act_func = "LeakyReLU"
        init = keras.initializers.GlorotNormal() #Xavier initializer

        num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar + config_parameter.num_horizoncar
        self.conv_layer1 = Conv2D(32, kernel_size=3, activation=act_func, kernel_initializer=init, padding="same")
        self.bn1 = keras.layers.BatchNormalization()
        self.maxpool1 = MaxPooling2D()
        self.conv_layer2 = Conv2D(64, kernel_size=3, activation=act_func, kernel_initializer=init, padding="same")
        self.dropout2 = keras.layers.Dropout(0.2)
        self.bn2 = keras.layers.BatchNormalization()
        self.maxpool2 = MaxPooling2D()
        self.conv_layer3 = Conv2D(128, kernel_size=3, activation=act_func, kernel_initializer=init, padding="same")
        self.bn3 = keras.layers.BatchNormalization()
        self.conv_layer4 = Conv2D(256,kernel_size=3,activation=act_func,kernel_initializer=init,padding="same")
        self.bn4 = keras.layers.BatchNormalization()
        self.dropout3 = keras.layers.Dropout(0.15)

        self.maxpool3 = MaxPooling2D()
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.flatten = Flatten()
        #self.dense_1 = Dense(1200, activation=act_func, kernel_initializer=init)
        #self.dense_2 = Dense(1200, activation=act_func, kernel_initializer=init)
        #self.dense_3 = Dense(600, activation=act_func, kernel_initializer=init)

        #self.dense_5 = Dense(20, activation=act_func, kernel_initializer=init)
        #self.dense_6 = Dense(20, activation=act_func, kernel_initializer=init)
        num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar +config_parameter.num_horizoncar
        #parameter_size = config_parameter.rf_size*config_parameter.vehicle_antenna_size + 2*config_parameter.rf_size*num_vehicle
        parameter_size = 2 * config_parameter.vehicle_antenna_size * num_vehicle
        #self.out = Dense(parameter_size, activation='softmax', kernel_initializer=init)
        #self.dense_4 = Dense(parameter_size, activation='softmax', kernel_initializer=init)
        #self.dense_4 = Dense(600, activation=act_func, kernel_initializer=init)
        self.fc = tf.keras.layers.Dense(units=parameter_size)
        self.act = tf.keras.layers.LeakyReLU(alpha=1)

    def call(self, input):
        out = self.conv_layer1(input)
        #out = self.maxpool1(out)
        out = self.bn1(out)
        out = self.conv_layer2(out)
        out= self.dropout2(out)
        #out = self.maxpool2(out)
        out = self.bn2(out)
        out = self.conv_layer3(out)
        out = self.bn3(out)
        out =self.conv_layer4(out)
        out =self.bn4(out)
        out = self.dropout3(out)
        out = self.avgpool(out)
        #out = self.maxpool3(out)
        #out = self.bn3(out)
        out = self.flatten(out)

        #out = self.dense_1(out)
        #out = self.dense_2(out)
        #out = self.dense_3(out)
        #out = self.dense_4(out)
        out = self.fc(out)
        x = self.act(out)
        #x = tf.where(tf.math.greater(x, 0), tf.minimum(x, 5.0), tf.maximum(x, -5.0))
        #out = tf.clip_by_value(out, 1e-10, 5-(1e-10))
        #Parameter = tf.clip_by_value(out,1e-10,1-1e-10)


        return x
class DL_method_NN_for_v2x_hybrid(keras.Model):
    def __init__(self):
        super().__init__()
        act_func = "LeakyReLU"
        init = keras.initializers.GlorotNormal() #Xavier initializer

        num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar + config_parameter.num_horizoncar
        self.conv_layer1 = Conv2D(32, kernel_size=3, activation=act_func, kernel_initializer=init, padding="same")
        self.bn1 = keras.layers.BatchNormalization()
        self.maxpool1 = MaxPooling2D()
        self.conv_layer2 = Conv2D(64, kernel_size=3, activation=act_func, kernel_initializer=init, padding="same")
        self.dropout2 = keras.layers.Dropout(0.05)
        self.bn2 = keras.layers.BatchNormalization()
        self.maxpool2 = MaxPooling2D()
        self.conv_layer3 = Conv2D(128, kernel_size=3, activation=act_func, kernel_initializer=init, padding="same")
        self.bn3 = keras.layers.BatchNormalization()
        self.conv_layer4 = Conv2D(256,kernel_size=3,activation=act_func,kernel_initializer=init,padding="same")
        self.bn4 = keras.layers.BatchNormalization()
        self.conv_layer5 = Conv2D(512,kernel_size=3,activation=act_func,kernel_initializer=init,padding="same")
        self.bn5 = keras.layers.BatchNormalization()
        self.dropout3 = keras.layers.Dropout(0.35)

        self.maxpool3 = MaxPooling2D()
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.flatten = Flatten()
        #self.dense_1 = Dense(1200, activation=act_func, kernel_initializer=init)
        #self.dense_2 = Dense(1200, activation=act_func, kernel_initializer=init)
        #self.dense_3 = Dense(600, activation=act_func, kernel_initializer=init)

        #self.dense_5 = Dense(20, activation=act_func, kernel_initializer=init)
        #self.dense_6 = Dense(20, activation=act_func, kernel_initializer=init)
        num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar +config_parameter.num_horizoncar
        parameter_size = config_parameter.rf_size*config_parameter.vehicle_antenna_size + 2*config_parameter.rf_size*num_vehicle
        #parameter_size = 2 * config_parameter.vehicle_antenna_size * num_vehicle
        #self.out = Dense(parameter_size, activation='softmax', kernel_initializer=init)
        #self.dense_4 = Dense(parameter_size, activation='softmax', kernel_initializer=init)
        #self.dense_4 = Dense(600, activation=act_func, kernel_initializer=init)
        self.fc = tf.keras.layers.Dense(units=parameter_size)
        self.act = tf.keras.layers.LeakyReLU(alpha=1)

    def call(self, input):
        out = self.conv_layer1(input)
        #out = self.maxpool1(out)
        out = self.bn1(out)
        out = self.conv_layer2(out)
        out= self.dropout2(out)
        #out = self.maxpool2(out)
        out = self.bn2(out)
        out = self.conv_layer3(out)
        out = self.bn3(out)
        out =self.conv_layer4(out)
        out =self.bn4(out)
        out =self.conv_layer5(out)
        out =self.bn5(out)
        out = self.dropout3(out)
        out = self.avgpool(out)
        #out = self.maxpool3(out)
        #out = self.bn3(out)
        out = self.flatten(out)

        #out = self.dense_1(out)
        #out = self.dense_2(out)
        #out = self.dense_3(out)
        #out = self.dense_4(out)
        out = self.fc(out)
        x = self.act(out)
        #x = tf.where(tf.math.greater(x, 0), tf.minimum(x, 5.0), tf.maximum(x, -5.0))
        #out = tf.clip_by_value(out, 1e-10, 5-(1e-10))
        #Parameter = tf.clip_by_value(out,1e-10,1-1e-10)


        return x



class DL_method_NN_with_theta(keras.Model):
    def __init__(self):
        super().__init__()
        act_func = "relu"
        init = keras.initializers.GlorotNormal() #Xavier initializer
        self.conv_layer1 = Conv2D(32, kernel_size=3, activation=act_func, input_shape=(1, 10, 5, 2), kernel_initializer=init, padding="same")
        self.bn1 = keras.layers.BatchNormalization()
        self.maxpool1 = MaxPooling2D()
        self.conv_layer2 = Conv2D(64, kernel_size=3, activation=act_func, kernel_initializer=init, padding="same")
        self.bn2 = keras.layers.BatchNormalization()
        self.maxpool2 = MaxPooling2D()
        self.conv_layer3 = Conv2D(64, kernel_size=3, activation=act_func, kernel_initializer=init, padding="same")
        self.bn3 = keras.layers.BatchNormalization()
        self.maxpool3 = MaxPooling2D()
        self.flatten = Flatten()
        self.dense_1 = Dense(1200, activation=act_func, kernel_initializer=init)
        self.dense_2 = Dense(1200, activation=act_func, kernel_initializer=init)
        self.dense_3 = Dense(600, activation=act_func, kernel_initializer=init)
        self.dense_4 = Dense(600, activation=act_func, kernel_initializer=init)
        #self.dense_5 = Dense(20, activation=act_func, kernel_initializer=init)
        #self.dense_6 = Dense(20, activation=act_func, kernel_initializer=init)
        parameter_size = config_parameter.rf_size*config_parameter.antenna_size + 2*config_parameter.rf_size*config_parameter.num_vehicle +\
                         config_parameter.num_vehicle# the last item is the theta prediction
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

class DL_method_NN(keras.Model):
    def __init__(self):
        super().__init__()
        act_func = "relu"
        init = keras.initializers.GlorotNormal() #Xavier initializer
        self.conv_layer1 = Conv2D(32, kernel_size=3, activation=act_func, input_shape=(1, 10, 5, 3), kernel_initializer=init, padding="same")
        self.bn1 = keras.layers.BatchNormalization()
        self.maxpool1 = MaxPooling2D()
        self.conv_layer2 = Conv2D(64, kernel_size=3, activation=act_func, kernel_initializer=init, padding="same")
        self.bn2 = keras.layers.BatchNormalization()
        self.maxpool2 = MaxPooling2D()
        self.conv_layer3 = Conv2D(64, kernel_size=3, activation=act_func, kernel_initializer=init, padding="same")
        self.bn3 = keras.layers.BatchNormalization()
        self.maxpool3 = MaxPooling2D()
        self.flatten = Flatten()
        #self.dense_1 = Dense(1200, activation=act_func, kernel_initializer=init)
        #self.dense_2 = Dense(1200, activation=act_func, kernel_initializer=init)
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
        #out = self.conv_layer3(out)
        #out = self.maxpool3(out)
        #out = self.bn3(out)
        out = self.flatten(out)

        out = self.dense_1(out)
        out = self.dense_2(out)
        out = self.dense_3(out)
        out = self.out(out)
        #Parameter = tf.clip_by_value(out,1e-10,1-1e-10)
        Parameter = tf.clip_by_value(out, 0.001, 0.999)


        return Parameter
class DL_method_NN_Digital(keras.Model):
    def __init__(self):
        super().__init__()
        act_func = "relu"
        init = keras.initializers.GlorotNormal() #Xavier initializer
        self.conv_layer1 = Conv2D(32, kernel_size=3, activation=act_func, input_shape=(1, 10, 5, 2), kernel_initializer=init, padding="same")
        self.bn1 = keras.layers.BatchNormalization()
        self.maxpool1 = MaxPooling2D()
        self.conv_layer2 = Conv2D(64, kernel_size=3, activation=act_func, kernel_initializer=init, padding="same")
        self.bn2 = keras.layers.BatchNormalization()
        self.maxpool2 = MaxPooling2D()
        self.conv_layer3 = Conv2D(64, kernel_size=3, activation=act_func, kernel_initializer=init, padding="same")
        self.bn3 = keras.layers.BatchNormalization()
        self.maxpool3 = MaxPooling2D()
        self.flatten = Flatten()
        self.dense_1 = Dense(1200, activation=act_func, kernel_initializer=init)
        self.dense_2 = Dense(1200, activation=act_func, kernel_initializer=init)
        self.dense_3 = Dense(600, activation=act_func, kernel_initializer=init)
        self.dense_4 = Dense(600, activation=act_func, kernel_initializer=init)
        #self.dense_5 = Dense(20, activation=act_func, kernel_initializer=init)
        #self.dense_6 = Dense(20, activation=act_func, kernel_initializer=init)
        parameter_size = 2*config_parameter.antenna_size*config_parameter.num_vehicle
        self.out = Dense(parameter_size, activation='softmax', kernel_initializer=init)


    def call(self, input):
        out = self.conv_layer1(input)
        out = self.maxpool1(out)
        out = self.bn1(out)
        out = self.conv_layer2(out)
        out = self.maxpool2(out)
        out = self.bn2(out)
        #out = self.conv_layer3(out)
        #out = self.maxpool3(out)
        #out = self.bn3(out)
        out = self.flatten(out)

        out = self.dense_1(out)
        out = self.dense_2(out)
        out = self.dense_3(out)
        out = self.out(out)
        #Parameter = tf.clip_by_value(out,1e-10,1-1e-10)
        Parameter = tf.clip_by_value(out, 0.001, 0.999)

        return Parameter
class DL_method_NN_naive_digital(keras.Model):
    def __init__(self):
        super().__init__()
        act_func = "relu"
        init = keras.initializers.GlorotNormal() #Xavier initializer
        self.dense_1 = Dense(1200, activation=act_func,input_shape=(1, 10, 5, 2), kernel_initializer=init)
        self.dense_2 = Dense(1200, activation=act_func, kernel_initializer=init)
        self.dense_3 = Dense(600, activation=act_func, kernel_initializer=init)
        #self.dense_4 = Dense(600, activation=act_func, kernel_initializer=init)
        #self.dense_5 = Dense(20, activation=act_func, kernel_initializer=init)
        #self.dense_6 = Dense(20, activation=act_func, kernel_initializer=init)
        parameter_size = 2*config_parameter.antenna_size*config_parameter.num_vehicle
        self.out = Dense(parameter_size, activation='softmax', kernel_initializer=init)


    def call(self, input):


        out = self.dense_1(input)
        out = self.dense_2(out)
        out = self.dense_3(out)
        out = self.out(out)
        Parameter = tf.clip_by_value(out,1e-10,1-1e-10)


        return Parameter


class DL_method_NN_naive_hybrid(keras.Model):
    def __init__(self):
        super().__init__()
        act_func = "relu"
        init = keras.initializers.GlorotNormal()  # Xavier initializer
        self.dense_1 = Dense(1200, activation=act_func, input_shape=(1, 10, config_parameter.num_vehicle, 2), kernel_initializer=init)
        self.dense_2 = Dense(1200, activation=act_func, kernel_initializer=init)
        self.dense_3 = Dense(600, activation=act_func, kernel_initializer=init)
        #self.dense_4 = Dense(600, activation=act_func, kernel_initializer=init)
        # self.dense_5 = Dense(20, activation=act_func, kernel_initializer=init)
        # self.dense_6 = Dense(20, activation=act_func, kernel_initializer=init)
        parameter_size = config_parameter.rf_size * config_parameter.antenna_size + 2 * config_parameter.rf_size * config_parameter.num_vehicle
        self.out = Dense(parameter_size, activation='softmax', kernel_initializer=init)

    def call(self, input):
        out = self.dense_1(input)
        out = self.dense_2(out)
        out = self.dense_3(out)
        out = self.out(out)
        Parameter = tf.clip_by_value(out, 1e-10, 1 - 1e-10)

        return Parameter
class BottleNeck(tf.keras.layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(filters=filter_num * 4,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization()

        self.downsample = tf.keras.Sequential()
        self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num * 4,
                                                   kernel_size=(1, 1),
                                                   strides=stride))
        self.downsample.add(tf.keras.layers.BatchNormalization())

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.leaky_relu(x)
        x = self.conv3(x)
        x = self.bn3(x, training=training)

        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output