from __future__ import absolute_import, division, print_function
import tensorflow as tf


import math
import sys
import loss
import config_parameter

sys.path.append("..")

from network import DL_method_NN
from config_parameter import iters
sys.path.append("..")
import numpy as np
#from loss import Estimate_delay_and_doppler
Nt=8
Nr=8




def load_model():


    model = DL_method_NN()
    model.build(input_shape=(None, int(config_parameter.train_data_period/config_parameter.Radar_measure_slot), \
                             config_parameter.num_vehicle,2))
    model.summary()
    if config_parameter.FurtherTrain ==True:
        model = tf.saved_model.load('Keras_models/new_model')
    return model


if __name__ == '__main__':
    model = load_model()
    Antenna_Gain = math.sqrt(config_parameter.antenna_size * config_parameter.receiver_antenna_size)
    c = 3e8
    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    optimizer_1 = tf.keras.optimizers.Adagrad(
        learning_rate=0.001,
        initial_accumulator_value=0.1,
        epsilon=1e-07,
        name="Adagrad"
    )


    @tf.function
    def train_step(input, real_distance, step):
        with tf.GradientTape() as tape:
            theta_list = model(input)  # dont forget here we are inputing a whole batch
            print('oooooooooooo', step, output)

            Analog_matrix, Digital_matrix, predict_theta_list = loss.Output2PrecodingMatrix(Output=output)
            precoding_matrix = loss.Precoding_matrix_combine(Analog_matrix, Digital_matrix)
            # print(predictions)
            estimated_theta_list = input[0, step, :, 1]
            print("theta_list_shape", estimated_theta_list.shape)
            # steering_vector = [loss.calculate_steer_vector(predict_theta_list[v] for v in range(config_parameter.num_vehicle)
            #communication_loss = loss.loss_Sumrate(real_distance, precoding_matrix, predict_theta_list)


            gradients=tape.gradient(communication_loss, model.trainable_variables)

            optimizer_1.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

            return gradients\



    for i in range(0, config_parameter.iters):
        with open('data.txt', 'r') as file:
            # Read the lines of the file
            lines = file.readlines()
        #every line contains the real
        # Strip newline characters and whitespace from each line
        lines = [line.strip() for line in lines]

        # Print the lines
        for line in lines:
            print(line)
        #compute the angle data