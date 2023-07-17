import random

import loss
import numpy as np
import config_parameter
import math
import tensorflow as tf
from network import DL_method_NN_for_v2x_mod, DL_method_NN_for_v2x_hybrid
import matplotlib.pyplot as plt
from evaluation_simultaneously import generate_input_v2i, generate_input_v2v, \
    load_model, load_model_hybrid_combine, load_model_digitalwith_combine, \
    load_model_digitalwithout_combine, load_model_only_communication_hybrid, \
    load_model_only_communication_digitalwith, load_model_only_communication_digitalwithout
Test = 'V2I'
rf_size = [8,16,32]
config_parameter.vehicle_antenna_size = 32
config_parameter.antenna_size = 32
#weight = [1,40,1e7]
def load_model_8():
    config_parameter.rf_size = 8

    model = load_model(digital=False)
    model.load_weights(filepath='allmodel1/Keras_models_rf8_combine/new_model')
    return model
def load_model_16():
    config_parameter.rf_size = 16
    model = load_model(digital=False)
    model.load_weights(filepath='allmodel1/Keras_models_rf16_combine/new_model')
    return model
def load_model_24():
    config_parameter.rf_size = 24
    model = load_model(digital=False)
    model.load_weights(filepath='allmodel1/Keras_models_rf24_combine/new_model')
    return model
def load_antenna_32():
    config_parameter.antenna_size = 32
    config_parameter.vehicle_antenna_size = 32
    model = load_model()
    model.load_weights(filepath='allmodel1/Keras_models_antenna32_combine/new_model')
    return model
def load_antenna_64():
    config_parameter.antenna_size = 64
    config_parameter.vehicle_antenna_size = 64
    model = load_model()
    model.load_weights(filepath='allmodel1/Keras_models_antenna64_combine/new_model')
    return model
def load_antenna_128():
    config_parameter.antenna_size = 128
    config_parameter.vehicle_antenna_size = 128
    model = load_model()
    model.load_weights(filepath='allmodel1/Keras_models_antenna128_combine/new_model')
    return model


def comparison(combined,CSI, distance, beta, theta, steering_vector_this_o,
                                      mode):

    rf_size = [8, 16, 24]
    #num_vehicle = [1, 2, 3, 4]
    model1 = load_model_8()
    model2 = load_model_16()
    model3 = load_model_24()

    sum_rate_list = []
    crbd_list = []
    crba_list = []

    config_parameter.rf_size = 8
    output1 = model1(combined)
    analog1, digital1 = loss.tf_Output2PrecodingMatrix_rad(output1)
    precoder1 = loss.tf_Precoding_matrix_combine(analog1, digital1)
    sum_rate1_1, sinr = loss.tf_loss_sumrate(CSI, precoder1)
    sum_rate1_1 = tf.cast(sum_rate1_1, tf.float32)
    sum_rate_list.append(tf.math.log(tf.reduce_mean(sum_rate1_1)) / tf.math.log(2.0))
    Sigma_time_delay1 = loss.tf_sigma_delay_square(steering_vector_this_o, precoder1, beta)
    CRB_d1 = tf.reduce_sum(loss.tf_CRB_distance(Sigma_time_delay1), axis=1) / 4
    crbd_list.append(tf.math.log(tf.reduce_mean(CRB_d1)) / tf.math.log(2.0))
    CRB_angle1 = tf.reduce_sum(loss.tf_CRB_angle(beta, precoder1, theta), axis=1) / 4
    CRB_angle1 =tf.cast(CRB_angle1,tf.float32)
    crba_list.append(tf.math.log(tf.reduce_mean(CRB_angle1)) / tf.math.log(2.0))



    config_parameter.rf_size = 16
    output2 = model2(combined)
    analog2, digital2 = loss.tf_Output2PrecodingMatrix_rad(output2)
    precoder2 = loss.tf_Precoding_matrix_combine(analog2, digital2)
    sum_rate1_2, sinr = loss.tf_loss_sumrate(CSI, precoder2)
    sum_rate1_2 = tf.cast(sum_rate1_2, tf.float32)
    sum_rate_list.append(tf.math.log(tf.reduce_mean(sum_rate1_2)) / tf.math.log(2.0))
    Sigma_time_delay2 = loss.tf_sigma_delay_square(steering_vector_this_o, precoder2, beta)
    CRB_d2 = tf.reduce_sum(loss.tf_CRB_distance(Sigma_time_delay2), axis=1) / 4
    crbd_list.append(tf.math.log(tf.reduce_mean(CRB_d2)) / tf.math.log(2.0))
    CRB_angle2 = tf.reduce_sum(loss.tf_CRB_angle(beta, precoder2, theta), axis=1) / 4
    CRB_angle2 = tf.cast(CRB_angle2, tf.float32)
    crba_list.append(tf.math.log(tf.reduce_mean(CRB_angle2)) / tf.math.log(2.0))




    config_parameter.rf_size = 24
    output3 = model3(combined)
    analog3, digital3 = loss.tf_Output2PrecodingMatrix_rad(output3)
    precoder3 = loss.tf_Precoding_matrix_combine(analog3, digital3)
    sum_rate1_3, sinr = loss.tf_loss_sumrate(CSI, precoder3)
    sum_rate1_3 = tf.cast(sum_rate1_3, tf.float32)
    sum_rate_list.append(tf.math.log(tf.reduce_mean(sum_rate1_3)) / tf.math.log(2.0))
    Sigma_time_delay2 = loss.tf_sigma_delay_square(steering_vector_this_o, precoder2, beta)
    CRB_d2 = tf.reduce_sum(loss.tf_CRB_distance(Sigma_time_delay2), axis=1) / 4
    crbd_list.append(tf.math.log(tf.reduce_mean(CRB_d2)) / tf.math.log(2.0))
    CRB_angle3 = tf.reduce_sum(loss.tf_CRB_angle(beta, precoder3, theta), axis=1) / 4
    CRB_angle3 = tf.cast(CRB_angle3, tf.float32)
    crba_list.append(tf.math.log(tf.reduce_mean(CRB_angle3)) / tf.math.log(2.0))

    fig, ax1 = plt.subplots()
    x = range(len(rf_size))
    if mode == "sumrate":
        ax1.plot(x, sum_rate_list, 'b-.', label='hybrid ISAC')
        # ax2 = ax1.twinx()
        #ax1.plot(x, sum_rate2, 'm--', label='digital ISAC,with initial point')
        #ax1.plot(x, sum_rate3, 'r.-', label='digital ISAC,without initial point')
        # ax1.plot(range(10), sum_rate4, 'm:', label='hybrid only communication')
        # ax1.plot(range(10), sum_rate5, 'r-', label='digital with initial point only communication')
        # ax1.plot(range(10), sum_rate6, 'b-', label='digital without initial point only communication')
        #ax1.plot(x, sum_rate7, 'g-', label='ZF precoder')
        ax1.set_xlabel("number of rf chains")
        ax1.set_ylabel('log2(sum rate(bits/s/hz))', color='b')
        ax1.tick_params('y', colors='b')
        _ = plt.xticks(x, rf_size)
    elif mode == "crba":
        ax1.set_xlabel("number of rf chains")
        ax1.plot(x, crba_list, 'b-.', label='hybrid ISAC')
        # ax2 = ax1.twinx()
        #ax1.plot(x, crba2, 'm--', label='digital ISAC,with initial point')
        #ax1.plot(x, crba3, 'r.-', label='digital ISAC,without initial point')
        # ax1.plot(range(10), sum_rate4, 'm:', label='hybrid only communication')
        # ax1.plot(range(10), sum_rate5, 'r-', label='digital with initial point only communication')
        # ax1.plot(range(10), sum_rate6, 'b-', label='digital without initial point only communication')
        #ax1.plot(x, crba7, 'g-', label='ZF precoder')

        ax1.set_ylabel('log2(CRB angle rad\u00B2)', color='b')
        ax1.tick_params('y', colors='b')
        _ = plt.xticks(x, rf_size)
    elif mode == "crbd":
        ax1.set_xlabel("number of rf chains")
        ax1.plot(x, crbd_list, 'b-.', label='hybrid ISAC')
        # ax2 = ax1.twinx()
        #ax1.plot(x, crbd2, 'm--', label='digital ISAC,with initial point')
        #ax1.plot(x, crbd3, 'r.-', label='digital ISAC,without initial point')
        # ax1.plot(range(10), sum_rate4, 'm:', label='hybrid only communication')
        # ax1.plot(range(10), sum_rate5, 'r-', label='digital with initial point only communication')
        # ax1.plot(range(10), sum_rate6, 'b-', label='digital without initial point only communication')
        #ax1.plot(x, crbd_list, 'g-', label='ZF precoder')
        ax1.set_ylabel('log2(CRB distance m\u00B2)', color='b')
        ax1.tick_params('y', colors='b')
        _ = plt.xticks(x, rf_size)
    fig.tight_layout()

    # 添加图例
    #lines = [ax1.get_lines()[0], \
     #        ax1.get_lines()[1], \
      #       ax1.get_lines()[2], \
       #      ax1.get_lines()[3]]

    lines = [ax1.get_lines()[0]]
    labels = [line.get_label() for line in lines]
    labels =[]
    ax1.legend(lines,labels, bbox_to_anchor=(0.5, 0.6))
    plt.show()

#random.seed(2)
if Test == "V2V":
    real_distance, real_theta = generate_input_v2v()
elif Test == "V2I":
    real_distance, real_theta = generate_input_v2i()

antenna_size = 32

config_parameter.antenna_size = 32
#combined = loss.Conversion2CSI(real_distance, real_theta)P
combined = loss.Conversion2input_small(real_theta.T[:40], real_distance.T[:40])
combined = tf.expand_dims(combined, axis=3)
CSI = tf.complex(combined[:, :, 6 * antenna_size:7 * antenna_size, 0],
                 combined[:, :, 7 * antenna_size:8 * antenna_size, 0])
steering_vector_this_o = tf.complex(combined[:, :, 0:antenna_size, 0],
                                    combined[:, :, antenna_size:2 * antenna_size, 0])

distance = combined[:, :, 3 * antenna_size:4 * antenna_size, 0]
beta = loss.Reflection_coefficient(distance)

theta = combined[:, :, 2 * antenna_size, 0]
mode = "crbd"
comparison(combined,CSI, distance, beta, theta, steering_vector_this_o, mode)