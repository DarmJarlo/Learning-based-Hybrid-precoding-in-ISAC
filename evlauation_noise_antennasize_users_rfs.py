import random

import loss
import numpy as np
import config_parameter
import math
import tensorflow as tf
from network import DL_method_NN_for_v2x_mod,DL_method_NN_for_v2x_hybrid
import matplotlib.pyplot as plt
from evaluation import generate_input_v2i,generate_input_v2v,\
    load_model,load_model_hybrid_combine,load_model_digitalwith_combine,\
    load_model_digitalwithout_combine
import evaluation_simultaneously
import evaluation
#Test = "V2V"
Test = "V2I"
if config_parameter.mode == "V2I":
    antenna_size = config_parameter.antenna_size
    num_vehicle = config_parameter.num_vehicle
elif config_parameter.mode == "V2V":
    antenna_size = config_parameter.vehicle_antenna_size
    num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar + config_parameter.num_horizoncar

def comparison(combined,mode):
   #CSI = tf.complex(combined[:,:,0:antenna_size],combined[:,:,antenna_size:2*antenna_size])
    CSI = tf.complex(combined[:, :, 6 * antenna_size:7 * antenna_size, 0],
                     combined[:, :, 7 * antenna_size:8 * antenna_size, 0])

    #CSI = CSI[1:]

    zf_matrix = tf.complex(combined[:, :, 4 * antenna_size:5 * antenna_size, 0],
                       combined[:, :, 5 * antenna_size:6 * antenna_size, 0])
    #zf_matrix = zf_matrix[:-1]

    distance = combined[:, :, 3 * antenna_size:4 * antenna_size, 0]
    #distance = distance[1:]
    beta = loss.Reflection_coefficient(distance)


    steering_vector_this_o = tf.complex(combined[:, :, 0:antenna_size, 0],
                                    combined[:, :, antenna_size:2 * antenna_size, 0])

    #steering_vector_this_o = steering_vector_this_o[1:]

    theta = combined[:, :, 2 * antenna_size, 0]
    #theta = theta[1:]
    model1 = load_model_hybrid_combine()
    model2 = load_model_digitalwith_combine()
    model3 = load_model_digitalwithout_combine()
    #model4 = load_model_only_communication_hybrid()
    #model5 = load_model_only_communication_digitalwith()
    #model6 = load_model_only_communication_digitalwithout()
    #output1 = model1(combined)[:-1]
    #output2 = model2(combined)[:-1]
    #output3 = model3(combined)[:-1]
    output1 = model1(combined)
    output2 = model2(combined)
    output3 = model3(combined)
    analog1,digital1 = loss.tf_Output2PrecodingMatrix_rad(output1)
    precoder1 = loss.tf_Precoding_matrix_combine(analog1,digital1)
    precoder2 = loss.tf_Output2digitalPrecoding(output2,zf_matrix=zf_matrix,distance=None)
    precoder3 = loss.tf_Output2digitalPrecoding(output3, zf_matrix=None,distance=None)
    #analog4,digital4 = loss.tf_Output2PrecodingMatrix_rad(output4)
    #precoder4 = loss.tf_Precoding_matrix_combine(analog4,digital4)
    #precoder5 = loss.tf_Output2digitalPrecoding(output5,zf_matrix=zf_matrix,distance=None)
    #precoder6 = loss.tf_Output2digitalPrecoding(output6,zf_matrix=None,distance=None)
    #precoder7 = loss.random_beamforming()
    if mode == "sumrate":
        different_k(CSI,distance,beta,theta,steering_vector_this_o,precoder1,precoder2,precoder3,zf_matrix,mode)
        #different_power(CSI,distance,beta,theta,steering_vector_this_o,precoder1,precoder2,precoder3,zf_matrix,mode)
    elif mode =="crbd":
        #different_k(CSI,distance,beta,theta,steering_vector_this_o,precoder1,precoder2,precoder3,zf_matrix,mode)
        #different_power(CSI,distance,beta,theta,steering_vector_this_o,precoder1,precoder2,precoder3,zf_matrix,mode)
        different_reflection_coefficient(CSI,distance,beta,theta,steering_vector_this_o,precoder1,precoder2,precoder3,zf_matrix,mode)
    elif mode=="crba":
        #different_reflection_coefficient(CSI, distance,beta,theta,steering_vector_this_o,precoder1, precoder2, precoder3, zf_matrix, mode)
        different_power(CSI, distance,beta,theta,steering_vector_this_o,precoder1, precoder2, precoder3, zf_matrix, mode)


def different_reflection_coefficient(CSI,distance,beta,theta,steering_vector_this_o,precoder1,precoder2,precoder3,zf_matrix,mode):
    refle = [0.1j+0.1,0.2j+0.2,0.5j+0.5,1j+1,2j+2,5+5j,10+10j]

    crbd1 = []
    crbd2 = []
    crbd3 = []
    crbd7 = []

    crba1 = []
    crba2 = []
    crba3 = []
    crba7 = []
    for i in range(len(refle)):
        config_parameter.fading_coefficient = refle[i]
        beta = loss.Reflection_coefficient(distance)
        Sigma_time_delay1 = loss.tf_sigma_delay_square(steering_vector_this_o, precoder1, beta)
        Sigma_time_delay2 = loss.tf_sigma_delay_square(steering_vector_this_o, precoder2, beta)
        Sigma_time_delay3 = loss.tf_sigma_delay_square(steering_vector_this_o, precoder3, beta)
        Sigma_time_delay7 = loss.tf_sigma_delay_square(steering_vector_this_o, tf.transpose(zf_matrix,perm=[0,2,1]), beta)
        CRB_d1 = tf.reduce_sum(loss.tf_CRB_distance(Sigma_time_delay1), axis=1) / 4
        CRB_d2 = tf.reduce_sum(loss.tf_CRB_distance(Sigma_time_delay2), axis=1) / 4
        CRB_d3 = tf.reduce_sum(loss.tf_CRB_distance(Sigma_time_delay3), axis=1) / 4
        CRB_d7 = tf.reduce_sum(loss.tf_CRB_distance(Sigma_time_delay7), axis=1) / 4
        crbd1.append(tf.math.log(tf.reduce_mean(CRB_d1))/tf.math.log(2.0))
        crbd2.append(tf.math.log(tf.reduce_mean(CRB_d2))/tf.math.log(2.0))
        crbd3.append(tf.math.log(tf.reduce_mean(CRB_d3))/tf.math.log(2.0))
        crbd7.append(tf.math.log(tf.reduce_mean(CRB_d7))/tf.math.log(2.0))

        CRB_angle1 = tf.reduce_sum(loss.tf_CRB_angle(beta, precoder1, theta), axis=1) / 4
        CRB_angle2 = tf.reduce_sum(loss.tf_CRB_angle(beta, precoder2, theta), axis=1) / 4
        CRB_angle3 = tf.reduce_sum(loss.tf_CRB_angle(beta, precoder3, theta), axis=1) / 4
        CRB_angle7 = tf.reduce_sum(loss.tf_CRB_angle(beta, tf.transpose(zf_matrix,perm=[0,2,1]), theta), axis=1) / 4
        CRB_angle1 = tf.cast(CRB_angle1, tf.float32)
        CRB_angle2 = tf.cast(CRB_angle2, tf.float32)
        CRB_angle3 = tf.cast(CRB_angle3, tf.float32)
        CRB_angle7 = tf.cast(CRB_angle7, tf.float32)
        crba1.append(tf.math.log(tf.reduce_mean(CRB_angle1))/tf.math.log(2.0))
        crba2.append(tf.math.log(tf.reduce_mean(CRB_angle2))/tf.math.log(2.0))
        crba3.append(tf.math.log(tf.reduce_mean(CRB_angle3))/tf.math.log(2.0))
        crba7.append(tf.math.log(tf.reduce_mean(CRB_angle7))/tf.math.log(2.0))


    fig, ax1 = plt.subplots()
    x=range(len(refle))
    if mode == "crba":
        ax1.set_xlabel('fading coefficient')
        ax1.plot(x, crba1, 'b-.', label='hybrid ISAC')
        #ax2 = ax1.twinx()
        ax1.plot(x, crba2, 'm--', label='digital ISAC,with initial point')
        ax1.plot(x, crba3, 'r.-', label='digital ISAC,without initial point')
        #ax1.plot(range(10), sum_rate4, 'm:', label='hybrid only communication')
        #ax1.plot(range(10), sum_rate5, 'r-', label='digital with initial point only communication')
        #ax1.plot(range(10), sum_rate6, 'b-', label='digital without initial point only communication')
        ax1.plot(x, crba7, 'g-', label='ZF precoder')

        ax1.set_ylabel('log2(CRB angle rad\u00B2)', color='b')
        ax1.tick_params('y', colors='b')
        _ = plt.xticks(x, refle)
    elif mode == "crbd":

        ax1.set_xlabel('fading coefficient')
        ax1.plot(x, crbd1, 'b-.', label='hybrid ISAC')
        #ax2 = ax1.twinx()
        ax1.plot(x, crbd2, 'm--', label='digital ISAC,with initial point')
        ax1.plot(x, crbd3, 'r.-', label='digital ISAC,without initial point')
        #ax1.plot(range(10), sum_rate4, 'm:', label='hybrid only communication')
        #ax1.plot(range(10), sum_rate5, 'r-', label='digital with initial point only communication')
        #ax1.plot(range(10), sum_rate6, 'b-', label='digital without initial point only communication')
        ax1.plot(x, crbd7, 'g-', label='ZF precoder')
        ax1.set_ylabel('log2(CRB distance m\u00B2)', color='b')
        ax1.tick_params('y', colors='b')
        _ = plt.xticks(x, refle)
    fig.tight_layout()

    # 添加图例
    lines = [ax1.get_lines()[0], \
            ax1.get_lines()[1], \
            ax1.get_lines()[2], \
            ax1.get_lines()[3]]

    # lines = [ax4.get_lines()[0]]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, bbox_to_anchor=(0.5, 0.6))
    plt.show()
    idx = np.arange(antenna_size)
    angle_set = np.arange(1, 181) / 180 * np.pi
    Hset = np.exp(1j * np.pi * idx.reshape(-1, 1) * np.cos(angle_set))

    #precoder5 = precoder5.numpy()
    #print(precoder5.shape)
    #r1 = np.matmul(Hset.T, precoder1[9, :, :])
    #plt.polar(angle_set, np.abs(r1))
    #   plt.show()



def different_k(CSI, distance,beta,theta,steering_vector_this_o,precoder1, precoder2, precoder3, zf_matrix, mode):
    k= [1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9]
    sum_rate1 = []
    sum_rate2 = []
    sum_rate3 = []
    sum_rate7 = []
    crbd1 = []
    crbd2 = []
    crbd3 = []
    crbd7 = []

    crba1 = []
    crba2 = []
    crba3 = []
    crba7 = []

    for i in range(len(k)):
        config_parameter.sigma_k = k[i]
        config_parameter.sigma_z = k[i]
        sum_rate1_1,sinr = loss.tf_loss_sumrate(CSI,precoder1)
        sum_rate1_1 = tf.cast(sum_rate1_1,tf.float32)
        print("sum_rate1_1",sum_rate1_1)
        sum_rate1.append(tf.math.log(tf.reduce_mean(sum_rate1_1))/tf.math.log(2.0))

        sum_rate2_1,sinr = loss.tf_loss_sumrate(CSI,precoder2)
        sum_rate2_1 = tf.cast(sum_rate2_1, tf.float32)
        sum_rate2.append(tf.math.log(tf.reduce_mean(sum_rate2_1))/tf.math.log(2.0))

        sum_rate3_1,sinr = loss.tf_loss_sumrate(CSI,precoder3)
        sum_rate3_1 = tf.cast(sum_rate3_1, tf.float32)
        sum_rate3.append(tf.math.log(tf.reduce_mean(sum_rate3_1))/tf.math.log(2.0))
        sum_rate7_1,sinr = loss.tf_loss_sumrate(CSI,tf.transpose(zf_matrix,perm=[0,2,1]))
        sum_rate7_1 = tf.cast(sum_rate7_1, tf.float32)
        sum_rate7.append(tf.math.log(tf.reduce_mean(sum_rate7_1))/tf.math.log(2.0))
        Sigma_time_delay1 = loss.tf_sigma_delay_square(steering_vector_this_o, precoder1, beta)
        Sigma_time_delay2 = loss.tf_sigma_delay_square(steering_vector_this_o, precoder2, beta)
        Sigma_time_delay3 = loss.tf_sigma_delay_square(steering_vector_this_o, precoder3, beta)
        Sigma_time_delay7 = loss.tf_sigma_delay_square(steering_vector_this_o, tf.transpose(zf_matrix,perm=[0,2,1]), beta)
        CRB_d1 = tf.reduce_sum(loss.tf_CRB_distance(Sigma_time_delay1), axis=1) / 4
        CRB_d2 = tf.reduce_sum(loss.tf_CRB_distance(Sigma_time_delay2), axis=1) / 4
        CRB_d3 = tf.reduce_sum(loss.tf_CRB_distance(Sigma_time_delay3), axis=1) / 4
        CRB_d7 = tf.reduce_sum(loss.tf_CRB_distance(Sigma_time_delay7), axis=1) / 4
        crbd1.append(tf.math.log(tf.reduce_mean(CRB_d1))/tf.math.log(2.0))
        crbd2.append(tf.math.log(tf.reduce_mean(CRB_d2))/tf.math.log(2.0))
        crbd3.append(tf.math.log(tf.reduce_mean(CRB_d3))/tf.math.log(2.0))
        crbd7.append(tf.math.log(tf.reduce_mean(CRB_d7))/tf.math.log(2.0))

        CRB_angle1 = tf.reduce_sum(loss.tf_CRB_angle(beta, precoder1, theta), axis=1) / 4
        CRB_angle2 = tf.reduce_sum(loss.tf_CRB_angle(beta, precoder2, theta), axis=1) / 4
        CRB_angle3 = tf.reduce_sum(loss.tf_CRB_angle(beta, precoder3, theta), axis=1) / 4
        CRB_angle7 = tf.reduce_sum(loss.tf_CRB_angle(beta, tf.transpose(zf_matrix,perm=[0,2,1]), theta), axis=1) / 4
        CRB_angle1 = tf.cast(CRB_angle1, tf.float32)
        CRB_angle2 = tf.cast(CRB_angle2, tf.float32)
        CRB_angle3 = tf.cast(CRB_angle3, tf.float32)
        CRB_angle7 = tf.cast(CRB_angle7, tf.float32)


        crba1.append(tf.math.log(tf.reduce_mean(CRB_angle1))/tf.math.log(2.0))
        crba2.append(tf.math.log(tf.reduce_mean(CRB_angle2))/tf.math.log(2.0))
        crba3.append(tf.math.log(tf.reduce_mean(CRB_angle3))/tf.math.log(2.0))
        crba7.append(tf.math.log(tf.reduce_mean(CRB_angle7))/tf.math.log(2.0))

    #sum_rate8,sinr = loss.tf_loss_sumrate(CSI,precoder7)
    print("sum_rate1",sum_rate1)
    fig, ax1 = plt.subplots()
    x=range(len(k))
    if mode == "sumrate":
        ax1.plot(x, sum_rate1, 'b-.', label='hybrid ISAC')
        #ax2 = ax1.twinx()
        ax1.plot(x, sum_rate2, 'm--', label='digital ISAC,with initial point')
        ax1.plot(x, sum_rate3, 'r.-', label='digital ISAC,without initial point')
        #ax1.plot(range(10), sum_rate4, 'm:', label='hybrid only communication')
        #ax1.plot(range(10), sum_rate5, 'r-', label='digital with initial point only communication')
        #ax1.plot(range(10), sum_rate6, 'b-', label='digital without initial point only communication')
        ax1.plot(x, sum_rate7, 'g-', label='ZF precoder')
        ax1.set_xlabel("Noise:\u03C3\u2096\u00B2")
        ax1.set_ylabel('log2(sum rate(bits/s/hz))', color='b')
        ax1.tick_params('y', colors='b')
        _ = plt.xticks(x, k)
    elif mode == "crba":
        ax1.set_xlabel("Noise:\u03C3\u2096\u00B2")
        ax1.plot(x, crba1, 'b-.', label='hybrid ISAC')
        #ax2 = ax1.twinx()
        ax1.plot(x, crba2, 'm--', label='digital ISAC,with initial point')
        ax1.plot(x, crba3, 'r.-', label='digital ISAC,without initial point')
        #ax1.plot(range(10), sum_rate4, 'm:', label='hybrid only communication')
        #ax1.plot(range(10), sum_rate5, 'r-', label='digital with initial point only communication')
        #ax1.plot(range(10), sum_rate6, 'b-', label='digital without initial point only communication')
        ax1.plot(x, crba7, 'g-', label='ZF precoder')

        ax1.set_ylabel('log2(CRB angle rad\u00B2)', color='b')
        ax1.tick_params('y', colors='b')
        _ = plt.xticks(x, k)
    elif mode == "crbd":
        ax1.set_xlabel("Noise:\u03C3\u2096\u00B2")
        ax1.plot(x, crbd1, 'b-.', label='hybrid ISAC')
        #ax2 = ax1.twinx()
        ax1.plot(x, crbd2, 'm--', label='digital ISAC,with initial point')
        ax1.plot(x, crbd3, 'r.-', label='digital ISAC,without initial point')
        #ax1.plot(range(10), sum_rate4, 'm:', label='hybrid only communication')
        #ax1.plot(range(10), sum_rate5, 'r-', label='digital with initial point only communication')
        #ax1.plot(range(10), sum_rate6, 'b-', label='digital without initial point only communication')
        ax1.plot(x, crbd7, 'g-', label='ZF precoder')
        ax1.set_ylabel('log2(CRB distance m\u00B2)', color='b')
        ax1.tick_params('y', colors='b')
        _ = plt.xticks(x, k)
        


#plt.xticks(k, rotation='vertical')

#plot.legend(loc='best')
    fig.tight_layout()

    # 添加图例
    lines = [ax1.get_lines()[0], \
            ax1.get_lines()[1], \
            ax1.get_lines()[2], \
            ax1.get_lines()[3]]

    # lines = [ax4.get_lines()[0]]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, bbox_to_anchor=(0.5, 0.6))
    plt.show()
    idx = np.arange(antenna_size)
    angle_set = np.arange(1, 181) / 180 * np.pi
    Hset = np.exp(1j * np.pi * idx.reshape(-1, 1) * np.cos(angle_set))

    #precoder5 = precoder5.numpy()
    #print(precoder5.shape)
    #r1 = np.matmul(Hset.T, precoder1[9, :, :])
    #plt.polar(angle_set, np.abs(r1))
    #   plt.show()
def different_power(CSI, distance,beta,theta,steering_vector_this_o,precoder1, precoder2, precoder3, zf_matrix, mode):
    sum_rate1 = []
    sum_rate2 = []
    sum_rate3 = []
    sum_rate7 = []
    crbd1 = []
    crbd2 = []
    crbd3 = []
    crbd7 = []

    crba1 = []
    crba2 = []
    crba3 = []
    crba7 = []

    p = [0.1,0.2,0.5,1,2,5,10,20,50,100]
    p1 = [20,23,27,30,33,37,40,43,47,50]
    for i in range(len(p)):
        config_parameter.power = p[i]
        p2 = math.sqrt(p[i])
        #config_parameter.sigma_z = k[i]
        sum_rate1_1, sinr = loss.tf_loss_sumrate(CSI, p2*precoder1)
        sum_rate1_1 = tf.cast(sum_rate1_1, tf.float32)
        print("sum_rate1_1", sum_rate1_1)
        sum_rate1.append(tf.math.log(tf.reduce_mean(sum_rate1_1)) / tf.math.log(2.0))

        sum_rate2_1, sinr = loss.tf_loss_sumrate(CSI, p2*precoder2)
        sum_rate2_1 = tf.cast(sum_rate2_1, tf.float32)
        sum_rate2.append(tf.math.log(tf.reduce_mean(sum_rate2_1)) / tf.math.log(2.0))

        sum_rate3_1, sinr = loss.tf_loss_sumrate(CSI, p2*precoder3)
        sum_rate3_1 = tf.cast(sum_rate3_1, tf.float32)
        sum_rate3.append(tf.math.log(tf.reduce_mean(sum_rate3_1)) / tf.math.log(2.0))
        sum_rate7_1, sinr = loss.tf_loss_sumrate(CSI, p2*tf.transpose(zf_matrix, perm=[0, 2, 1]))
        sum_rate7_1 = tf.cast(sum_rate7_1, tf.float32)
        sum_rate7.append(tf.math.log(tf.reduce_mean(sum_rate7_1)) / tf.math.log(2.0))
        Sigma_time_delay1 = loss.tf_sigma_delay_square(steering_vector_this_o, p2*precoder1, beta)
        Sigma_time_delay2 = loss.tf_sigma_delay_square(steering_vector_this_o, p2*precoder2, beta)
        Sigma_time_delay3 = loss.tf_sigma_delay_square(steering_vector_this_o, p2*precoder3, beta)
        Sigma_time_delay7 = loss.tf_sigma_delay_square(steering_vector_this_o, p2*tf.transpose(zf_matrix, perm=[0, 2, 1]),
                                                       beta)
        CRB_d1 = tf.reduce_sum(loss.tf_CRB_distance(Sigma_time_delay1), axis=1) / 4
        CRB_d2 = tf.reduce_sum(loss.tf_CRB_distance(Sigma_time_delay2), axis=1) / 4
        CRB_d3 = tf.reduce_sum(loss.tf_CRB_distance(Sigma_time_delay3), axis=1) / 4
        CRB_d7 = tf.reduce_sum(loss.tf_CRB_distance(Sigma_time_delay7), axis=1) / 4
        crbd1.append(tf.math.log(tf.reduce_mean(CRB_d1)) / tf.math.log(2.0))
        crbd2.append(tf.math.log(tf.reduce_mean(CRB_d2)) / tf.math.log(2.0))
        crbd3.append(tf.math.log(tf.reduce_mean(CRB_d3)) / tf.math.log(2.0))
        crbd7.append(tf.math.log(tf.reduce_mean(CRB_d7)) / tf.math.log(2.0))

        CRB_angle1 = tf.reduce_sum(loss.tf_CRB_angle(beta, p2*precoder1, theta), axis=1) / 4
        CRB_angle2 = tf.reduce_sum(loss.tf_CRB_angle(beta, p2*precoder2, theta), axis=1) / 4
        CRB_angle3 = tf.reduce_sum(loss.tf_CRB_angle(beta, p2*precoder3, theta), axis=1) / 4
        CRB_angle7 = tf.reduce_sum(loss.tf_CRB_angle(beta, p2*tf.transpose(zf_matrix, perm=[0, 2, 1]), theta), axis=1) / 4
        CRB_angle1 = tf.cast(CRB_angle1, tf.float32)
        CRB_angle2 = tf.cast(CRB_angle2, tf.float32)
        CRB_angle3 = tf.cast(CRB_angle3, tf.float32)
        CRB_angle7 = tf.cast(CRB_angle7, tf.float32)
        crba1.append(tf.math.log(tf.reduce_mean(CRB_angle1)) / tf.math.log(2.0))
        crba2.append(tf.math.log(tf.reduce_mean(CRB_angle2)) / tf.math.log(2.0))
        crba3.append(tf.math.log(tf.reduce_mean(CRB_angle3)) / tf.math.log(2.0))
        crba7.append(tf.math.log(tf.reduce_mean(CRB_angle7)) / tf.math.log(2.0))

    # sum_rate8,sinr = loss.tf_loss_sumrate(CSI,precoder7)
    print("sum_rate1", sum_rate1)
    fig, ax1 = plt.subplots()
    x = range(len(p))
    if mode == "sumrate":
        ax1.plot(x, sum_rate1, 'b-.', label='hybrid ISAC')
        # ax2 = ax1.twinx()
        ax1.plot(x, sum_rate2, 'm--', label='digital ISAC,with initial point')
        ax1.plot(x, sum_rate3, 'r.-', label='digital ISAC,without initial point')
        # ax1.plot(range(10), sum_rate4, 'm:', label='hybrid only communication')
        # ax1.plot(range(10), sum_rate5, 'r-', label='digital with initial point only communication')
        # ax1.plot(range(10), sum_rate6, 'b-', label='digital without initial point only communication')
        ax1.plot(x, sum_rate7, 'g-', label='ZF precoder')
        ax1.set_xlabel("Power/dBm")
        ax1.set_ylabel('log2(sum rate(bits/s/hz))', color='b')
        ax1.tick_params('y', colors='b')
        _ = plt.xticks(x, p1)
    elif mode == "crba":
        ax1.set_xlabel("Power/dBm")
        ax1.plot(x, crba1, 'b-.', label='hybrid ISAC')
        # ax2 = ax1.twinx()
        ax1.plot(x, crba2, 'm--', label='digital ISAC,with initial point')
        ax1.plot(x, crba3, 'r.-', label='digital ISAC,without initial point')
        # ax1.plot(range(10), sum_rate4, 'm:', label='hybrid only communication')
        # ax1.plot(range(10), sum_rate5, 'r-', label='digital with initial point only communication')
        # ax1.plot(range(10), sum_rate6, 'b-', label='digital without initial point only communication')
        ax1.plot(x, crba7, 'g-', label='ZF precoder')

        ax1.set_ylabel('log2(CRB angle rad\u00B2)', color='b')
        ax1.tick_params('y', colors='b')
        _ = plt.xticks(x, p1)
    elif mode == "crbd":
        ax1.set_xlabel("Power/dBm")
        ax1.plot(x, crbd1, 'b-.', label='hybrid ISAC')
        # ax2 = ax1.twinx()
        ax1.plot(x, crbd2, 'm--', label='digital ISAC,with initial point')
        ax1.plot(x, crbd3, 'r.-', label='digital ISAC,without initial point')
        # ax1.plot(range(10), sum_rate4, 'm:', label='hybrid only communication')
        # ax1.plot(range(10), sum_rate5, 'r-', label='digital with initial point only communication')
        # ax1.plot(range(10), sum_rate6, 'b-', label='digital without initial point only communication')
        ax1.plot(x, crbd7, 'g-', label='ZF precoder')
        ax1.set_ylabel('log2(CRB distance m\u00B2)', color='b')
        ax1.tick_params('y', colors='b')
        _ = plt.xticks(x, p1)

    # sum_rate8,sinr = loss.tf_loss_sumrate(CSI,precoder7)
    #print("sum_rate1", sum_rate1)
    #fig, ax1 = plt.subplots()
    #x = range(len(p))
    #ax1.plot(x, sum_rate1, 'b-.', label='hybrid ISAC')
    # ax2 = ax1.twinx()
    #ax1.plot(x, sum_rate2, 'm--', label='digital ISAC,with initial point')
    #ax1.plot(x, sum_rate3, 'r.-', label='digital ISAC,without initial point')
    # ax1.plot(range(10), sum_rate4, 'm:', label='hybrid only communication')
    # ax1.plot(range(10), sum_rate5, 'r-', label='digital with initial point only communication')
    # ax1.plot(range(10), sum_rate6, 'b-', label='digital without initial point only communication')
    #ax1.plot(x, sum_rate7, 'g-', label='ZF precoder')
    #ax1.set_xlabel("Power/dBm")
    #ax1.set_ylabel('log2(sum rate(bits/s/hz))', color='b')
    #ax1.tick_params('y', colors='b')
    #_ = plt.xticks(x, p1)

    # plt.xticks(k, rotation='vertical')

    # plot.legend(loc='best')
    fig.tight_layout()

    # 添加图例
    lines = [ax1.get_lines()[0], \
             ax1.get_lines()[1], \
             ax1.get_lines()[2], \
             ax1.get_lines()[3]]

    # lines = [ax4.get_lines()[0]]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, bbox_to_anchor=(0.5, 0.6))
    plt.show()
    idx = np.arange(antenna_size)
    angle_set = np.arange(1, 181) / 180 * np.pi
    Hset = np.exp(1j * np.pi * idx.reshape(-1, 1) * np.cos(angle_set))
    print(sum_rate2_1)
    #different_k()

if Test == "V2V":
    real_distance, real_theta = evaluation_simultaneously.generate_input_v2v()
elif Test == "V2I":
    real_distance, real_theta = evaluation_simultaneously.generate_input_v2i()

random.seed(5)
#combined = loss.Conversion2CSI(real_distance, real_theta)P
combined = loss.Conversion2input_small(real_theta.T[:10], real_distance.T[:10])

combined = tf.expand_dims(combined, axis=3)
comparison(combined,mode="crba")
#comparison_between_crb_distance(combined)
#comparison_between_crb_angle(combined)