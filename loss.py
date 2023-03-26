from math import cos, sqrt,pi
import numpy as np
import config_parameter
from scipy.optimize import fmin_bfgs
"This is the loss of every link quality"
Nt=8
Nr=8
Antenna_Gain = sqrt(Nt*Nr)
def calculate_steer_vector(Nt,theta):
    #this theta here is a estimated value
    steering_vector = []
    for n1 in range(0, Nt):
        steering_vector = steering_vector.append(np.exp(complex(-pi * n1 * cos(theta))))
    return steering_vector

def Echo2RSU(self, time_delay, doppler_frequncy, theta, time):
    Reflection_coefficient = RCS / (2 * dist[self.index])
    exp_doppler = np.exp(j * 2 * pi * t * doppler_frequency)
    phase = np.exp(complex(0, 2 * pi * time_delay * time))
    signal_umformiert = self.signal_umform(singal, RSU.precoding_matrix, t - time_delay)
    for n1 in range(0, Nt):
        transmit_steering = transmit_steering.append(np.exp(complex(-pi * n1 * cos(theta))
                                                            for n2 in range(0, Nr):
        receive_steering = receive_steering.append(np.exp(-j * pi * n2 * cos(theta)))
        transmit_steering = transmit_steering / sqrt(Nt)
        receive_steering = receive_steering / sqrt(Nr)
        Attenna_Gain = sqrt(Nt * Nr)

        noise = CSCG
        noise
        # how to represent this CSCG noise
        echo = Attenna_Gain * Reflection_Coefficient * exp_doppler * transmit_steering.T.dot(signal_umformiert) + noise
    return echo



"following are the utilities for the calculation of sum rate in communication part"
def Path_loss(distance):
    pathloss = config_parameter.alpha * ((distance / config_parameter.d0) ** config_parameter.path_loss_exponent)
    return pathloss
def Precoding_matrix_combine(Analog_matrix,Digital_matrix):
    #think here analog_matrix is 64x8, digital_matrix is 8x4
    return np.dot(Analog_matrix,Digital_matrix)
def This_signal(index,pathloss,transmit_steering,combined_precoding_matrix):
    gain = sqrt(config_parameter.antenna_size)
    channel_vector = gain*pathloss*transmit_steering # remember this transmit_steering is already a transpose one
    w_k = combined_precoding_matrix[:, index]
    this_signal = np.dot(channel_vector,w_k)
    #this should be a dot product or elementwise?
    return abs(this_signal)
#This_signal_list is the list containing all the signals ratio
def Sum_signal(signal_index,This_signal_list):
    signal_sum = 0
    for i in range(0,config_parameter.num_vehicle):
        if i !=signal_index:
            signal_sum += This_signal_list[i]**2
    return signal_sum
#def combine_this_signal_list(This_signal)

def calculate_link_sinr(pathloss,this_signal,signal_sum):
    #equation 16
    numerator = this_signal**2
    denominator = signal_sum + config_parameter.sigma_k**2
    sinr = numerator/denominator
    return sinr

    for n1 in range(0, Nt):
        transmit_steering = transmit_steering.append(np.exp(-j * pi * n1 * cos(theta)))
    transmit_steering = transmit_steering / sqrt(Nt)
    for i in range(0, config_parameter.num_vehicle):
        w_k = precoding_matrix[:, index]
        w_k = precoding_matrix[:, i]
        link = abs(gain * pathloss * transmit_steering * w_k)

        link = link ** 2
        if i == index:
            self.link_own = link
        else:
            self.link_inter += link




def loss_Sumrate(sinr_list):
    Sum_rate = 0
    for index in config_parameter.num_vehicle:
        Sum_rate += np.log2(1+sinr_list[index])

    return Sum_rate


"following is the utilities used for calculation of CRB"
def sigma_time_delay_square():

def sigma_doppler_frequency_square():
def Matched_filter(echo):
    Noise_Zkn = np.random.normal(0,config_parameter.sigma_z)
    for n1 in range(0, Nt):
        transmit_steering = transmit_steering.append(np.exp(-j * pi * n1 * cos(theta)))
    transmit_steering = transmit_steering / sqrt(Nt)

def CRB_distance():
    return CRB_d


def CRB_angle():
    return CRB_theta




"following is the loss function after transforming to a multitask learning"
def loss_combined(sumrate,CRB_d,CRB_thet,sigma_sumrate,sigma_CRB_d,sigma_CRB_theta):
    #caution: aiming to maximize sumrate


    return final_loss


