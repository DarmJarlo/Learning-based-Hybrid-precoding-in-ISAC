from math import cos, sqrt,pi,sin
import numpy as np
import cmath
import config_parameter
from scipy.optimize import fmin_bfgs
"This is the loss of every link quality"
Nt=8
Nr=8
Antenna_Gain = sqrt(Nt*Nr)

"following are some general utilities"
def Output2PrecodingMatrix(Output):
    Analog_Matrix = np.zeros((config_parameter.antenna_size,config_parameter.rf_size))
    Digital_real_Matrix = np.zeros((config_parameter.rf_size, config_parameter.num_vehicle))
    Digital_im_Matrix = np.zeros((config_parameter.rf_size, config_parameter.num_vehicle))
    Digital_Matrix = np.zeros((config_parameter.rf_size, config_parameter.num_vehicle))

    Analog_part = cmath.exp(index for index in Output[0:config_parameter.antenna_size*config_parameter.rf_size-1])
    Analog_Matrix_org = np.reshape(Analog_part,(config_parameter.antenna_size,config_parameter.rf_size))
    for i in range(0,config_parameter.antenna_size):
        for j in range(0,config_parameter.rf_size):
            Analog_Matrix[i][j]=np.exp(Analog_Matrix_org[i][j])
    adder = config_parameter.antenna_size*config_parameter.rf_size
    Digital_real = Output[adder : adder +config_parameter.rf_size*config_parameter.num_vehicle-1]
    adder2 = adder +config_parameter.rf_size*config_parameter.num_vehicle
    Digital_imginary=Output[adder2:]
    Digital_real_Matrix_org = np.reshape(Digital_real,(config_parameter.rf_size,config_parameter.num_vehicle))
    Digital_im_Matrix_org = np.reshape(Digital_imginary,(config_parameter.rf_size,config_parameter.num_vehicle))
    for k in range(0,config_parameter.rf_size):
        for m in range(0,config_parameter.num_vehicle):
            #Digital_real_Matrix[k][m] = Digital_real_Matrix_org[k][m]
            #Digital_im_Matrix[k][m] = 1j * Digital_im_Matrix_org[k][m]
            Digital_Matrix[k][m] = complex(Digital_real_Matrix_org[k][m],Digital_im_Matrix_org[k][m])
    return Analog_Matrix,Digital_Matrix

def calculate_steer_vector(Nt,theta):
    #this theta here is a estimated value
    steering_vector = []
    for n1 in range(0, Nt):
        steering_vector = steering_vector.append(np.exp(complex(-pi * n1 * cos(theta))))
    return np.array(steering_vector).T




def Echo2RSU(self, time_delay, doppler_frequncy, theta, time):
    Reflection_coefficient = RCS / (2 * dist[self.index])
    exp_doppler = np.exp(1j * 2 * pi * t * doppler_frequency)
    phase = np.exp(complex(0, 2 * pi * time_delay * time))
    signal_umformiert = self.signal_umform(signal, RSU.precoding_matrix, t - time_delay)
    for n1 in range(0, Nt):
        transmit_steering = transmit_steering.append(np.exp(complex(-pi * n1 * cos(theta))
                                                            for n2 in range(0, Nr):
        receive_steering = receive_steering.append(np.exp(-1j * pi * n2 * cos(theta)))
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
    transmit_steering_hermite = transmit_steering.T.conjugate()
    #steering gong e zhuan zhi
    channel_vector = gain*pathloss*transmit_steering_hermite # remember this transmit_steering is already a transpose one
    w_k = combined_precoding_matrix[index,]
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

def calculate_link_sinr(this_signal,signal_sum):
    #equation 16
    numerator = this_signal**2
    denominator = signal_sum + config_parameter.sigma_k**2
    sinr = numerator/denominator
    return sinr

  '''  for n1 in range(0, Nt):
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
    '''



def loss_Sumrate(sinr_list):
    Sum_rate = 0
    for index in config_parameter.num_vehicle:
        Sum_rate += np.log2(1+sinr_list[index])

    return Sum_rate


"following is the utilities used for calculation of CRB"
def sigma_time_delay_square():
    return Sigma_timedelay_2
#def sigma_doppler_frequency_square():

def Matched_filter(echo):
    Noise_Zkn = np.random.normal(0,config_parameter.sigma_z)
    for n1 in range(0, Nt):
        transmit_steering = transmit_steering.append(np.exp(-j * pi * n1 * cos(theta)))
    transmit_steering = transmit_steering / sqrt(Nt)
def Matched_filtering_gain():
    return mfg
def Reflection_coefficient(distance_this_vehicle):
    beta = config_parameter.fading_coefficient/2*distance_this_vehicle
    return beta
def Echo_partial_Theta(beta,combined_precoding_matrix,vehicle_index,matched_filter_gain,theta_this):
    sin_theta =sin(theta_this)
    partial = 0
    for nt in range(2,config_parameter.antenna_size):
        partial += -sqrt(config_parameter.antenna_size)*beta*matched_filter_gain*\
                   combined_precoding_matrix*combined_precoding_matrix[vehicle_index][nt]\
                   *(cmath.exp(1j*(nt-1)*cos(theta_this)))*1j*pi*(nt-1)*sin_theta
    return partial



def CRB_distance(Sigma_timedelay_2):
    c = 300000000
    crlb_d_inv =(1/Sigma_timedelay_2)*((2/c)**2)
    CRB_d = np.linalg.inv(crlb_d_inv)
    return CRB_d


def CRB_angle(partial):
    partial_hermite = partial.T.conjugate()
    sigma_rk_inv = 1/config_parameter.sigma_rk

    CRB_theta = np.linalg.inv(sigma_rk_inv*partial*partial_hermite)
    return CRB_theta




"following is the loss function after transforming to a multitask learning"
def loss_combined(sumrate,CRB_d,CRB_thet,sigma_sumrate,sigma_CRB_d,sigma_CRB_theta):
    #caution: aiming to maximize sumrate


    return final_loss


