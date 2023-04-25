from math import cos, sqrt,pi,sin
import numpy as np
import cmath
import config_parameter
from scipy.signal import correlate
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



'''
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

'''

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
    w_k = combined_precoding_matrix[:,index]
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



def loss_Sumrate(real_distance,precoding_matrix,estimated_theta):
    pathloss = []
    transmit_steering = []
    this_link = []
    for index in range(0,config_parameter.num_vehicle):
        pathloss.append(Path_loss(real_distance[index]))
        transmit_steering.append(calculate_steer_vector(config_parameter.antenna_size,estimated_theta[index]))
        this_link.append(This_signal(index,pathloss[index],transmit_steering[index],precoding_matrix))
    Sum_rate = 0
    sinr_list = []
    for index in range(0,config_parameter.num_vehicle):
        signal_sum = Sum_signal(index,this_link)
        sinr_list.append(calculate_link_sinr(this_link[index],signal_sum))
        Sum_rate += np.log2(1+sinr_list[index])

    return Sum_rate


"following is the utilities used for calculation of CRB"
def sigma_time_delay_square(index,distance_list,estimated_theta_list,precoding_matrix):
    Antenna_Gain_square = config_parameter.antenna_size*config_parameter.receiver_antenna_size
    numerator1 = (config_parameter.rou_timedelay**2)*()
    for k in range(0,config_parameter.num_vehicle):
        if k !=index:
            beta_that = Reflection_coefficient(distance_list[k])
            transmit_steering_that = calculate_steer_vector(config_parameter.antenna_size,estimated_theta_list[k])
            transmit_steering_that_Hermite = transmit_steering_that.T.conjugate()
            numerator1 +=abs(beta_that)**2* (abs(transmit_steering_that_Hermite)*precoding_matrix[k])**2
    beta_this= Reflection_coefficient(distance_list[index])
    transmit_steering_this = calculate_steer_vector(config_parameter.antenna_size,estimated_theta_list[index])
    transmit_steering_this_Hermite = transmit_steering_this.T.conjugate()
    denominator=(abs(beta_this)**2)*(abs(transmit_steering_this_Hermite*\
        precoding_matrix[index,:])**2)
    Sigma_time = numerator1/denominator
    return Sigma_time
#def sigma_doppler_frequency_square():


#def Estimate_delay_and_doppler():


 #   return Estimated_delay, Estimated_doppler_frequency
def Chirp_signal():
    #this chirp has no consideration about carrier frequency
    slope = config_parameter.bandwidth / config_parameter.pulse_duration
    t = np.arange(0, config_parameter.pulse_duration, 1e-9)
    chirp = np.exp((1j * np.pi * slope * t ** 2)+ 1j * 2 * np.pi * config_parameter.Frequency_original * t )

    #t = np.linspace(0, T, N, endpoint=False)
    #f = np.linspace(-B/2, B/2, N, endpoint=False)
    #chirp = np.exp(1j * 2 * np.pi * (f0*t + 0.5*B*t**2/T))
    return chirp

def Received_Signal(transmitted_signal,target_velocity,target_coordinates):
    t = np.arange(0, config_parameter.pulse_duration, 1e-9)
    """coordination and velocity should be like
    radar_loc = np.array([50, 100])
    target_loc = np.array([0, 0])
    target_vel = np.array([0, -10])"""
    target_range = np.linalg.norm(target_coordinates - config_parameter.RSU_location)
    real_time_delay = 2 * target_range / config_parameter.c

    #here we need to reconsider if there should be a minus
    real_doppler_shift = -2 * target_velocity.dot(target_coordinates - \
                                                  config_parameter.RSU_location) / config_parameter.c


    reflection_coeff = Reflection_coefficient(target_range)

    #here we need to think how to get this theta
    transmit_steering_vector = calculate_steer_vector(config_parameter.antenna_size,theta)


    received_signal = reflection_coeff*transmit_steering_vector*\
                      np.exp(-1j * 2 * np.pi * config_parameter.Frequency_original * (t - real_time_delay)) * np.exp(1j * 2 * np.pi * target_doppler_shift * f)

    #received_signal = np.sqrt(received_power) * reference_signal * np.exp(
     #   1j * 2 * np.pi * target_doppler_shift * t) * np.heaviside(t - target_delay, 1) + np.sqrt(noise_power) * (
      #                            np.random.randn(len(reference_signal)) + 1j * np.random.randn(len(reference_signal)))



    return received_signal
def Matched_filter(reference_signal,tx,last_location_y):
    correlation = correlate(reference_signal,tx)
    peak_index = np.argmax(np.abs(correlation))
    latency = peak_index / config_parameter.sampling_rate
    estimated_distance = 0.5*latency*3e8
    #estimated_velocity = peak_index * 3e8 / (2 * target_range * \
                                              #     num_pulses * chirp_bandwidth / signal_bandwidth)
    estimated_velocity = (sqrt(estimated_distance**2 - config_parameter.RSU_location[1]**2) - last_location_y)/
    doppler_frequency_shift = 2 * estimated_velocity * config_parameter.Frequency_original / 3e8
    return latency,estimated_distance,estimated_velocity,doppler_frequency_shift




def Reflection_coefficient(distance_this_vehicle):
    beta = config_parameter.fading_coefficient/(2*distance_this_vehicle)
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
    c = config_parameter.c
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
    final_loss = 1/sumrate + CRB_d + CRB_thet

    return final_loss


