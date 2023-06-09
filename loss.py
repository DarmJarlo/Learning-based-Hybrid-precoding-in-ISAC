'''
our coordination is general. angle is
'''
from math import cos, sqrt,pi,sin
import numpy as np
import cmath
import config_parameter
from scipy.signal import correlate
from scipy.optimize import fmin_bfgs
from scipy import signal
import tensorflow as tf

Nt=8
Nr=8
Antenna_Gain = sqrt(Nt*Nr)
"following are the conventional precoding method"
def tf_matrix_mse(reference_precoding,predict_precoding):
    #convert two input to dtype tf.complex128
    reference_precoding=tf.cast(reference_precoding,dtype=tf.complex128)
    predict_precoding=tf.cast(predict_precoding,dtype=tf.complex128)

    diff_matrix = tf.subtract(reference_precoding, predict_precoding)


    abs_sum = tf.reduce_sum(tf.abs(diff_matrix)**2)
    return abs_sum

def zero_forcing(CSI):
    H_inv = np.linalg.pinv(CSI)
    return H_inv
def tf_zero_forcing(channel_matrix):
    num_tx_antennas = tf.shape(channel_matrix)[-1]
    num_rx_antennas = tf.shape(channel_matrix)[-2]
    channel_matrix=tf.transpose(channel_matrix)
    channel_conj_trans = tf.linalg.adjoint(channel_matrix)

    # Compute the pseudo-inverse of the channel matrix
    pseudo_inverse = tf.linalg.matmul(channel_conj_trans,
                                      tf.linalg.inv(tf.linalg.matmul(channel_matrix,
                                                                     channel_conj_trans)))

    # Compute the precoding matrix
    #precoding_matrix = tf.linalg.matmul(pseudo_inverse,
     #                                   tf.eye(num_rx_antennas, num_tx_antennas,dtype=tf.complex128))
    matrix = tf.transpose(pseudo_inverse)
    max_power = tf.constant(config_parameter.power,dtype=tf.float32)
    magnitude_sum = tf.reduce_sum(tf.abs(matrix))
    magnitude_sum = tf.cast(magnitude_sum,dtype=tf.float32)
    adjustment_factor = max_power / magnitude_sum
    adjustment_factor = tf.cast(adjustment_factor,dtype=tf.complex128)
    normalized_array = tf.multiply(matrix, adjustment_factor)



    return normalized_array
#def tf_WMMSE(CSI):
    # set R as identity weight
 #   R = tf.eye(Nr, dtype=tf.complex64)
    #
"following are some general utilities"

# convert the output of the NN to the precoding matrix
def Output2PrecodingMatrix_with_theta(Output):
    Analog_Matrix = np.zeros((config_parameter.antenna_size,config_parameter.rf_size))
    Digital_real_Matrix = np.zeros((config_parameter.rf_size, config_parameter.num_vehicle))
    Digital_im_Matrix = np.zeros((config_parameter.rf_size, config_parameter.num_vehicle))
    Digital_Matrix = np.zeros((config_parameter.rf_size, config_parameter.num_vehicle))

    Analog_part = cmath.exp(index for index in Output[0:config_parameter.antenna_size*config_parameter.rf_size-1])
    Analog_Matrix_org = np.reshape(Analog_part,(config_parameter.antenna_size,config_parameter.rf_size))
    for i in range(0,config_parameter.antenna_size):
        for r in range(0,config_parameter.rf_size):
            Analog_Matrix[i][r]=np.exp(1j*Analog_Matrix_org[i][r])
    adder = config_parameter.antenna_size*config_parameter.rf_size
    Digital_real = Output[adder : adder +config_parameter.rf_size*config_parameter.num_vehicle-1]
    adder2 = adder +config_parameter.rf_size*config_parameter.num_vehicle
    Digital_imginary=Output[adder2:-(config_parameter.num_vehicle)]
    Digital_real_Matrix_org = np.reshape(Digital_real,(config_parameter.rf_size,config_parameter.num_vehicle))
    Digital_im_Matrix_org = np.reshape(Digital_imginary,(config_parameter.rf_size,config_parameter.num_vehicle))
    for k in range(0,config_parameter.rf_size):
        for m in range(0,config_parameter.num_vehicle):
            #Digital_real_Matrix[k][m] = Digital_real_Matrix_org[k][m]
            #Digital_im_Matrix[k][m] = 1j * Digital_im_Matrix_org[k][m]
            Digital_Matrix[k][m] = complex(Digital_real_Matrix_org[k][m],Digital_im_Matrix_org[k][m])
    theta_list = Output[-(config_parameter.num_vehicle):]
    return Analog_Matrix,Digital_Matrix,theta_list
def Output2PrecodingMatrix(Output):
    if config_parameter.mode == "V2I":
        antenna_size = config_parameter.antenna_size
        num_vehicle = config_parameter.num_vehicle
    elif config_parameter.mode == "V2V":
        antenna_size = config_parameter.vehicle_antenna_size
        num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar + config_parameter.num_horizoncar


    Analog_Matrix = np.zeros((antenna_size,config_parameter.rf_size),dtype=complex)
    Digital_real_Matrix = np.zeros((config_parameter.rf_size, num_vehicle))
    Digital_im_Matrix = np.zeros((config_parameter.rf_size, num_vehicle))
    Digital_Matrix = np.zeros((config_parameter.rf_size, num_vehicle),dtype=complex)
    print(Output.shape)
    Analog_part = Output[:,0:antenna_size*config_parameter.rf_size]
    print(Output[0:antenna_size*config_parameter.rf_size])
    #Analog_part = cmath.exp(Analog_part)
    Analog_Matrix_org = np.reshape(Analog_part,(antenna_size,config_parameter.rf_size))
    print("analog matrix",Analog_Matrix_org)
    for i in range(0,antenna_size):
        for r in range(0,config_parameter.rf_size):
            Analog_Matrix[i][r]=np.exp(1j*Analog_Matrix_org[i][r])
    print("analog matrix", Analog_Matrix)
    adder = antenna_size*config_parameter.rf_size
    Digital_real = Output[:,adder : adder +config_parameter.rf_size*num_vehicle]
    adder2 = adder +config_parameter.rf_size*num_vehicle
    Digital_imginary=Output[:,adder2:]
    Digital_real_Matrix_org = np.reshape(Digital_real,(config_parameter.rf_size,num_vehicle))
    Digital_im_Matrix_org = np.reshape(Digital_imginary,(config_parameter.rf_size,num_vehicle))
    for k in range(0,config_parameter.rf_size):
        for m in range(0,num_vehicle):
            #Digital_real_Matrix[k][m] = Digital_real_Matrix_org[k][m]
            #Digital_im_Matrix[k][m] = 1j * Digital_im_Matrix_org[k][m]
            Digital_Matrix[k][m] = Digital_real_Matrix_org[k][m]+1j*Digital_im_Matrix_org[k][m]
    #theta_list = Output[-(config_parameter.num_vehicle):]
    print(Digital_Matrix)
    return Analog_Matrix,Digital_Matrix
def generate_random_sample():
    if config_parameter.mode == "V2I":
        antenna_size = config_parameter.antenna_size
        num_vehicle = config_parameter.num_vehicle
    elif config_parameter.mode == "V2V":
        antenna_size = config_parameter.vehicle_antenna_size
        num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar + config_parameter.num_horizoncar

    real_distance = np.zeros((num_vehicle,100))
    theta = np.zeros((num_vehicle,100))

    for i in range(num_vehicle):
        real_distance[i]= np.random.uniform(50,200,100)

        #generate 100 random number between -1 and 1
        theta[i] = np.random.uniform(-1,1,100)
        #replace the 0 in theta_list with 0.1
        theta[i] = np.where(theta[i] == 0, 0.1, theta[i])




    theta = theta * np.pi
    steering_vector = np.zeros((100,num_vehicle,antenna_size,),dtype=complex)
    pathloss = np.zeros((100,num_vehicle,antenna_size))
    beta = np.zeros((100,num_vehicle,antenna_size),dtype=complex)
    for i in range(antenna_size):
        for j in range(num_vehicle):
            for m in range(len(theta[j])):
                steering_vector[m][j][i] = np.exp(1j*np.pi*i*np.cos(theta[j][m]))
                #attention this steering_vector is the transposed steering vector
                pathloss[m][j][i] = Path_loss(real_distance[j,m])
                beta[m][j][i] = Reflection_coefficient(real_distance[j,m])
    CSI = np.multiply(pathloss,np.conjugate(steering_vector))
    print(CSI.shape)
    CSI = sqrt(antenna_size)*CSI
    zf_matrix = np.zeros((100,num_vehicle,antenna_size),dtype=complex)
    for n in range(100):

        zf_matrix[n,:,:] = zero_forcing(CSI[n,:,:]).T
        #print(zf_matrix.shape)
    #generate the input for the neural network
    input_whole = np.zeros(shape=(100, num_vehicle,
               10 * antenna_size))
    input_whole[:, :, 0:antenna_size] = np.real(steering_vector)
    input_whole[:, :, antenna_size:2 * antenna_size] = np.imag(steering_vector)
    input_whole[:,:,2*antenna_size:3*antenna_size] = np.real(CSI)
    input_whole[:,:,3*antenna_size:4*antenna_size] = np.imag(CSI)
    input_whole[:,:,4*antenna_size:5*antenna_size] = np.real(zf_matrix)
    input_whole[:, :, 5 * antenna_size:6 * antenna_size] = np.imag(zf_matrix)
    input_whole[:, :, 6 * antenna_size:7 * antenna_size] = np.real(beta)
    input_whole[:,:,7*antenna_size:8*antenna_size] = np.imag(beta)
    for i in range(0, antenna_size):
        input_whole[:, :, 8 * antenna_size + i] = theta.T
        input_whole[:, :, 9 * antenna_size + i] = real_distance.T
    with open('angleanddistance', "w") as file:
        file.write("real_theta")
        file.write(str(theta) + "\n")
        file.write("real_distance")
        file.write(str(real_distance) + "\n")
    return input_whole









def tf_Output2PrecodingMatrix(Output):
    shape = tf.shape(Output)
    batch_size = shape[0]


    if config_parameter.mode == "V2I":
        antenna_size = config_parameter.antenna_size
        num_vehicle = config_parameter.num_vehicle
    elif config_parameter.mode == "V2V":
        antenna_size = config_parameter.vehicle_antenna_size
        num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar + config_parameter.num_horizoncar
    #Analog_Matrix = tf.zeros((antenna_size, config_parameter.rf_size), dtype=tf.complex128)
    #Digital_real_Matrix = tf.zeros((config_parameter.rf_size, num_vehicle))
    #Digital_im_Matrix = tf.zeros((config_parameter.rf_size, num_vehicle))
    #Digital_Matrix = tf.zeros((config_parameter.rf_size, num_vehicle), dtype=tf.complex128)
    Analog_part = Output[:, 0:antenna_size * config_parameter.rf_size]
    Analog_part_reshaped = tf.reshape(Analog_part, (batch_size,antenna_size, config_parameter.rf_size))
    Analog_Matrix_real = tf.cos(Analog_part_reshaped)
    Analog_Matrix_imaginary = tf.sin(Analog_part_reshaped)
    Analog_Matrix = tf.complex(Analog_Matrix_real, Analog_Matrix_imaginary)

    adder = antenna_size * config_parameter.rf_size
    Digital_real = Output[:, adder: adder + config_parameter.rf_size * num_vehicle]
    Digital_imginary = Output[:, adder + config_parameter.rf_size * num_vehicle:]

    Digital_real_reshaped = tf.reshape(Digital_real, (batch_size,config_parameter.rf_size, num_vehicle))
    Digital_imginary_reshaped = tf.reshape(Digital_imginary, (batch_size,config_parameter.rf_size, num_vehicle))

    Digital_Matrix = tf.complex(Digital_real_reshaped, Digital_imginary_reshaped)

    print(Digital_Matrix)
    return Analog_Matrix, Digital_Matrix
def calculate_CSI(distance,theta):
    if config_parameter.mode == "V2I":
        antenna_size = config_parameter.antenna_size
        num_vehicle = config_parameter.num_vehicle
    elif config_parameter.mode == "V2V":
        antenna_size = config_parameter.vehicle_antenna_size
        num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar + config_parameter.num_horizoncar
    print("false distance",distance)
    pathloss=Path_loss(distance)
    gain = sqrt(antenna_size)
    steering_vector = calculate_steer_vector_this(theta)
    steering_hermite = steering_vector.T.conjugate()
    CSI=gain*pathloss*steering_hermite
    return CSI
#calculate the steering vector of theta k,n as a np array
def calculate_steer_vector(theta_list):
    if config_parameter.mode == "V2I":
        antenna_size = config_parameter.antenna_size
        num_vehicle = config_parameter.num_vehicle
    elif config_parameter.mode == "V2V":
        antenna_size = config_parameter.vehicle_antenna_size
        num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar + config_parameter.num_horizoncar
    #this theta here is a estimated value
    steering_vector=np.zeros(shape=(antenna_size,num_vehicle),dtype=complex)
    for v in range(num_vehicle):

        for n1 in range(0, antenna_size):
            steering_vector[n1,v]=np.exp(-1j*pi * n1 * cos(theta_list[v]))

    return steering_vector

def calculate_steer_vector_this(theta):
    if config_parameter.mode == "V2I":
        antenna_size = config_parameter.antenna_size
        num_vehicle = config_parameter.num_vehicle
    elif config_parameter.mode == "V2V":
        antenna_size = config_parameter.vehicle_antenna_size
        num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar + config_parameter.num_horizoncar
    #this theta here is a estimated value
    steering_vector_this = np.zeros(shape=(antenna_size,1),dtype=complex)

    for n1 in range(0, antenna_size):
        steering_vector_this[n1,0]=np.exp(-1j*pi * n1 * cos(theta))

    return np.reshape(steering_vector_this,(8,))

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

"""PL = PL0 + 10 * n * log10(d/d0)

Where:

PL is the path loss in decibels (dB)
PL0 is the reference path loss at a reference distance d0
n is the path loss exponent
d is the distance between the transmitter and receiver
path_loss = reference_path_loss + 10 * path_loss_exponent * np.log10(distance / reference_distance)
"""
def Path_loss(distance):
    print(distance)
    print(distance / config_parameter.d0)
    print(config_parameter.alpha * ((distance / config_parameter.d0) ** config_parameter.path_loss_exponent))
    pathloss = sqrt(config_parameter.alpha * ((distance / config_parameter.d0) ** config_parameter.path_loss_exponent))
    print("pathloss",pathloss)
    return pathloss
def tf_Path_loss(distance):
    #distance as a list
    alpha = tf.constant(config_parameter.alpha, dtype=tf.float32)
    d0 = tf.constant(config_parameter.d0, dtype=tf.float32)
    path_loss_exponent = tf.constant(config_parameter.path_loss_exponent, dtype=tf.float32)
    distance = tf.cast(distance, dtype=tf.float32)
    path_loss = tf.sqrt(alpha * ((distance/ d0) ** path_loss_exponent))
    return path_loss
def Precoding_matrix_combine(Analog_matrix,Digital_matrix):
    #think here analog_matrix is 64x8, digital_matrix is 8x4
    return np.dot(Analog_matrix,Digital_matrix)

def tf_Precoding_matrix_combine(Analog_matrix, Digital_matrix):
    max_power = tf.constant(config_parameter.power, dtype=tf.float32)

    matrix = tf.matmul(Analog_matrix, Digital_matrix)

    magnitude_sum = tf.reduce_sum(tf.abs(matrix), axis=[1, 2], keepdims=True)
    adjustment_factor = max_power / magnitude_sum
    adjustment_factor = tf.cast(adjustment_factor, dtype=tf.complex64)

    normalized_array = tf.multiply(matrix, adjustment_factor)

    return normalized_array


'''
def This_signal(index,pathloss,transmit_steering,combined_precoding_matrix):
    if config_parameter.mode == "V2I":
        antenna_size = config_parameter.antenna_size
        num_vehicle = config_parameter.num_vehicle
    elif config_parameter.mode == "V2V":
        antenna_size = config_parameter.vehicle_antenna_size
        num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar + config_parameter.num_horizoncar
    gain = sqrt(antenna_size)
    transmit_steering_hermite = transmit_steering.T.conjugate()
    #conjugated and transposed steering vector
    channel_vector = gain*pathloss*transmit_steering_hermite # remember this transmit_steering is already a transpose one
    w_k = combined_precoding_matrix[:,index]
    this_signal = np.dot(channel_vector,w_k)
    #this should be a dot product or elementwise?
    return abs(this_signal)
#This_signal_list is the list containing all the signals ratio
# function Sum_signal computes the Sum of signal except this signal
def Sum_signal(signal_index,This_signal_list):
    if config_parameter.mode == "V2I":
        antenna_size = config_parameter.antenna_size
        num_vehicle = config_parameter.num_vehicle
    elif config_parameter.mode == "V2V":
        antenna_size = config_parameter.vehicle_antenna_size
        num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar + config_parameter.num_horizoncar
    signal_sum = 0
    for i in range(0,num_vehicle):
        if i !=signal_index:
            signal_sum += This_signal_list[i]**2
    return signal_sum
    '''
#def combine_this_signal_list(This_signal)
#def tensor_Sum_rate(CSI,precoding_matrix):
 #   return tf.py_function(tf_loss_sumrate,[CSI,precoding_matrix],tf.complex64)


def tf_loss_sumrate(CSI,precoding_matrix):
    if config_parameter.mode == "V2I":
        antenna_size = config_parameter.antenna_size
        num_vehicle = config_parameter.num_vehicle
    elif config_parameter.mode == "V2V":
        antenna_size = config_parameter.vehicle_antenna_size
        num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar + config_parameter.num_horizoncar
    #sinr = tf.zeros(shape=(1,num_vehicle))
    sumrate = 0
    CSI = tf.cast(CSI, dtype=tf.complex64)
    #values_array = tf.TensorArray(dtype=tf.complex64, size=3)
    for v in range(0,num_vehicle):
        sum_other = 0
        #CSI_v = tf.expand_dims(CSI[v], axis=0)


        #precoding_matrix_v = tf.expand_dims(precoding_matrix[:,v], axis=1)

        this_sinr =tf.square(tf.abs(tf.matmul(CSI_v,precoding_matrix_v)))
        for i in range(0,num_vehicle):
            if v!=i:
                #CSI_i = tf.expand_dims(CSI[i], axis=0)

                precoding_matrix_i= tf.expand_dims(precoding_matrix[:, i], axis=1)
                sum_other += tf.square(tf.abs(tf.matmul(CSI_v,precoding_matrix_i)))
        sum_other += config_parameter.sigma_k
        sinr = this_sinr/sum_other
        #values_array = values_array.write(v, sinr)
        sumrate += tf.math.log(1.0 + sinr) / tf.math.log(2.0)
    #values_tensor = values_array.stack()

        # 计算张量的和
    #sum_value = tf.reduce_sum(values_tensor)
    #res1 = tf.abs(sum_value/2 - values_tensor[0])
    #res2 = tf.abs(sum_value/2 - values_tensor[1])
    print("sinr",sinr)

        #sinr[:,v]=this_sinr/sum_other
        #sumrate += np.log2(1+sinr[:,v])
    return sumrate[0]

#calculate the sinr of this link
def calculate_link_sinr(this_signal,signal_sum):
    #equation 16
    numerator = this_signal**2
    denominator = signal_sum + config_parameter.sigma_k
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



def loss_Sumrate(real_distance,precoding_matrix,theta):
    if config_parameter.mode == "V2I":
        antenna_size = config_parameter.antenna_size
        num_vehicle = config_parameter.num_vehicle
    elif config_parameter.mode == "V2V":
        antenna_size = config_parameter.vehicle_antenna_size
        num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar + config_parameter.num_horizoncar
    pathloss = []
    transmit_steering = []
    this_link = []
    for index in range(0,num_vehicle):
        pathloss.append(Path_loss(real_distance[index]))
        transmit_steering.append(calculate_steer_vector_this(theta[index]))
        this_link.append(This_signal(index,pathloss[index],transmit_steering[index],precoding_matrix))
    Sum_rate = 0
    sinr_list = []
    for index in range(0,num_vehicle):
        signal_sum = Sum_signal(index,this_link)
        sinr_list.append(calculate_link_sinr(this_link[index],signal_sum))
        Sum_rate += np.log2(1+sinr_list[index])

    return Sum_rate


"following is the utilities used for calculation of CRB"
def Sigma_time_delay_square(index,distance_list,estimated_theta_list,precoding_matrix):
    if config_parameter.mode == "V2I":
        antenna_size = config_parameter.antenna_size
        num_vehicle = config_parameter.num_vehicle
    elif config_parameter.mode == "V2V":
        antenna_size = config_parameter.vehicle_antenna_size
        num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar + config_parameter.num_horizoncar


    Antenna_Gain_square = antenna_size*config_parameter.receiver_antenna_size
    numerator1 = 0
    for k in range(0,num_vehicle):
        if k !=index:
            beta_that = Reflection_coefficient(distance_list[k])
            transmit_steering_that = calculate_steer_vector_this(estimated_theta_list[k])
            transmit_steering_that_Hermite = transmit_steering_that.T.conjugate()
            print(transmit_steering_that_Hermite.shape)
            print(precoding_matrix.shape)
            numerator1 +=abs(beta_that)**2* (abs(np.dot(transmit_steering_that_Hermite,precoding_matrix[:,k])**2))

    numerator1 = (config_parameter.rou_timedelay ** 2) * (numerator1+config_parameter.sigma_z)
    beta_this= Reflection_coefficient(distance_list[index])
    transmit_steering_this = calculate_steer_vector_this(estimated_theta_list[index])
    transmit_steering_this_Hermite = transmit_steering_this.T.conjugate()
    denominator=(abs(beta_this)**2)*(abs(np.dot(transmit_steering_this_Hermite,\
        precoding_matrix[:,index]))**2)
    Sigma_time = numerator1/denominator
    print(Sigma_time)
    return Sigma_time
def Sigma_doppler_square(index,distance_list,estimated_theta_list,precoding_matrix):
    if config_parameter.mode == "V2I":
        antenna_size = config_parameter.antenna_size
        num_vehicle = config_parameter.num_vehicle
    elif config_parameter.mode == "V2V":
        antenna_size = config_parameter.vehicle_antenna_size
        num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar + config_parameter.num_horizoncar

    Antenna_Gain_square = antenna_size*config_parameter.receiver_antenna_size
    #numerator1 = (config_parameter.rou_timedelay**2)*()
    numerator1 = 0
    for k in range(0,num_vehicle):
        if k !=index:
            beta_that = Reflection_coefficient(distance_list[k])
            transmit_steering_that = calculate_steer_vector(antenna_size,estimated_theta_list[k])
            transmit_steering_that_Hermite = transmit_steering_that.T.conjugate()
            numerator1 +=abs(beta_that)**2* (abs(transmit_steering_that_Hermite)*precoding_matrix[:,k])**2
    numerator1 = (config_parameter.rou_dopplershift**2)* (numerator1+config_parameter.sigma_z)
    beta_this= Reflection_coefficient(distance_list[index])
    transmit_steering_this = calculate_steer_vector(antenna_size,estimated_theta_list[index])
    transmit_steering_this_Hermite = transmit_steering_this.T.conjugate()
    denominator=(abs(beta_this)**2)*(abs(transmit_steering_this_Hermite*\
        precoding_matrix[:,index])**2)
    Sigma_doppler = numerator1/denominator
    return Sigma_doppler
#def sigma_doppler_frequency_square():


#def Estimate_delay_and_doppler():


 #   return Estimated_delay, Estimated_doppler_frequency
'''def Chirp_signal(t):
    #this chirp has no consideration about carrier frequency
    slope = config_parameter.bandwidth / config_parameter.pulse_duration
    #In the context of a chirp signal, the slope refers to the rate of change of the frequency with respect to time. It determines how quickly the frequency of the signal increases or decreases over time.
    t = np.arange(0, config_parameter.pulse_duration, 1e-9)
    chirp = np.exp((1j * np.pi * slope * t ** 2)+ 1j * 2 * np.pi * config_parameter.Frequency_original * t )
    chirp_signal = np.cos(2 * np.pi * frequency * t)
    #t = np.linspace(0, T, N, endpoint=False)
    #f = np.linspace(-B/2, B/2, N, endpoint=False)
    #chirp = np.exp(1j * 2 * np.pi * (f0*t + 0.5*B*t**2/T))
    return chirp
'''
def Chirp_signal_new():
    slope = 0.5
    sampling_rate = 1000
    chirp = []
    t = np.linspace(0, config_parameter.length_echo, int(sampling_rate * config_parameter.length_echo))
    #chirp = signal.chirp(t, f0=config_parameter.Frequency_original, t1=config_parameter.length_echo, \
                         #f1=config_parameter.Frequency_original + slope * config_parameter.length_echo,
                         #method='linear')
    chirp_signal = np.cos(2 * np.pi * config_parameter.Frequency_original * t)
    for v in config_parameter.num_vehicle:
        chirp.append(chirp_signal)
    return chirp
def Real_doppler_shift():
    velocity_direction_norm = np.linalg.norm(target_velocity)
    target_velocity_between = np.dot(target_velocity,
                                     (target_coordinates - config_parameter.RSU_location)) * velocity_direction_norm / (
                                          real_distance * velocity_direction_norm)
    observer_velocity = (0,0)
    observer_velocity_between = np.dot(np.dot(observer_velocity,
                                     (target_coordinates - config_parameter.RSU_location)) * velocity_direction_norm / (
                                          real_distance * velocity_direction_norm))
    f_observed =  (1 + (target_velocity_between / config_parameter.c)) * config_parameter.Frequency_original
    real_doppler_shift = f_observed-config_parameter.Frequency_original
    return real_doppler_shift
def Received_Signal(real_distance,index,target_velocity,target_coordinates,precoding_matrix):
    #input target_velocity as a coordinate
    real_time_delay = 2 * real_distance / config_parameter.c
    target_direction_norm = np.linalg.norm(target_coordinates - config_parameter.RSU_location)
    velocity_direction_norm = np.linalg.norm(target_velocity)
    target_velocity_between = np.dot(target_velocity, (target_coordinates - config_parameter.RSU_location))*velocity_direction_norm / (real_distance*velocity_direction_norm)
    #the velocity on the direction between target and base station
    #here we need to reconsider if there should be a minus
    #if target goes far, target_coordinates- RSU_location is actually getting larger
    f_observed =  (1 + (target_velocity_between / config_parameter.c)) * config_parameter.Frequency_original
    real_doppler_shift = f_observed-config_parameter.Frequency_original
    antenna_gain = sqrt(config_parameter.antenna_size*config_parameter.receiver_antenna_size)
    #reflection_coeff = []
    reflection_coeff=Reflection_coefficient(real_distance)
    transmit_steering_vector = calculate_steer_vector_this(theta)
    chirp = Chirp_signal_new()
    echo_signal = np.pad(chirp[index], (int(real_time_delay * config_parameter.sampling_rate), 0), mode='constant')
    echo_signal =antenna_gain*reflection_coeff*transmit_steering_vector*echo_signal*\
                 np.exp(1j * 2 * np.pi * real_doppler_shift * t)
    noise = np.random.normal(0, np.sqrt(config_parameter.Signal_noise_power / 2), len(echo_signal)) + 1j * np.random.normal(0, np.sqrt(
        config_parameter.Signal_noise_power / 2), len(echo_signal))
    return echo_signal+noise

'''def Received_Signal_old(transmitted_signal,target_velocity_x,target_coordinates,real_distance):
    #target velocity is so far a number not a coordinate,can be transfered to a coordinates
    t = np.arange(0, config_parameter.pulse_duration, 1e-9)
    """coordination and velocity should be like
    radar_loc = np.array([50, 100])
    target_loc = np.array([0, 0])
    target_vel = np.array([0, -10])"""
    #target_range = np.linalg.norm(target_coordinates - config_parameter.RSU_location)
    real_time_delay = 2 * real_distance / config_parameter.c
    target_velocity = target_velocity_x*(target_coordinates[0]-config_parameter.RSU_location[0])/real_distance
    #here we need to reconsider if there should be a minus
    #if target goes far, target_coordinates- RSU_location is actually getting
    real_doppler_shift = -2 * target_velocity.dot(target_coordinates - \
                                                  config_parameter.RSU_location) / config_parameter.c

    echo_signal = np.pad(transmitted_signal, (int(time_delay * sampling_rate), 0), mode='constant')
    reflection_coeff = Reflection_coefficient(target_range)

    #here we need to think how to get this theta
    transmit_steering_vector = calculate_steer_vector(config_parameter.antenna_size,theta)


    received_signal = reflection_coeff*transmit_steering_vector*\
                      np.exp(-1j * 2 * np.pi * config_parameter.Frequency_original * (t - real_time_delay)) * np.exp(1j * 2 * np.pi * target_doppler_shift * f)

    #received_signal = np.sqrt(received_power) * reference_signal * np.exp(
     #   1j * 2 * np.pi * target_doppler_shift * t) * np.heaviside(t - target_delay, 1) + np.sqrt(noise_power) * (
      #                            np.random.randn(len(reference_signal)) + 1j * np.random.randn(len(reference_signal)))



    return received_signal
    '''
def Matched_filter(reference_signal,received_signal,estimated_distance_last):
    correlation = correlate(reference_signal,received_signal)
    peak_index = np.argmax(np.abs(correlation))
    latency = peak_index / config_parameter.sampling_rate
    estimated_distance = 0.5*latency*config_parameter.c
    #estimated_velocity = peak_index * 3e8 / (2 * target_range * \
                                              #     num_pulses * chirp_bandwidth / signal_bandwidth)
    #estimated_velocity = (sqrt(estimated_distance**2 - config_parameter.RSU_location[1]**2) - last_location_x)/config_parameter.Radar_measure_slot
    #movement_norm = np.linalg.norm()
    estimated_velocity_between_norm = (estimated_distance-estimated_distance_last)/config_parameter.Radar_measure_slot

    #attention this velocity is about velocity on x-axis

    doppler_frequency_shift = 2 * estimated_velocity_between_norm * config_parameter.Frequency_original / 3e8
    return latency,estimated_distance,estimated_velocity_between_norm,doppler_frequency_shift




def Reflection_coefficient(distance_this_vehicle):
    beta = config_parameter.fading_coefficient/(2*distance_this_vehicle)
    return beta
def Echo_partial_Theta(beta,combined_precoding_matrix,vehicle_index,matched_filter_gain,theta_this):
    if config_parameter.mode == "V2I":
        antenna_size = config_parameter.antenna_size
        num_vehicle = config_parameter.num_vehicle
    elif config_parameter.mode == "V2V":
        antenna_size = config_parameter.vehicle_antenna_size
        num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar + config_parameter.num_horizoncar

    sin_theta =sin(theta_this)
    if sin_theta == 0:
        sin_theta = 0.01
    partial = 0
    for nt in range(2,antenna_size):
        partial += -sqrt(antenna_size)*beta*matched_filter_gain*\
                  combined_precoding_matrix[nt][vehicle_index]\
                   *(cmath.exp(1j*(nt-1)*cos(theta_this)))*1j*pi*(nt-1)*sin_theta

    return partial
#def matched_filter_gain(estimated_dopplershift,real_dopplershift,estimated_timedelay,real_timedelay):
 #   def integrand(estimated_dopplershift,real_dopplershift,estimated_timedelay,real_timedelay):
  #      signal1 =  np.pad(Chirp_signal_new(), (int(estimated_timedelay * sampling_rate), 0), mode='constant')
   #     signal2 =  Chirp_signal(t-estimated_timedelay)


def Matched_filtering_gain():
    '''
        power_signal = np.sum(np.abs(filtered_signal) ** 2)
    power_noise = np.sum(np.abs(original_signal - filtered_signal) ** 2)

    gain = power_signal / power_noise
    gain_dB = 10 * np.log10(gain)

    '''
    #because the power of signal, the power of attenna and the power of noise is constant.
    return 10



def CRB_distance(Sigma_time_delay_2):
    #Sigma_timedelay_2 = sigma_time_delay_square(index,distance_list,estimated_theta_list,precoding_matrix)
    c = config_parameter.c
    crlb_d_inv =(1/Sigma_time_delay_2)*((2/c)**2)
    #CRB_d = np.linalg.inv(crlb_d_inv)
    CRB_d = 1/ crlb_d_inv
    return abs(CRB_d)
def tf_CRB_distance(Sigma_time_delay_2):
    c = tf.constant(config_parameter.c, dtype=tf.float32)
    crlb_d_inv = tf.divide(1.0, Sigma_time_delay_2) * tf.square(tf.divide(2.0, c))
    CRB_d = tf.divide(1.0, crlb_d_inv)
    abs_CRB_d = tf.abs(CRB_d)
    return abs_CRB_d
def CRB_angle(index,distance_list,precoding_matrix,estimated_theta_list):
    beta = Reflection_coefficient(distance_list[index])
    matched_filter_gain = Matched_filtering_gain()
    partial = Echo_partial_Theta(beta,precoding_matrix,index,matched_filter_gain,estimated_theta_list[index])
    partial_hermite = partial.T.conjugate()
    sigma_rk_inv = 1/config_parameter.sigma_rk

    CRB_theta = 1/(sigma_rk_inv*partial*partial_hermite)
    return abs(CRB_theta)

def CRB_sum(CRB_list):
    if config_parameter.mode == "V2I":
        antenna_size = config_parameter.antenna_size
        num_vehicle = config_parameter.num_vehicle
    elif config_parameter.mode == "V2V":
        antenna_size = config_parameter.vehicle_antenna_size
        num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar + config_parameter.num_horizoncar

    CRB_sum = 0
    print('------',CRB_list)
    for index in range(0,num_vehicle):
        CRB_sum += CRB_list[index]
    return abs(CRB_sum)/num_vehicle
"following is the MUSIC algorithm to estimate the angle"
"following is the loss function after transforming to a multitask learning"
#def uncertainty_weighting(real_distance,precoding_matrix,estimated_theta):


def loss_combined(CRB_d_list,CRB_thet_list,sumrate_inv_list,CRB_d_this_sum,CRB_thet_this_sum,sumrate_inv_this):

    #var_sumrate = np.var(loss_Sumrate(real_distance,precoding_matrix,estimated_theta))
    #CRB_d_list = []
    #CRB_thet_list = []
    #Sumrate_list


    var_CRB_distance = np.var(CRB_d_list) #CRB_d_list is all the previous timeslots' CRB_d
    var_CRB_angle = np.var(CRB_thet_list)
    CRB_combined = CRB_thet_this_sum/2*var_CRB_angle + CRB_d_this_sum/2*var_CRB_distance+\
        np.log(var_CRB_distance*var_CRB_angle)
    var_CRB_combined = np.var(CRB_combined)

    var_sumrate = np.var(sumrate_inv_list)



    final_loss = sumrate_inv_this/(2*var_sumrate) + CRB_combined/(2*var_CRB_combined) + np.log(var_CRB_combined*var_sumrate)

    return final_loss

def loss_CRB_combined(CRB_d_list,CRB_thet_list,CRB_d_this_sum,CRB_thet_this_sum):
    var_CRB_distance = np.var(CRB_d_list)  # CRB_d_list is all the previous timeslots' CRB_d
    var_CRB_angle = np.var(CRB_thet_list)
    return   CRB_thet_this_sum/2*var_CRB_angle + CRB_d_this_sum/2*var_CRB_distance+\
        np.log(var_CRB_distance*var_CRB_angle)
"following is the MUSIC algorithm to estimate the angle. But sofar I assume everytime the \
estimated angle is equivalent to the real angle"
def pseudospectrum(echo):
    Rxx = np.outer(echo, echo.conj())
    eigvals, eigvecs = np.linalg.eig(Rxx)
    En = eigvecs[:, -n_sensors:]
    S = En @ En.conj().T
    D, V = np.linalg.eig(S)
    D = np.real(D)
    D = np.where(D > 0, 1/D, 0)
    P = np.abs(np.dot(V, np.conj(steering_vector(np.arange(-90, 91))))**2)
    return P / np.max(P)

'''
def MUSIC():
    steering_vector = calculate_steer_vector()




    AoA =
    print(AoA)
    return AoA

'''