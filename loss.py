from math import cos, sqrt,pi,sin
import numpy as np
import cmath
import config_parameter
from scipy.signal import correlate
from scipy.optimize import fmin_bfgs
from scipy import signal
import tensorflow as tf

"reference precoder"
def complex_matrix_to_polar(matrix):
    polar_matrix = np.zeros(matrix.shape, dtype=complex)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            # Get the complex number from the matrix
            complex_number = matrix[i, j]

            # Calculate magnitude (a) and angle (theta)
            magnitude = np.abs(complex_number)
            angle = np.angle(complex_number)  # Convert angle to degrees
            #if angle < 0:
             #   angle += 360
            # Convert to polar form
            #polar_number = magnitude * np.exp(1j * np.deg2rad(angle))

            # Store the polar number in the polar matrix
            polar_matrix[i, j] = np.exp(1j * angle)

    return polar_matrix
def random_beamforming():
    batch_size = config_parameter.batch_size
    if config_parameter.mode == "V2I":
        antenna_size = config_parameter.antenna_size
        num_vehicle = config_parameter.num_vehicle
    elif config_parameter.mode == "V2V":
        antenna_size = config_parameter.vehicle_antenna_size
        num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar + config_parameter.num_horizoncar
    real_part = tf.random.uniform(shape=(batch_size,antenna_size, num_vehicle))
    imag_part = tf.random.uniform(shape=(batch_size,antenna_size, num_vehicle))
    max_power = tf.constant(config_parameter.power, dtype=tf.float32)
    # 将实部和虚部组合成复数矩阵
    complex_matrix = tf.complex(real_part, imag_part)

    # 计算所有元素的绝对值之和
    abs_sum = tf.reduce_sum(tf.abs(complex_matrix), axis=(1, 2))
    factor = tf.cast(max_power/abs_sum, dtype=tf.complex64)
    # 根据最大的绝对值之和进行缩放
    scaled_matrix = tf.multiply(complex_matrix,factor[:, tf.newaxis, tf.newaxis])
    return scaled_matrix
def svd_csi(CSI):

    U, s, Vh = np.linalg.svd(CSI, full_matrices=False)
    num_rf = config_parameter.rf_size
    #print("VH",Vh.shape)
    analog_part = Vh[:num_rf,:]

    #print("analog_part",analog_part)
    #analog_part = complex_matrix_to_polar(analog_part)
    analog_part = np.angle(analog_part)
    #print("analog_part",analog_part)
    CSI_h = np.transpose(CSI.conj())
    #print("CSI_h",CSI_h.shape)
    digital_part = np.dot(analog_part,CSI_h)
    #digital_part = np.transpose(digital_part.conj()) #not right here
    return analog_part.T,digital_part

def svd_zf(zf_matrix):
    U, s, Vh = np.linalg.svd(zf_matrix, full_matrices=False)
    #print("zf",zf_matrix.shape)
    num_rf = config_parameter.rf_size
    #digital =
    analog_part = Vh[:num_rf, :]
    digital_part = np.diag(s[:num_rf]) @ Vh[:num_rf, :]

    # 取前8行的U矩阵，得到（8,8）的U'矩阵
    #U_prime = U[:, :8]

    # 调整奇异值矩阵的对角线元素
    #Sigma_prime = np.diag(s[:8])

    # 取前4列的Vh矩阵，得到（4,4）的Vh'矩阵
    #Vh_prime = Vh[:num_rf, :]

    # 构造两个新的矩阵
    #analog_part = U_prime @ Sigma_prime[:, :num_rf]  # 得到（8,5）的矩阵B
    #digital_part = Sigma_prime[:num_rf, :] @ Vh_prime  # 得到（5,4）的矩阵C
    #analog_precoder = complex_matrix_to_polar(analog_part)
    analog_rad = np.angle(analog_part)
    #print("analog_rad",analog_rad)
    digital_precoder = np.matmul(analog_rad.conj(),zf_matrix.T)
    #print("digital_precoder",digital_precoder.shape)
    #digital_precoder = complex_matrix_to_polar(digital_part)
    return analog_rad.T,digital_precoder
"load data"
def load_data():
    if config_parameter.mode == "V2I":
        antenna_size = config_parameter.antenna_size
        num_vehicle = config_parameter.num_vehicle
    elif config_parameter.mode == "V2V":
        antenna_size = config_parameter.vehicle_antenna_size
        num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar + config_parameter.num_horizoncar

    data = np.load("dataset.npy")
    distance = data[:, num_vehicle:]
    print("distance",distance[10,:])
    angle = data[:, :num_vehicle]
    print("angle",angle[10,:])
    return angle,distance
def Conversion2input_mod(angle,distance):
    theta = angle.T
    real_distance = distance.T
    if config_parameter.mode == "V2I":
        antenna_size = config_parameter.antenna_size
        num_vehicle = config_parameter.num_vehicle
    elif config_parameter.mode == "V2V":
        antenna_size = config_parameter.vehicle_antenna_size
        num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar + config_parameter.num_horizoncar
    num_sample = theta.shape[1]
    steering_vector = np.zeros((num_sample, num_vehicle, antenna_size,), dtype=complex)
    pathloss = np.zeros((num_sample, num_vehicle, antenna_size))
    beta = np.zeros((num_sample, num_vehicle, antenna_size), dtype=complex)

    for i in range(antenna_size):
        for j in range(num_vehicle):
            for m in range(num_sample):
                steering_vector[m][j][i] = np.exp(-1j * np.pi * i * np.cos(theta[j][m]))

                # attention this steering_vector is the transposed steering vector
                pathloss[m][j][i] = Path_loss(real_distance[j, m])
                beta[m][j][i] = Reflection_coefficient(real_distance[j, m])
    CSI_o = np.multiply(pathloss, np.conjugate(steering_vector))
    CSI = sqrt(antenna_size) * CSI_o
    #analog_rad = np.zeros((num_sample, antenna_size, config_parameter.rf_size))
    #digital_part = np.zeros((num_sample, num_vehicle, antenna_size), dtype=complex)
    zf_matrix = np.zeros((num_sample, num_vehicle, antenna_size), dtype=complex)
    for n in range(num_sample):
        zf_matrix[n, :, :] = zero_forcing(CSI[n, :, :]).T
        # print(zf_matrix.shape)

        #analog_rad[n],digital_part[n] =svd_zf(zf_matrix[n,:,:].T)
    input_whole = np.zeros(shape=(num_sample, num_vehicle,
                                  14 * antenna_size))
    input_whole[:, :, 0:antenna_size] = np.real(steering_vector)
    input_whole[:, :, antenna_size:2 * antenna_size] = np.imag(steering_vector)
    input_whole[:, :, 2 * antenna_size:3 * antenna_size] = np.real(CSI)
    input_whole[:, :, 3 * antenna_size:4 * antenna_size] = np.imag(CSI)
    input_whole[:, :, 4 * antenna_size:5 * antenna_size] = np.real(zf_matrix)
    input_whole[:, :, 5 * antenna_size:6 * antenna_size] = np.imag(zf_matrix)
    # input_whole[:,:,4*antenna_size:5*antenna_size] = np.real(normal_precoder)
    # input_whole[:, :, 5 * antenna_size:6 * antenna_size] = np.imag(normal_precoder)
    input_whole[:, :, 6 * antenna_size:7 * antenna_size] = np.real(beta)
    input_whole[:, :, 7 * antenna_size:8 * antenna_size] = np.imag(beta)
    input_whole[:, :, 10 * antenna_size:11 * antenna_size] = zf_analog_precoder[0:4]
    input_whole[:, :, 11 * antenna_size:12 * antenna_size] = zf_analog_precoder[4:8]
    input_whole[:, :, 12 * antenna_size:13 * antenna_size] = np.real(zf_digital_precoder[0:4])
    input_whole[:, :, 13 * antenna_size:14 * antenna_size] = np.imag(zf_digital_precoder[4:8])
    for i in range(0, antenna_size):
        input_whole[:, :, 8 * antenna_size + i] = theta.T
        input_whole[:, :, 9 * antenna_size + i] = real_distance.T

    with open('angleanddistance', "w") as file:
        file.write("real_theta")
        file.write(str(theta) + "\n")
        file.write("real_distance")
        file.write(str(real_distance) + "\n")
    return input_whole
def Conversion2input_small(angle,distance):
    theta = angle.T
    real_distance = distance.T
    if config_parameter.mode == "V2I":
        antenna_size = config_parameter.antenna_size
        num_vehicle = config_parameter.num_vehicle
    elif config_parameter.mode == "V2V":
        antenna_size = config_parameter.vehicle_antenna_size
        num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar + config_parameter.num_horizoncar
    num_sample = theta.shape[1]
    steering_vector = np.zeros((num_sample,num_vehicle,antenna_size,),dtype=complex)
    pathloss = np.zeros((num_sample,num_vehicle,antenna_size))
    zf_matrix = np.zeros((num_sample,num_vehicle,antenna_size),dtype=complex)



    for i in range(antenna_size):
        for j in range(num_vehicle):
            for m in range(num_sample):
                steering_vector[m][j][i] = np.exp(-1j*np.pi*i*np.cos(theta[j][m]))

                #attention this steering_vector is the transposed steering vector
                pathloss[m][j][i] = Path_loss(real_distance[j,m])

    CSI_o = np.multiply(pathloss, np.conjugate(steering_vector))
    CSI = sqrt(antenna_size) * CSI_o
    input_whole = np.zeros(shape=(num_sample, num_vehicle,
               9 * antenna_size))
    for n in range(num_sample):

        zf_matrix[n,:,:] = zero_forcing(CSI[n,:,:]).T
        analog_rad,digital_part =svd_zf(zf_matrix[n,:,:])
        #input_whole[n,0:config_parameter.rf_size,8*antenna_size:9*antenna_size] = np.transpose(analog_rad)
        #input_whole[n, 0:config_parameter.rf_size, 4 * antenna_size:4 * antenna_size + num_vehicle] = np.transpose(
         #   np.real(digital_part))
        #input_whole[n,0:config_parameter.rf_size,5*antenna_size:5*antenna_size+num_vehicle] = np.transpose(np.imag(digital_part))


    input_whole[:,:,0:1*antenna_size] = np.real(np.conjugate(steering_vector))
    input_whole[:,:,1*antenna_size:2*antenna_size] = np.imag(np.conjugate(steering_vector))
    input_whole[:,:,4*antenna_size:5*antenna_size] = np.real(zf_matrix)
    input_whole[:, :, 5 * antenna_size:6 * antenna_size] = np.imag(zf_matrix)
    input_whole[:,:,6*antenna_size:7*antenna_size] = np.real(CSI)
    input_whole[:, :, 7 * antenna_size:8 * antenna_size] = np.imag(CSI)
    print(input_whole.shape)
    for i in range(0, antenna_size):
        input_whole[:, :, 2 * antenna_size + i] = theta.T
        input_whole[:, :, 3 * antenna_size + i] = real_distance.T

    return input_whole
def Conversion2input_small2(angle,distance):
    theta = angle.T
    real_distance = distance.T
    if config_parameter.mode == "V2I":
        antenna_size = config_parameter.antenna_size
        num_vehicle = config_parameter.num_vehicle
    elif config_parameter.mode == "V2V":
        antenna_size = config_parameter.vehicle_antenna_size
        num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar + config_parameter.num_horizoncar
    num_sample = theta.shape[1]
    steering_vector = np.zeros((num_sample,num_vehicle,antenna_size,),dtype=complex)
    pathloss = np.zeros((num_sample,num_vehicle,antenna_size))
    zf_matrix = np.zeros((num_sample,num_vehicle,antenna_size),dtype=complex)



    for i in range(antenna_size):
        for j in range(num_vehicle):
            for m in range(num_sample):
                steering_vector[m][j][i] = np.exp(-1j*np.pi*i*np.cos(theta[j][m]))

                #attention this steering_vector is the transposed steering vector
                pathloss[m][j][i] = Path_loss(real_distance[j,m])

    CSI_o = np.multiply(pathloss, np.conjugate(steering_vector))
    CSI = sqrt(antenna_size) * CSI_o
    input_whole = np.zeros(shape=(num_sample, num_vehicle,
               7 * antenna_size))
    #analog_rad = np.zeros((num_sample,num_vehicle,antenna_size),dtype=complex)
    for n in range(num_sample):

        zf_matrix[n,:,:] = zero_forcing(CSI[n,:,:]).T
        analog_rad,digital_part =svd_(CSI[n,:,:])
        input_whole[n,0:config_parameter.rf_size,6*antenna_size:7*antenna_size] = np.transpose(analog_rad)
        input_whole[n, 0:config_parameter.rf_size, 3 * antenna_size:3 * antenna_size + num_vehicle] = np.transpose(
            np.real(digital_part))
        input_whole[n,0:config_parameter.rf_size,4*antenna_size:4*antenna_size+num_vehicle] = np.transpose(np.imag(digital_part))

    input_whole[:,:,0:1*antenna_size] = np.real(np.conjugate(steering_vector))
    input_whole[:,:,1*antenna_size:2*antenna_size] = np.imag(np.conjugate(steering_vector))
    #input_whole[:,0:config_parameter.rf_size,6*antenna_size:7*antenna_size] = np.transpose(analog_rad,axes=(0,2,1))
    #input_whole[:,0:config_parameter.rf_size,3*antenna_size:3*antenna_size+num_vehicle] = np.transpose(np.real(digital_part),axes=(0,2,1))
    #input_whole[:,0:config_parameter.rf_size,4*antenna_size:4*antenna_size+num_vehicle] = np.transpose(np.imag(digital_part),axes=(0,2,1))
    for i in range(0, antenna_size):
        input_whole[:, :, 4 * antenna_size + i] = theta.T
        input_whole[:, :, 5 * antenna_size + i] = (real_distance/100).T

    return input_whole
def Conversion2input_small3(angle,distance):
    theta = angle.T
    real_distance = distance.T
    if config_parameter.mode == "V2I":
        antenna_size = config_parameter.antenna_size
        num_vehicle = config_parameter.num_vehicle
    elif config_parameter.mode == "V2V":
        antenna_size = config_parameter.vehicle_antenna_size
        num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar + config_parameter.num_horizoncar
    num_sample = theta.shape[1]
    steering_vector = np.zeros((num_sample,num_vehicle,antenna_size,),dtype=complex)
    pathloss = np.zeros((num_sample,num_vehicle,antenna_size))
    zf_matrix = np.zeros((num_sample,num_vehicle,antenna_size),dtype=complex)



    for i in range(antenna_size):
        for j in range(num_vehicle):
            for m in range(num_sample):
                steering_vector[m][j][i] = np.exp(-1j*np.pi*i*np.cos(theta[j][m]))

                #attention this steering_vector is the transposed steering vector
                pathloss[m][j][i] = Path_loss(real_distance[j,m])

    CSI_o = np.multiply(pathloss, np.conjugate(steering_vector))
    CSI = sqrt(antenna_size) * CSI_o
    input_whole = np.zeros(shape=(num_sample, num_vehicle,
               9 * antenna_size))
    for n in range(num_sample):

        zf_matrix[n,:,:] = zero_forcing(CSI[n,:,:]).T
        analog_rad,digital_part =svd_zf(zf_matrix[n,:,:])
        input_whole[n,0:config_parameter.rf_size,8*antenna_size:9*antenna_size] = np.transpose(analog_rad)
        #input_whole[n, 0:config_parameter.rf_size, 9 * antenna_size:10 * antenna_size] = np.transpose(np.imag(analog_rad))
        input_whole[n, 0:config_parameter.rf_size, 4 * antenna_size:4 * antenna_size + num_vehicle] = np.transpose(
            np.real(digital_part))
        input_whole[n,0:config_parameter.rf_size,5*antenna_size:5*antenna_size+num_vehicle] = np.transpose(np.imag(digital_part))


    input_whole[:,:,0:1*antenna_size] = np.real(np.conjugate(steering_vector))
    input_whole[:,:,1*antenna_size:2*antenna_size] = np.imag(np.conjugate(steering_vector))
    #input_whole[:,:,4*antenna_size:5*antenna_size] = np.real(zf_matrix)
    #input_whole[:, :, 5 * antenna_size:6 * antenna_size] = np.imag(zf_matrix)
    input_whole[:,:,6*antenna_size:7*antenna_size] = np.real(CSI)
    input_whole[:, :, 7 * antenna_size:8 * antenna_size] = np.imag(CSI)
    print(input_whole.shape)
    for i in range(0, antenna_size):
        input_whole[:, :, 2 * antenna_size + i] = theta.T
        input_whole[:, :, 3 * antenna_size + i] = real_distance.T

    return input_whole
def Conversion2input(angle,distance):
    theta = angle.T
    real_distance = distance.T
    if config_parameter.mode == "V2I":
        antenna_size = config_parameter.antenna_size
        num_vehicle = config_parameter.num_vehicle
    elif config_parameter.mode == "V2V":
        antenna_size = config_parameter.vehicle_antenna_size
        num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar + config_parameter.num_horizoncar
    num_sample = theta.shape[1]
    normal_precoder = simple_precoder(theta,real_distance)
    normal_precoder = np.transpose(normal_precoder,(0,2,1))
    steering_vector = np.zeros((num_sample,num_vehicle,antenna_size,),dtype=complex)
    pathloss = np.zeros((num_sample,num_vehicle,antenna_size))
    beta = np.zeros((num_sample,num_vehicle,antenna_size),dtype=complex)

    for i in range(antenna_size):
        for j in range(num_vehicle):
            for m in range(num_sample):
                steering_vector[m][j][i] = np.exp(-1j*np.pi*i*np.cos(theta[j][m]))

                #attention this steering_vector is the transposed steering vector
                pathloss[m][j][i] = Path_loss(real_distance[j,m])
                beta[m][j][i] = Reflection_coefficient(real_distance[j,m])
    CSI_o = np.multiply(pathloss, np.conjugate(steering_vector))
    CSI = sqrt(antenna_size) * CSI_o
    zf_matrix = np.zeros((num_sample,num_vehicle,antenna_size),dtype=complex)
    for n in range(num_sample):

        zf_matrix[n,:,:] = zero_forcing(CSI[n,:,:]).T
        #print(zf_matrix.shape)
    #generate the input for the neural network
    input_whole = np.zeros(shape=(num_sample, num_vehicle,
               10 * antenna_size))
    input_whole[:, :, 0:antenna_size] = np.real(np.conjugate(steering_vector))
    input_whole[:, :, antenna_size:2 * antenna_size] = np.imag(np.conjugate(steering_vector))
    input_whole[:,:,2*antenna_size:3*antenna_size] = np.real(CSI)
    input_whole[:,:,3*antenna_size:4*antenna_size] = np.imag(CSI)
    #input_whole[:,:,4*antenna_size:5*antenna_size] = np.real(zf_matrix)
    #input_whole[:, :, 5 * antenna_size:6 * antenna_size] = np.imag(zf_matrix)
    input_whole[:,:,4*antenna_size:5*antenna_size] = np.real(normal_precoder)
    input_whole[:, :, 5 * antenna_size:6 * antenna_size] = np.imag(normal_precoder)
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
def Conversion2CSI_mod(angle,distance):
    theta = angle.T
    real_distance = distance.T
    if config_parameter.mode == "V2I":
        antenna_size = config_parameter.antenna_size
        num_vehicle = config_parameter.num_vehicle
    elif config_parameter.mode == "V2V":
        antenna_size = config_parameter.vehicle_antenna_size
        num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar + config_parameter.num_horizoncar
    num_sample = theta.shape[1]
    steering_vector = np.zeros((num_sample,num_vehicle,antenna_size,),dtype=complex)
    pathloss = np.zeros((num_sample,num_vehicle,antenna_size))
    zf_matrix = np.zeros((num_sample,num_vehicle,antenna_size),dtype=complex)

    for i in range(antenna_size):
        for j in range(num_vehicle):
            for m in range(num_sample):
                steering_vector[m][j][i] = np.exp(-1j * np.pi * i * np.cos(theta[j][m]))

                # attention this steering_vector is the transposed steering vector
                pathloss[m][j][i] = Path_loss(real_distance[j, m])

    CSI_o = np.multiply(pathloss, np.conjugate(steering_vector))
    CSI = sqrt(antenna_size) * CSI_o
    analog_rad = np.zeros((num_sample,antenna_size,config_parameter.rf_size))
    digital_part = np.zeros((num_sample,config_parameter.rf_size,num_vehicle),dtype=complex)
    for n in range(num_sample):
        zf_matrix[n,:,:] = zero_forcing(CSI[n,:,:]).T
        analog_rad[n],digital_part[n] = svd_csi(CSI[n,:,:])
        #analog_rad[n],digital_part[n] = svd_zf(zf_matrix[n,:,:].T)
    input_whole = np.zeros(shape=(num_sample, num_vehicle,
               5 * antenna_size))
    input_whole[:,:,0:1*antenna_size] = np.real(CSI)
    input_whole[:,:,1*antenna_size:2*antenna_size] = np.imag(CSI)
    #input_whole[:,:,2*antenna_size:3*antenna_size] = np.real(zf_matrix)
    #input_whole[:, :, 3 * antenna_size:4 * antenna_size] = np.imag(zf_matrix)
    input_whole[:,0:config_parameter.rf_size,2*antenna_size:3*antenna_size] = np.transpose(analog_rad,axes=(0,2,1))
    input_whole[:,0:config_parameter.rf_size,3*antenna_size:3*antenna_size+num_vehicle] = np.transpose(np.real(digital_part),axes=(0,2,1))
    input_whole[:,0:config_parameter.rf_size,4*antenna_size:4*antenna_size+num_vehicle] = np.transpose(np.imag(digital_part),axes=(0,2,1))
    return input_whole
def tf_Output2PrecodingMatrix_rad(Output):
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
    antenna_size_o = tf.cast(antenna_size, tf.float32)
    g = tf.sqrt(antenna_size_o)
    g= tf.cast(g, tf.complex64)
    Analog_part_reshaped_o = tf.cast(Analog_part_reshaped, tf.complex64)
    Analog_Matrix = g*tf.exp(1j * Analog_part_reshaped_o)


    adder = antenna_size * config_parameter.rf_size
    Digital_real = Output[:, adder: adder + config_parameter.rf_size * num_vehicle]
    Digital_imginary = Output[:, adder + config_parameter.rf_size * num_vehicle:]

    Digital_real_reshaped = tf.reshape(Digital_real, (batch_size,config_parameter.rf_size, num_vehicle))
    Digital_imginary_reshaped = tf.reshape(Digital_imginary, (batch_size,config_parameter.rf_size, num_vehicle))


    Digital_Matrix = tf.complex(Digital_real_reshaped, Digital_imginary_reshaped)

    print(Digital_Matrix)
    return Analog_Matrix, Digital_Matrix
def Conversion2CSI(angle,distance):
    theta = angle.T
    real_distance = distance.T
    if config_parameter.mode == "V2I":
        antenna_size = config_parameter.antenna_size
        num_vehicle = config_parameter.num_vehicle
    elif config_parameter.mode == "V2V":
        antenna_size = config_parameter.vehicle_antenna_size
        num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar + config_parameter.num_horizoncar
    num_sample = theta.shape[1]
    steering_vector = np.zeros((num_sample,num_vehicle,antenna_size,),dtype=complex)
    pathloss = np.zeros((num_sample,num_vehicle,antenna_size))
    zf_matrix = np.zeros((num_sample,num_vehicle,antenna_size),dtype=complex)



    for i in range(antenna_size):
        for j in range(num_vehicle):
            for m in range(num_sample):
                steering_vector[m][j][i] = np.exp(-1j*np.pi*i*np.cos(theta[j][m]))

                #attention this steering_vector is the transposed steering vector
                pathloss[m][j][i] = Path_loss(real_distance[j,m])

    CSI_o = np.multiply(pathloss, np.conjugate(steering_vector))
    CSI = sqrt(antenna_size) * CSI_o
    for n in range(num_sample):

        zf_matrix[n,:,:] = zero_forcing(CSI[n,:,:]).T
    input_whole = np.zeros(shape=(num_sample, num_vehicle,
               4 * antenna_size))
    input_whole[:,:,0:1*antenna_size] = np.real(CSI)
    input_whole[:,:,1*antenna_size:2*antenna_size] = np.imag(CSI)
    #input_whole[:,:,2*antenna_size:3*antenna_size] = np.real(zf_matrix)
    #input_whole[:, :, 3 * antenna_size:4 * antenna_size] = np.imag(zf_matrix)

    return input_whole
"input generation"
def simple_precoder(theta,distance):
    # theta shape[num_vehicle, 200]
    # distance shape[num_vehicle, 200]
    #print(distance)
    if config_parameter.mode == "V2I":
        antenna_size = config_parameter.antenna_size
        num_vehicle = config_parameter.num_vehicle
    elif config_parameter.mode == "V2V":
        antenna_size = config_parameter.vehicle_antenna_size
        num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar + config_parameter.num_horizoncar

    idx = np.arange(antenna_size)
    precoder= np.zeros((theta.shape[1],antenna_size, num_vehicle), dtype=complex)
    distance_sum = np.sum(distance,axis = 0)
    print("distance_sum",distance_sum)
    distance_norm = config_parameter.power / distance_sum
    distance_mod = np.zeros((num_vehicle,theta.shape[1]))
    for m in range(num_vehicle):

        distance_mod[m] = distance_norm* distance[m]
    print("distance shape",distance_mod)
    for batchindex in range(theta.shape[1]):
        for carindex in range(theta.shape[0]):

            precoder[batchindex,:,carindex] = np.exp(1j * np.pi * idx * np.cos(theta[carindex,batchindex]))
            precoder[batchindex,:,carindex] = precoder[batchindex,:,carindex] * distance_mod[carindex,batchindex]
    return precoder
def zero_forcing(CSI):
    print(CSI)
    H_inv = np.linalg.pinv(CSI)


    max_power = config_parameter.power
    magnitude_sum = np.sum(np.abs(H_inv))

    adjustment_factor = max_power / magnitude_sum
    H_inv = np.multiply(H_inv, adjustment_factor)

    return H_inv
def tf_zero_forcing(CSI):

    H_inv_real = tf.linalg.pinv(tf.math.real(CSI))
    H_inv_imag = tf.linalg.pinv(tf.math.imag(CSI))
    H_inv = tf.complex(H_inv_real,H_inv_imag)
    max_power = config_parameter.power
    magnitude_sum = tf.reduce_sum(tf.abs(H_inv),axis=[1,2],keepdims=True)

    adjustment_factor = max_power / magnitude_sum
    adjustment_factor= tf.cast(adjustment_factor,tf.complex128)
    H_inv = tf.multiply(H_inv, adjustment_factor)

    return H_inv
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
def Path_loss(distance):
    #print("distance",distance.dtype)
    #distan#ce = float(distance)
    pathloss = sqrt(config_parameter.alpha * ((distance / config_parameter.d0) ** config_parameter.path_loss_exponent))
    #print("pathloss",pathloss)
    #pathloss= pathloss.astype(np.float32)
    return pathloss
def tf_Path_loss(distance):
    #distance as a list
    alpha = tf.constant(config_parameter.alpha, dtype=tf.float32)
    d0 = tf.constant(config_parameter.d0, dtype=tf.float32)
    path_loss_exponent = tf.constant(config_parameter.path_loss_exponent, dtype=tf.float32)
    distance = tf.cast(distance, dtype=tf.float32)
    path_loss = tf.sqrt(alpha * ((distance/ d0) ** path_loss_exponent))
    return path_loss
def Reflection_coefficient(distance_this_vehicle):
    distance = tf.cast(distance_this_vehicle,tf.complex64)
    fading_coefficient = tf.cast(config_parameter.fading_coefficient,tf.complex64)
    beta = fading_coefficient/(2*distance)
    return beta
"conversion of output"
def tf_Output2PrecodingMatrix_rad_mod(Output,analog_ref,digital_ref):
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
    antenna_size_o = tf.cast(antenna_size, tf.float32)
    g = tf.sqrt(antenna_size_o)
    g= tf.cast(g, tf.complex128)
    Analog_part_reshaped_o = tf.cast(Analog_part_reshaped, tf.complex128)
    #analog_ref, digital_ref = svd_zf(zf_matrix)
    #Analog_Matrix = g*tf.exp(1j * Analog_part_reshaped_o)
    analog_ref = tf.cast(analog_ref, tf.complex128)
    analog_ref = tf.transpose(analog_ref,perm=[0,2,1])

    Analog_Matrix=g*tf.multiply(tf.exp(1j*analog_ref),tf.exp(1j*Analog_part_reshaped_o))

    adder = antenna_size * config_parameter.rf_size
    Digital_real = Output[:, adder: adder + config_parameter.rf_size * num_vehicle]
    Digital_imginary = Output[:, adder + config_parameter.rf_size * num_vehicle:]

    Digital_real_reshaped = tf.reshape(Digital_real, (batch_size,config_parameter.rf_size, num_vehicle))
    Digital_imginary_reshaped = tf.reshape(Digital_imginary, (batch_size,config_parameter.rf_size, num_vehicle))
    Digital_real_reshaped = tf.cast(Digital_real_reshaped, tf.float64)
    print("digital_ref",digital_ref)
    Digital_real_a = Digital_real_reshaped+tf.math.real(digital_ref)
    Digital_imginary_reshaped = tf.cast(Digital_imginary_reshaped, tf.float64)
    Digital_imginary_a = Digital_imginary_reshaped+tf.math.imag(digital_ref)


    Digital_Matrix = tf.complex(Digital_real_a, Digital_imginary_a)

    print(Digital_Matrix)
    return Analog_Matrix, Digital_Matrix
def tf_Output2digitalPrecoding(Output,zf_matrix,distance):
    shape = tf.shape(Output)
    batch_size = shape[0]

    if config_parameter.mode == "V2I":
        antenna_size = config_parameter.antenna_size
        num_vehicle = config_parameter.num_vehicle
    elif config_parameter.mode == "V2V":
        antenna_size = config_parameter.vehicle_antenna_size
        num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar + config_parameter.num_horizoncar

    antenna_size_f = tf.cast(antenna_size, tf.float32)
    # dont forget here we are inputing a whole batch
    G = tf.math.sqrt(antenna_size_f)
    Real = Output[:, 0:antenna_size * num_vehicle]
    Imag = Output[:, antenna_size * num_vehicle:]
    # this reshape has been tested
    Real_reshaped = tf.reshape(Real, (batch_size,antenna_size, num_vehicle))
    Imag_reshaped = tf.reshape(Imag, (batch_size,antenna_size, num_vehicle))
    Digital_Matrix = tf.complex(Real_reshaped, Imag_reshaped)
    max_power = tf.constant(config_parameter.power, dtype=tf.float32)


    #shape(8,2)
    magnitude_sum = tf.reduce_sum(tf.abs(Digital_Matrix), axis=[1, 2], keepdims=True)
    adjustment_factor = max_power / magnitude_sum
    adjustment_factor = tf.cast(adjustment_factor, tf.complex64)
    Digital_Matrix = Digital_Matrix * adjustment_factor
    #Digital_Matrix = powerallocated(Digital_Matrix,distance)
    if zf_matrix is None:
        Digital_Matrix_e = Digital_Matrix
    else:
        Digital_Matrix_e = tf.transpose(zf_matrix,perm=[0,2,1]) + tf.cast(Digital_Matrix, tf.complex128)
    return Digital_Matrix_e
def tf_Output2PrecodingMatrix_powerallocated(Output):
    shape = tf.shape(Output)
    batch_size = shape[0]

    if config_parameter.mode == "V2I":
        antenna_size = config_parameter.antenna_size
        num_vehicle = config_parameter.num_vehicle
    elif config_parameter.mode == "V2V":
        antenna_size = config_parameter.vehicle_antenna_size
        num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar + config_parameter.num_horizoncar

    antenna_size_f = tf.cast(antenna_size, tf.float32)
    # dont forget here we are inputing a whole batch
    G = tf.math.sqrt(antenna_size_f)
    #Analog_Matrix = tf.zeros((antenna_size, config_parameter.rf_size), dtype=tf.complex128)
    #Digital_real_Matrix = tf.zeros((config_parameter.rf_size, num_vehicle))
    #Digital_im_Matrix = tf.zeros((config_parameter.rf_size, num_vehicle))
    #Digital_Matrix = tf.zeros((config_parameter.rf_size, num_vehicle), dtype=tf.complex128)
    Analog_part = Output[:, 0:antenna_size * config_parameter.rf_size]
    Analog_part_reshaped = tf.reshape(Analog_part, (batch_size,antenna_size, config_parameter.rf_size))
    antenna_size_o = tf.cast(antenna_size, tf.float32)
    g = tf.sqrt(antenna_size_o)
    g= tf.cast(g, tf.complex64)
    Analog_part_reshaped_o = tf.cast(Analog_part_reshaped, tf.complex64)
    Analog_Matrix = g*tf.exp(1j * Analog_part_reshaped_o)
    Power_allocated = Output[:, antenna_size * config_parameter.rf_size:antenna_size * config_parameter.rf_size + num_vehicle]
    Power_allocated_reshaped = tf.reshape(Power_allocated, (batch_size,1, num_vehicle))
def tf_Output2PrecodingMatrix(Output):
    shape = tf.shape(Output)
    batch_size = shape[0]
    print("batch_size",batch_size)

    if config_parameter.mode == "V2I":
        antenna_size = config_parameter.antenna_size
        num_vehicle = config_parameter.num_vehicle
    elif config_parameter.mode == "V2V":
        antenna_size = config_parameter.vehicle_antenna_size
        num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar + config_parameter.num_horizoncar

    antenna_size_f = tf.cast(antenna_size, tf.float32)
    # dont forget here we are inputing a whole batch
    G = tf.math.sqrt(antenna_size_f)
    #Analog_Matrix = tf.zeros((antenna_size, config_parameter.rf_size), dtype=tf.complex128)
    #Digital_real_Matrix = tf.zeros((config_parameter.rf_size, num_vehicle))
    #Digital_im_Matrix = tf.zeros((config_parameter.rf_size, num_vehicle))
    #Digital_Matrix = tf.zeros((config_parameter.rf_size, num_vehicle), dtype=tf.complex128)
    Analog_part = Output[:, 0:2*antenna_size * config_parameter.rf_size]
    Analog_real = Output[:,0:antenna_size * config_parameter.rf_size]
    Analog_imginary = Output[:,antenna_size * config_parameter.rf_size:2*antenna_size * config_parameter.rf_size]
    Analog_real_reshaped = tf.reshape(Analog_real, (batch_size,antenna_size, config_parameter.rf_size))
    Analog_imginary_reshaped = tf.reshape(Analog_imginary, (batch_size,antenna_size, config_parameter.rf_size))

    Analog_Matrix = tf.complex(Analog_real_reshaped, Analog_imginary_reshaped)
    Analog_abs = tf.abs(Analog_Matrix)*G

    divided_real = tf.math.real(Analog_Matrix) / Analog_abs
    divided_imag = tf.math.imag(Analog_Matrix) / Analog_abs
    divided_complex = tf.complex(divided_real, divided_imag)
    adder = 2*antenna_size * config_parameter.rf_size
    Digital_real = Output[:, adder: adder + config_parameter.rf_size * num_vehicle]
    Digital_imginary = Output[:, adder + config_parameter.rf_size * num_vehicle:]

    Digital_real_reshaped = tf.reshape(Digital_real, (batch_size,config_parameter.rf_size, num_vehicle))
    Digital_imginary_reshaped = tf.reshape(Digital_imginary, (batch_size,config_parameter.rf_size, num_vehicle))

    Digital_Matrix = tf.complex(Digital_real_reshaped, Digital_imginary_reshaped)

    print(Digital_Matrix)
    return Analog_Matrix, Digital_Matrix
def tf_Precoding_matrix_combine(Analog_matrix, Digital_matrix):
    max_power = tf.constant(config_parameter.power, dtype=tf.float64)

    matrix = tf.matmul(Analog_matrix, Digital_matrix)
    matrix = tf.cast(matrix, dtype=tf.complex128)
    #shape(8,2)
    magnitude_sum = tf.reduce_sum(tf.abs(matrix), axis=[1, 2], keepdims=True)

    adjustment_factor = max_power / magnitude_sum
    adjustment_factor = tf.cast(adjustment_factor,dtype=tf.complex128)
    #matrix = tf.cast(matrix, dtype=tf.complex128)
    normalized_array = tf.multiply(matrix, adjustment_factor)

    return normalized_array
def tf_Precoding_matrix_comb_Powerallocated(Analog_matrix,Digital_matrix,distance):
    if config_parameter.mode == "V2I":
        antenna_size = config_parameter.antenna_size
        num_vehicle = config_parameter.num_vehicle
    elif config_parameter.mode == "V2V":
        antenna_size = config_parameter.vehicle_antenna_size
        num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar + config_parameter.num_horizoncar

    max_power = tf.constant(config_parameter.power, dtype=tf.float64)
    distance_sum = tf.reduce_sum(distance, axis=1,keepdims=True)
    distance_sum =tf.tile(distance_sum, [1,tf.shape(distance)[1]])
    print("distance_sum",distance_sum)
    adjustment_power = distance * max_power / distance_sum
    adjustment_power = tf.cast(adjustment_power, dtype=tf.float64)
    adjustment_power = tf.expand_dims(adjustment_power, axis=1)
    #adjustment_power = tf.tile(adjustment_power, [1, antenna_size])
    print("adjustment_power",adjustment_power)
    matrix = tf.matmul(Analog_matrix, Digital_matrix)
    matrix = tf.cast(matrix, dtype=tf.complex128)
    #shape(8,2)
    magnitude_sum = tf.reduce_sum(tf.abs(matrix), axis=1, keepdims=True)
    print("magnitude_sum",magnitude_sum)
    adjustment_factor =  adjustment_power/ magnitude_sum
    print("adjustment_factor",adjustment_factor)
    adjustment_factor = tf.cast(adjustment_factor,dtype=tf.complex128)
    #matrix = tf.cast(matrix, dtype=tf.complex128)
    normalized_array = tf.multiply(matrix, adjustment_factor)

    return normalized_array
def powerallocated(matrix,distance):
    if config_parameter.mode == "V2I":
        antenna_size = config_parameter.antenna_size
        num_vehicle = config_parameter.num_vehicle
    elif config_parameter.mode == "V2V":
        antenna_size = config_parameter.vehicle_antenna_size
        num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar + config_parameter.num_horizoncar

    max_power = tf.constant(config_parameter.power, dtype=tf.float64)
    distance_sum = tf.reduce_sum(distance, axis=1,keepdims=True)
    distance_sum =tf.tile(distance_sum, [1,tf.shape(distance)[1]])
    print("distance_sum",distance_sum)
    adjustment_power = distance * max_power / distance_sum
    adjustment_power = tf.cast(adjustment_power, dtype=tf.float64)
    adjustment_power = tf.expand_dims(adjustment_power, axis=1)
    #adjustment_power = tf.tile(adjustment_power, [1, antenna_size])
    print("adjustment_power",adjustment_power)
    matrix = tf.cast(matrix, dtype=tf.complex128)
    # shape(8,2)
    magnitude_sum = tf.reduce_sum(tf.abs(matrix), axis=1, keepdims=True)
    print("magnitude_sum", magnitude_sum)
    adjustment_factor = adjustment_power / magnitude_sum
    print("adjustment_factor", adjustment_factor)
    adjustment_factor = tf.cast(adjustment_factor, dtype=tf.complex128)
    # matrix = tf.cast(matrix, dtype=tf.complex128)
    normalized_array = tf.multiply(matrix, adjustment_factor)

    return normalized_array
def tf_Precoding_comb_no_powerconstarint(Analog_matrix, Digital_matrix):
    matrix = tf.matmul(Analog_matrix, Digital_matrix)
    matrix = tf.cast(matrix, dtype=tf.complex128)
    return matrix
"calculation for sum rate"
def tf_loss_sumrate(CSI, precoding_matrix):
    if config_parameter.mode == "V2I":
        antenna_size = config_parameter.antenna_size
        num_vehicle = config_parameter.num_vehicle
    elif config_parameter.mode == "V2V":
        antenna_size = config_parameter.vehicle_antenna_size
        num_vehicle = (
            config_parameter.num_uppercar
            + config_parameter.num_lowercar
            + config_parameter.num_horizoncar
        )

    CSI = tf.cast(CSI, dtype=tf.complex128)
    precoding_matrix = tf.cast(precoding_matrix, dtype=tf.complex128)
    #print("csi",CSI)
    #precoding_shape : 8,2   CSI(2,8)
    this_sinr = tf.linalg.diag_part(
        tf.square(
            tf.math.abs(
                tf.matmul(CSI, precoding_matrix)
            )
        ),
    )  # Shape: (batch_size, num_vehicle)
    print("sum",tf.square(
            tf.abs(
                tf.matmul(CSI, precoding_matrix)
            )
        ))
    sum_other = tf.reduce_sum(
        tf.square(
            tf.abs(
                tf.matmul(CSI, precoding_matrix)
            )
        ),
        axis=2
    )
    #print("sum_other",sum_other)
    #sum_other = tf.squeeze(sum_other, axis=-1)
    #shape = tf.shape(CSI)
    sum_other = sum_other-this_sinr+config_parameter.sigma_k
    #below_threshold = tf.less(sum_other, 1e-10)
    #sum_other = tf.where(below_threshold, tf.fill(tf.shape(sum_other, 1e-10), sum_other))
    #if sum_other < 10e-10:
        #sum_other = tf.constant(10e-10, dtype=tf.float32)
    #print("sum_other",sum_other)
    #sum_other = tf.tile(sum_other, [1, shape[1]]) -this_sinr + config_parameter.sigma_k  # Shape: (batch_size, num_vehicle)
    #print("thissinr",this_sinr)
    sinr = this_sinr / sum_other  # Shape: (batch_size, num_vehicle)
    #print("sinr",sinr)
    #print("sinr",tf.math.log1p(sinr))
    #sumrate = tf.reduce_sum(tf.math.log(1.0 + sinr), axis=1) / tf.math.log(2.0)
    lg2 =tf.cast(tf.math.log1p(1.0),dtype=tf.float64)
    sumrate = tf.reduce_sum(tf.math.log1p(sinr)/ lg2, axis=1)

    return sumrate,sinr

"calculation for sigma"

def tf_sigma_delay_square(steering_vector_h, precoding_matrix_c,beta):
    rou_delay = tf.constant(config_parameter.rou_timedelay, dtype=tf.float32)
    #rou_doppler = tf.constant(config_parameter.rou_doppler, dtype=tf.float32)
    sigma_z = tf.constant(config_parameter.sigma_z, dtype=tf.float32)
    beta_c = tf.cast(beta, dtype=tf.complex64)
    print("beta",beta_c.shape)
    steering_vector_h = tf.cast(steering_vector_h, dtype=tf.complex64)
    precoding_matrix = tf.cast(precoding_matrix_c, dtype=tf.complex64)
    if config_parameter.mode == "V2I":
        antenna_size = config_parameter.antenna_size
        num_vehicle = config_parameter.num_vehicle
    elif config_parameter.mode == "V2V":
        antenna_size = config_parameter.vehicle_antenna_size
        num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar + config_parameter.num_horizoncar
    G = tf.constant(antenna_size, dtype=tf.float32)

    #this_sinr = tf.linalg.diag_part(
     #   tf.square(
      #      tf.math.abs(
       #         tf.matmul(steering_vector_h, precoding_matrix)
        #    )
        #),
    #)
    #print("shape",this_sinr)
    #print("thissinr",this_sinr.numpy())
    #print("beta_in",beta_c)
    #this_sinr_b = tf.multiply(tf.square(tf.abs(tf.reduce_mean(beta_c,axis=2))),this_sinr)
    print(tf.square(tf.abs(beta_c[:,:,0])))
    #print("thissinr_b",this_sinr_b.numpy())
    #beta_n = tf.cast(tf.transpose(tf.abs(beta_c), perm=[0, 2, 1]), dtype=tf.complex64)
    beta_squ = tf.square(tf.abs(beta_c[:,:,0]))

    beta_squ = tf.expand_dims(beta_squ, axis=-1)
    #beta_squ = tf.fill(tf.shape(beta_squ),2.0)
    print("beta_squ",beta_squ)
    total = tf.multiply(beta_squ,
        tf.square(
            tf.abs(
                tf.matmul(steering_vector_h, precoding_matrix)
            )
        ))
    #precoding_withB = tf.multiply(beta_n,precoding_matrix)
    sum_all = tf.reduce_sum(total,axis=2)

    this_sinr_b = tf.linalg.diag_part(total)
    print("sum_all",tf.square(tf.abs(tf.matmul(steering_vector_h, precoding_matrix))))
    print("this_sinr_b",this_sinr_b)
    print("sum_all",beta_squ*tf.square(tf.abs(tf.matmul(steering_vector_h, precoding_matrix))))
    #print("sum_all1",tf.square(tf.abs(tf.matmul(steering_vector_h,precoding_withB))).numpy())
    #print("sum_all",sum_all.numpy())
    #sum_other = tf.squeeze(sum_other, axis=-1)
    #shape = tf.shape(CSI)
    sum_other = tf.square(G)*(sum_all-this_sinr_b)
    #print("sum_other",sum_other.numpy())
    this_sinr_gain = tf.square(G) * this_sinr_b
    print("this_sinr_gain",this_sinr_gain)
    print("sum_other",sum_other + sigma_z)
    #output should be(batch,num_vehicle)
    return tf.square(rou_delay)* (sum_other + sigma_z)/this_sinr_gain


"calculation for CRB"
def tf_CRB_distance(Sigma_time_delay_2):
    c = tf.constant(config_parameter.c, dtype=tf.float32)
    CRB_d = Sigma_time_delay_2 * tf.square(tf.divide(c,2.0))
    #crlb_d_inv = tf.divide(1.0, Sigma_time_delay_2) * tf.square(tf.divide(2.0, c))
    #CRB_d = tf.divide(1.0, crlb_d_inv)
    #abs_CRB_d = tf.abs(CRB_d)
    #len = tf.shape(abs_CRB_d)[1]
    #crb_dist = tf.reduce_sum(abs_CRB_d, axis=1) / tf.cast(len, dtype=tf.float32)
    return CRB_d

def tf_CRB_angle(beta,precoding_matrix,theta):
    #partial should be a(batch,num_vehicle)
    matched_filter_gain = tf.constant(config_parameter.matched_filtering_gain,tf.float32)
    # consider receive antenna equal to transmit antenna
    if config_parameter.mode == "V2I":
        antenna_size = config_parameter.antenna_size
        num_vehicle = config_parameter.num_vehicle
    elif config_parameter.mode == "V2V":
        antenna_size = config_parameter.vehicle_antenna_size
        num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar + config_parameter.num_horizoncar
    #G = tf.constant(antenna_size, dtype=tf.float32)
    #firstitem = tf.multiply(G,tf.square(tf.abs(beta)))
    #first_item_m = tf.multiply(firstitem,tf.square(matched_filter_gain))
    #nt = tf.range(2, antenna_size + 1)




    partial = tf_Echo_partial(beta,precoding_matrix,theta)

    partial_h = tf.math.conj(partial)

    #partial shape = (batch_size,vehicle)
    #partial_hermite = tf.transpose(tf.conj(partial), perm=[0, 2, 1])
    partial_square = tf.square(tf.abs(partial))
    #sigma_rk_inv = 1 / config_parameter.sigma_rk
    CRB_theta = config_parameter.sigma_rk / partial_square # I think it should be elementwise multiplication
    #reciprocal = tf.math.reciprocal(tensor)
    abs_CRB_theta = tf.abs(CRB_theta)
    #shape = tf.shape(abs_CRB_theta)
    #CRB_theta = tf.reduce_sum(abs_CRB_theta, axis=1)
    return abs_CRB_theta
def tf_Echo_partial(beta_o,precoding_matrix,theta_o):
    beta = beta_o[:,:,0]
    theta = theta_o
    print(theta)
    #this one is function Echo_partial_beta written in tensorflow
    matched_filter_gain = tf.constant(config_parameter.matched_filtering_gain,tf.float32)
    if config_parameter.mode == "V2I":
        antenna_size = config_parameter.antenna_size
        num_vehicle = config_parameter.num_vehicle
    elif config_parameter.mode == "V2V":
        antenna_size = config_parameter.vehicle_antenna_size
        num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar + config_parameter.num_horizoncar

    #sin_theta =tf.sin(theta)
    #cos_theta = tf.cos(theta)
    pi = tf.constant(np.pi,tf.complex128)

    nt = tf.range(1,antenna_size)
    #nt_complex = tf.cast(nt, tf.float32)  # Convert to float32
    nt_complex = tf.cast(nt, tf.complex128)

    antenna_size_o = tf.cast(antenna_size,tf.complex128)
    matched_filter_gain = tf.cast(matched_filter_gain,tf.complex128)
    sin_theta = tf.cast(tf.sin(theta),tf.complex128)
    cos_theta = tf.cast(tf.cos(theta),tf.complex128)
    beta = tf.cast(beta,tf.complex128)
    item_first = -tf.sqrt(antenna_size_o)*matched_filter_gain*beta
    print("item_first",item_first)
    item_exp2 = 1j*pi*nt_complex*sin_theta[:,:,tf.newaxis]
    print("item_exp2",item_exp2)
    item_exp1 = tf.exp(1j * pi * nt_complex * cos_theta[:,:,tf.newaxis])
    print("item_exp1",item_exp1)

    item_exp = tf.multiply(item_exp2,item_exp1) #nt is (antenna_size-1) theta is (batch,num_vehicle)
    print("item_exp",item_exp)
    #print("precoding_matrix",tf.shape(precoding_matrix[:,nt,:]))
    precoding_matrix = tf.cast(precoding_matrix,tf.complex128)
    item_second = tf.reduce_sum(tf.multiply(precoding_matrix[:,1:antenna_size,:],tf.transpose(item_exp,perm=[0,2,1])),axis=1)
    print("item_second",item_second)
    item_second = tf.squeeze(item_second)
    partial = tf.multiply(item_first,item_second)
    print("partial",partial)
    #partial =  tf.squeeze(partial)
    return partial
"loss combined"

