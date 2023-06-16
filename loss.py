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

def svd_csi(CSI):
    U, s, Vh = np.linalg.svd(CSI, full_matrices=False)
    num_rf = config_parameter.rf_size
    analog_part = Vh[:, :num_rf]
    digital_part = np.dot(CSI,analog_part) #not right here
    return analog_part,digital_part

def svd_zf(zf_matrix):
    U, s, Vh = np.linalg.svd(zf_matrix, full_matrices=False)
    num_rf = config_parameter.rf_size
    analog_part = U[:, :num_rf]
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
    analog_precoder = complex_matrix_to_polar(analog_part)
    digital_precoder = complex_matrix_to_polar(digital_part)
    return analog_precoder,digital_precoder
"load data"
def load_data():
    data = np.load("dataset.npy")
    distance = data[:, 4:]
    print("distance",distance[10,:])
    angle = data[:, :4]
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
    zf_matrix = np.zeros((num_sample, num_vehicle, antenna_size), dtype=complex)
    for n in range(num_sample):
        zf_matrix[n, :, :] = zero_forcing(CSI[n, :, :]).T
        # print(zf_matrix.shape)
    # generate the input for the neural network
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
    input_whole[:,:,2*antenna_size:3*antenna_size] = np.real(zf_matrix)
    input_whole[:, :, 3 * antenna_size:4 * antenna_size] = np.imag(zf_matrix)

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
    H_inv = np.linalg.pinv(CSI)


    max_power = config_parameter.power
    magnitude_sum = np.sum(np.abs(H_inv))

    adjustment_factor = max_power / magnitude_sum
    H_inv = np.multiply(H_inv, adjustment_factor)

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
    pathloss = sqrt(config_parameter.alpha * ((distance / config_parameter.d0) ** config_parameter.path_loss_exponent))
    return pathloss
def Reflection_coefficient(distance_this_vehicle):
    beta = config_parameter.fading_coefficient/(2*distance_this_vehicle)
    return beta
"conversion of output"
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
def tf_Output2digitalPrecoding(Output,zf_matrix):
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
    Digital_Matrix_e = tf.transpose(zf_matrix,perm=[0,2,1]) + tf.cast(Digital_Matrix, tf.complex128)
    return Digital_Matrix_e

def tf_Output2PrecodingMatrix(Output):
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
    max_power = tf.constant(config_parameter.power, dtype=tf.float32)

    matrix = tf.matmul(Analog_matrix, Digital_matrix)
    #shape(8,2)
    magnitude_sum = tf.reduce_sum(tf.abs(matrix), axis=[1, 2], keepdims=True)
    adjustment_factor = max_power / magnitude_sum
    adjustment_factor = tf.cast(adjustment_factor,dtype=tf.complex64)
    #matrix = tf.cast(matrix, dtype=tf.complex128)
    normalized_array = tf.multiply(matrix, adjustment_factor)

    return normalized_array
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
    print("csi",CSI)
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
    print("sum_other",sum_other)
    #sum_other = tf.squeeze(sum_other, axis=-1)
    #shape = tf.shape(CSI)
    sum_other = sum_other-this_sinr+config_parameter.sigma_k
    #below_threshold = tf.less(sum_other, 1e-10)
    #sum_other = tf.where(below_threshold, tf.fill(tf.shape(sum_other, 1e-10), sum_other))
    #if sum_other < 10e-10:
        #sum_other = tf.constant(10e-10, dtype=tf.float32)
    print("sum_other",sum_other)
    #sum_other = tf.tile(sum_other, [1, shape[1]]) -this_sinr + config_parameter.sigma_k  # Shape: (batch_size, num_vehicle)
    print("thissinr",this_sinr)
    sinr = this_sinr / sum_other  # Shape: (batch_size, num_vehicle)
    print("sinr",sinr)
    print("sinr",tf.math.log1p(sinr))
    #sumrate = tf.reduce_sum(tf.math.log(1.0 + sinr), axis=1) / tf.math.log(2.0)
    lg2 =tf.cast(tf.math.log1p(1.0),dtype=tf.float64)
    sumrate = tf.reduce_sum(tf.math.log1p(sinr)/ lg2, axis=1)

    return sumrate

"calculation for sigma"

def tf_sigma_delay_square(steering_vector_h, precoding_matrix_c,beta):
    rou_delay = tf.constant(config_parameter.rou_timedelay, dtype=tf.float32)
    #rou_doppler = tf.constant(config_parameter.rou_doppler, dtype=tf.float32)
    sigma_z = tf.constant(config_parameter.sigma_z, dtype=tf.float32)
    beta_c = tf.cast(beta, dtype=tf.complex64)
    steering_vector_h = tf.cast(steering_vector_h, dtype=tf.complex64)
    precoding_matrix = tf.cast(precoding_matrix_c, dtype=tf.complex64)
    if config_parameter.mode == "V2I":
        antenna_size = config_parameter.antenna_size
        num_vehicle = config_parameter.num_vehicle
    elif config_parameter.mode == "V2V":
        antenna_size = config_parameter.vehicle_antenna_size
        num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar + config_parameter.num_horizoncar
    G = tf.constant(antenna_size, dtype=tf.float32)

    this_sinr = tf.linalg.diag_part(
        tf.square(
            tf.math.abs(
                tf.matmul(steering_vector_h, precoding_matrix)
            )
        ),
    )
    print("shape",this_sinr)
    print("beta",beta_c)
    this_sinr_b = tf.multiply(tf.square(tf.abs(tf.reduce_mean(beta_c,axis=2))),this_sinr)
    beta_n = tf.cast(tf.transpose(tf.square(tf.abs(beta_c)), perm=[0, 2, 1]), dtype=tf.complex64)
    precoding_withB = tf.multiply(beta_n,precoding_matrix)
    sum_all = tf.reduce_sum(
        tf.square(
            tf.abs(
                tf.matmul(steering_vector_h, precoding_withB)
            )
        )
    ,axis=2)

    #sum_other = tf.squeeze(sum_other, axis=-1)
    #shape = tf.shape(CSI)
    sum_other = G*(sum_all-this_sinr_b)
    this_sinr_gain = G * this_sinr_b
    #output should be(batch,num_vehicle)
    return rou_delay* (sum_other + sigma_z)/this_sinr_gain


"calculation for CRB"
def tf_CRB_distance(Sigma_time_delay_2):
    c = tf.constant(config_parameter.c, dtype=tf.float32)
    crlb_d_inv = tf.divide(1.0, Sigma_time_delay_2) * tf.square(tf.divide(2.0, c))
    CRB_d = tf.divide(1.0, crlb_d_inv)
    abs_CRB_d = tf.abs(CRB_d)
    crb_dist = tf.reduce_sum(abs_CRB_d, axis=1)
    return crb_dist

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
    G = tf.constant(antenna_size, dtype=tf.float32)
    firstitem = tf.multiply(G,tf.square(tf.abs(beta)))
    first_item_m = tf.multiply(firstitem,tf.square(matched_filter_gain))
    nt = tf.range(2, antenna_size + 1)




    #partial = tf_Echo_partial(beta,precoding_matrix,theta)
    #partial_h = tf.conj(partial)

    #partial shape = (batch_size,vehicle)
    #partial_hermite = tf.transpose(tf.conj(partial), perm=[0, 2, 1])
    sigma_rk_inv = 1 / config_parameter.sigma_rk
    CRB_theta = 1 / (sigma_rk_inv * partial_square) # I think it should be elementwise multiplication
    #reciprocal = tf.math.reciprocal(tensor)
    #abs_CRB_theta = tf.abs(CRB_theta)
    #shape = tf.shape(abs_CRB_theta)
    CRB_theta = tf.reduce_sum(abs_CRB_theta, axis=1)
    return CRB_theta
def tf_Echo_partial(beta,precoding_matrix,theta):
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
    pi = tf.constant(np.pi,tf.complex64)

    nt = tf.range(2,antenna_size+1)
    nt_complex = tf.cast(nt, tf.float32)  # Convert to float32
    nt_complex = tf.cast(nt, tf.complex64)

    antenna_size_o = tf.cast(antenna_size,tf.complex128)
    matched_filter_gain = tf.cast(matched_filter_gain,tf.complex128)
    item_first = -tf.sqrt(antenna_size_o)*beta[:,tf.newaxis]*matched_filter_gain   # beta is ( batch,num_vehicle)

    item_exp = (tf.cos(pi*(nt_complex-1)*tf.cos(theta[:,tf.newaxis]))+1j*tf.sin(pi*(nt_complex-1)*tf.cos(theta[:,tf.newaxis])))*1j*(nt_complex-1)*sin(theta[:,tf.newaxis]) #nt is (antenna_size-1) theta is (batch,num_vehicle)
    item_second = precoding_matrix[:,nt,:]*item_exp
    partial = tf.reduce_sum(item_first*item_second,axis=1)
    partial =  tf.squeeze(partial)
    return partial
"loss combined"

