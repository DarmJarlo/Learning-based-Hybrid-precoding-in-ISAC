"This file contains all the test cases for the functions in the project"
import random
from math import cos, sqrt,pi,sin
import numpy as np
import cmath
import config_parameter
from scipy.signal import correlate
from scipy.optimize import fmin_bfgs
from scipy import signal
import tensorflow as tf
import loss
import matplotlib.pyplot as plt
def Test_sigma_delay(steering_vector_h,precoding_matrix_c,beta):
    sigma = loss.tf_sigma_delay_square(steering_vector_h, precoding_matrix_c, beta)

def Test_steering_vector_generation(angle,distance):
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
    pathloss = np.zeros((num_sample,num_vehicle,antenna_size))
    beta = np.zeros((num_sample,num_vehicle,antenna_size),dtype=complex)

    for i in range(antenna_size):
        for j in range(num_vehicle):
            for m in range(num_sample):
                steering_vector[m][j][i] = np.exp(-1j * np.pi * i * np.cos(theta[j][m]))
                pathloss[m][j][i] = loss.Path_loss(real_distance[j,m])
                beta[m][j][i] = loss.Reflection_coefficient(real_distance[j,m])
    #print("steering_vector",steering_vector)
    print("pathloss",pathloss)
    print("beta",beta)
    print("beta",beta.shape)
    return steering_vector,pathloss,beta
def Test_CSI_generation(steering_vector,path_loss):
    if config_parameter.mode == "V2I":
        antenna_size = config_parameter.antenna_size
        num_vehicle = config_parameter.num_vehicle
    elif config_parameter.mode == "V2V":
        antenna_size = config_parameter.vehicle_antenna_size
        num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar + config_parameter.num_horizoncar

    CSI_o = np.multiply(path_loss, np.conjugate(steering_vector))
    CSI = sqrt(antenna_size) * CSI_o
    print("CSI",CSI)
    return CSI


def Test_sumrate(CSI,precoding_matrix):
    sum_rate=loss.tf_loss_sumrate(CSI, precoding_matrix)
    print("sum_rate",sum_rate)
    print("sum_rate",sum_rate.numpy())

def Test_svd():
    real_part = np.random.randint(low=-10, high=10, size=(8, 4))
    imaginary_part = np.random.randint(low=-10, high=10, size=(8, 4))
    random.seed(1)
    # Create the complex matrix
    complex_matrix = real_part + 1j * imaginary_part
    matrix_rad = np.angle(complex_matrix)
    complex_matrix_norm = np.exp(1j*matrix_rad)
    # Print the complex matrix
    print("complex_matrix",complex_matrix)
    print("complex_matrix_rad",matrix_rad)
    # Perform SVD
    analog_precoder, digital_precoder = loss.svd_zf(complex_matrix_norm)
    print("analog_precoder",analog_precoder)
    print("digital_precoder",digital_precoder)
    reconstruct = np.dot(analog_precoder, digital_precoder)
    reconstruct_rad = np.angle(reconstruct)
    print("reconstruct",reconstruct_rad)
    print("error_rad",matrix_rad-reconstruct_rad)
    print("error",np.exp(1j*matrix_rad)-reconstruct)

    return complex_matrix,complex_matrix_norm,reconstruct,reconstruct_rad
'''
angle,distance = loss.load_data()
selected_angle = angle[0:4,:]
selected_distance = distance[0:4,:]
precoding_matrix = loss.simple_precoder(selected_angle.T,selected_distance.T)
CSI= np.zeros((4,4,8),dtype=complex)
print(CSI.shape)
for i in range(4):
    #cosangle = np.cos(np.pi*cos(selected_angle[i]))
    print("selected_angle",selected_angle[i])
    print("selected_angle",np.cos(selected_angle[i]))
    print("selected_angle",np.cos(np.pi*np.cos(selected_angle[i])))
    CSI[i] = loss.calculate_steer_vector(selected_angle[i]).T
print("csi",CSI)
CSI_o = np.multiply(5,CSI)
print("csi111111",CSI_o)
Test_sumrate(CSI,precoding_matrix)
'''
#angle = np.array([[0.1,0.2,0.1,0.2],[0.2,0.3,0.2,0.3],[0.7,0.8,0.7,0.8],[0.8,0.9,0.8,0.9]])
angle = np.array([[0.1,0.2,0.1,0.2],[0.2,0.5,0.2,0.5]])
#distance = np.array([[50,50,100,100],[100,100,200,200],[60,120,60,120],[120,60,120,60]])
distance = np.array([[50,20,200,100],[10,20,200,400]])
print(distance.shape)
print(angle.shape)


steering_vector,path_loss,beta=Test_steering_vector_generation(angle,distance)
print("steering_vector",steering_vector)
print("path_loss",path_loss)
print("beta1",beta)
#steering_vector=np.array([[[1+1j,2+2j],[5+5j,6+6j], [2,2],[5,6]],[[1+1j,2+2j],[3+3j,4+4j],[5+5j,6+6j],[7+7j,8+8j]]])
#precoding_matrix=np.array([[[1,2,3,4],[5,10,20,30]],[[40,50,15,20],[15,30,5,6]],[15,20]],[[15,20],[15,30],[5,6],[15,20]]])

precoding_matrix = abs(loss.simple_precoder(angle.T,distance.T))
print("precoding_matrix1",precoding_matrix)
partial = loss.tf_Echo_partial(beta,precoding_matrix,angle)
#sigma = Test_sigma_delay(steering_vector,precoding_matrix_c=precoding_matrix, beta=beta)
'''
complex_matrix,complex_matrix_norm,reconstruct,reconstruct_rad=Test_svd()
idx = np.arange(8)
angle_set = np.arange(1, 361) / 180 * np.pi
Hset = np.exp(-1j * np.pi * idx.reshape(-1, 1) * np.cos(angle_set))
r1 = np.dot(Hset.T, complex_matrix_norm)
r2 = np.dot(Hset.T, reconstruct)
print("r90", r1[28:35])

# r = np.dot(r1,r2)
plt.polar(angle_set, np.abs(r2))
plt.show()
'''
'''
angle = np.array([[0.1,0.2,0.1,0.2],[0.2,0.3,0.2,0.3],[0.7,0.8,0.7,0.8],[0.8,0.9,0.8,0.9]])

distance = np.array([[50,50,100,100],[100,100,200,200],[60,120,60,120],[120,60,120,60]])
print(distance.shape)
print(angle.shape)
steering_vector,path_loss,beta=Test_steering_vector_generation(angle,distance)
print("s2",steering_vector[1,1])
print(np.exp(-1j * np.pi * 1 * np.cos(0.3)))
print(np.exp(-1j * np.pi * 2 * np.cos(0.3)))
random.seed(1)
precoding_matrix = np.random.randn(4, 8, 4) + 1j * np.random.randn(4, 8, 4)
print(precoding_matrix)
CSI=Test_CSI_generation(steering_vector,path_loss)

#csi shape(batch,num vehicle,antenna_size)   precoding_matrix shape(batch,antenna_size,num vehicle)
CSI=np.array([[[1+1j,2+2j,3+3j,4+4j],[5+5j,6+6j,7+7j,8+8j]], [[2,2,5,5],[5,6,7,8]],[[1+1j,2+2j,3+3j,4+4j],[5+5j,6+6j,7+7j,8+8j]]])
precoding_matrix=np.array([[[1,2],[5,10],[20,30],[40,50]],[[15,20],[15,30],[5,6],[15,20]],[[15,20],[15,30],[5,6],[15,20]]])
print(CSI.shape)
print(precoding_matrix.shape)
Test_sumrate(CSI,precoding_matrix)
print("--------------------Test SVD---------------------")
A = np.random.rand(8, 4) + 1j * np.random.rand(8, 4)
analog, digital = loss.svd_zf(A)
print(A)
print(analog*digital)
'''