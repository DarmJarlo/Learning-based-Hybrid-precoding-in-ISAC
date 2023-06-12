import numpy as np
import matplotlib.pyplot as plt
import config_parameter
import loss
import loss_all
import tensorflow as tf
from Trainv2_4inputs import load_model, generate_input


def translate_precoding_matrix(matrix):
    translated_matrix = np.empty(matrix.shape, dtype=np.complex128)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            element = matrix[i, j]
            magnitude = np.abs(element)
            phase_degrees = np.angle(element, deg=True)
            phase_radians = np.deg2rad(phase_degrees)

            translated_element = magnitude * np.exp(1j * phase_radians)
            translated_matrix[i, j] = translated_element

    return translated_matrix
import math
model = load_model()
model.load_weights(filepath='Keras_models/new_model')
if config_parameter.mode == "V2I":
    antnna_size = config_parameter.antenna_size
    num_vehicle = config_parameter.num_vehicle
elif config_parameter.mode == "V2V":
    antenna_size = config_parameter.vehicle_antenna_size
    num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar + config_parameter.num_horizoncar

input_whole = loss_all.generate_random_sample()
for epo in range(10):
    print(input_whole.shape)
    input_single = input_whole[epo:epo + config_parameter.batch_size, :, :]
    input_single = tf.convert_to_tensor(input_single)
    input_single = tf.expand_dims(input_single, axis=0)
    input_single = tf.transpose(input_single,perm=[1,0,2,3])


    output = model(input_single)
    antenna_size_f = tf.cast(antenna_size, tf.float32)
    # dont forget here we are inputing a whole batch
    G = tf.math.sqrt(antenna_size_f)
    Analog_matrix, Digital_matrix = loss.tf_Output2PrecodingMatrix(Output=output)
    print("Analog_matrix",Analog_matrix)
    print("Digital_matrix",Digital_matrix)
    precoding_matrix = loss.tf_Precoding_matrix_combine(Analog_matrix, Digital_matrix)
    print("beamforming",precoding_matrix)

    print("theta",input_single[0,:,:,8*antenna_size+1])
    print("real distance",input_single[0,:,:,9*antenna_size+1])

    '''
    steering_vector_this = tf.complex(input_single[0,-1,:,0:antenna_size], input_single[0,-1,:,antenna_size:2*antenna_size])
    steering_vector_this = tf.reshape(steering_vector_this, (antenna_size, num_vehicle))
    steering_hermite = tf.transpose(tf.math.conj(steering_vector_this))
    pathloss = loss.tf_Path_loss(input_single[0, -1, :, 0])
    pathloss = tf.expand_dims(pathloss, axis=1)
    pathloss = tf.broadcast_to(pathloss, tf.shape(steering_hermite))
    CSI = tf.multiply(tf.cast(tf.multiply(G, pathloss), dtype=tf.complex128), steering_hermite)


# Example beamforming matrix and channel matrix
    #beamforming_matrix = np.random.randn(8, 4) + 1j * np.random.randn(8, 4)
    #channel_matrix = np.random.randn(4, 6) + 1j * np.random.randn(4, 6)
    
    '''
    #sv = loss.
    idx = np.arange(antenna_size)
    angle_set = np.arange(1, 361) / 180 * np.pi
    Hset = np.exp(-1j * np.pi * idx.reshape(-1, 1) * np.cos(angle_set))

    zf_matrix = tf.complex(input_single[0,-1,:,4*antenna_size:5*antenna_size], input_single[0,-1,:,5*antenna_size:6*antenna_size])
    print(input_single[0,-1,:,0])
    print(Hset.shape)
    print("zf_matrix",zf_matrix.numpy())

    precoding_matrix = precoding_matrix.numpy()
    #x = np.exp(1j * np.pi * idx * np.cos((30/180)*np.pi)).T.reshape(8,1)
    #print(x.shape)
    #x_hermite=np.transpose(np.conj(x))
    #precoding_matrix =translate_precoding_matrix(precoding_matrix)
    precoding_matrix_hermite = np.transpose(np.conj(precoding_matrix))
    Hset_hermite = np.transpose(np.conj(Hset))

    print("precoding",precoding_matrix)
    #r2 = np.dot(precoding_matrix_hermite, Hset_hermite.T)
    #r1 = np.dot(Hset.T, zf_matrix.numpy().T)
    r1 = np.dot(Hset.T, precoding_matrix[4])
    print("r90", r1[28:35])

    #r = np.dot(r1,r2)
    plt.polar(angle_set, np.abs(r1))
    plt.show()
    #r0 = np.dot(Hset.T, precoding_matrix[:,0])
    #print("r90",r1[28:35])
    #plt.polar(angle_set, np.abs(r0))
    #plt.show()
    # Compute the beam pattern
    '''
    beam_pattern = np.abs(np.matmul(precoding_matrix, CSI))
    # Compute the average magnitude for each antenna
    average_magnitude = np.mean(beam_pattern, axis=1)
    theta = np.linspace(0, 2 * np.pi, len(average_magnitude), endpoint=False)
    average_magnitude = np.concatenate((average_magnitude, [average_magnitude[0]]))  # Add the first element to the end
    theta = np.concatenate((theta, [theta[0]]))


    ax = plt.subplot(111, polar=True)
    ax.plot(theta, average_magnitude)
    ax.set_title('Antenna Pattern')
    ax.text(0, np.max(average_magnitude) * 1.1, 'Distance: {},Angel:{}'.format(input_single[0,-1,:,2*antenna_size],input_single[0,-1,:,3*antenna_size]), ha='center')

    plt.show()
    '''