import numpy as np
import matplotlib.pyplot as plt
import config_parameter
import loss
import tensorflow as tf
from Trainv2_4inputs import load_model
import math
model = load_model()
model.load_weights(filepath='Keras_models/new_model')
if config_parameter.mode == "V2I":
    antenna_size = config_parameter.antenna_size
    num_vehicle = config_parameter.num_vehicle
elif config_parameter.mode == "V2V":
    antenna_size = config_parameter.vehicle_antenna_size
    num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar + config_parameter.num_horizoncar
speed_own_dictionary = np.zeros(shape=(
    1, int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot)))
speed_upper_car_dictionary = np.zeros(shape=(
    config_parameter.num_uppercar,
    int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot)))
real_location_uppercar_x = np.zeros(shape=(
    config_parameter.num_uppercar,
    int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot)))

speed_lower_car_dictionary = np.zeros(shape=(
    config_parameter.num_lowercar,
    int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot)))
real_location_lowercar_x = np.zeros(shape=(
    config_parameter.num_lowercar,
    int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot)))
speed_horizon_car_dictionary = np.zeros(shape=(
    config_parameter.num_horizoncar,
    int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot)))
real_location_horizoncar_x = np.zeros(shape=(
    config_parameter.num_horizoncar,
    int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot)))
# real location at every timepoint
# this one doesn't include the initial location
# real angle at every timepoint in this iter
totalnum_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar + config_parameter.num_horizoncar
real_theta = np.zeros(shape=(
    totalnum_vehicle, int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot)))

speed_own_dictionary = np.random.uniform(config_parameter.horizonspeed_low,
                                         config_parameter.horizonspeed_high,
                                         size=speed_own_dictionary.shape)
speed_horizon_car_dictionary = np.random.uniform(config_parameter.horizonspeed_low,
                                                 config_parameter.horizonspeed_high,
                                                 size=speed_horizon_car_dictionary.shape)
speed_lower_car_dictionary = np.random.uniform(config_parameter.lowerspeed_low,
                                               config_parameter.lowerspeed_high,
                                               size=speed_lower_car_dictionary.shape)
speed_upper_car_dictionary = np.random.uniform(config_parameter.upperspeed_low,
                                               config_parameter.upperspeed_high,
                                               size=speed_upper_car_dictionary.shape)
speed_whole_dictionary = np.vstack((speed_own_dictionary, speed_lower_car_dictionary,
                                    speed_upper_car_dictionary, speed_horizon_car_dictionary))
print(speed_whole_dictionary.shape)
num_whole = totalnum_vehicle + 1  # 1 is the observer
initial_car_x = np.zeros(shape=(num_whole, 1))
initial_car_x[0, 0] = 0
location_car_y = np.zeros(
    shape=(num_whole, int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot) + 1))
location_car_y[0, :] = np.full(
    (1, int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot) + 1), 0)

initial_car_x[1, 0] = np.random.uniform(config_parameter.Initial_horizoncar1_min,
                                        config_parameter.Initial_horizoncar1_max)
location_car_y[1, :] = np.full(
    (1, int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot) + 1), 0)
initial_car_x[2, 0] = np.random.uniform(config_parameter.Initial_lowercar1_min,
                                        config_parameter.Initial_lowercar1_max)
location_car_y[2, :] = np.full(
    (1, int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot) + 1), -20)
initial_car_x[3, 0] = np.random.uniform(config_parameter.Initial_uppercar1_min,
                                        config_parameter.Initial_uppercar1_max)
location_car_y[3, :] = np.full(
    (1, int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot) + 1), 20)
# initial_car_x[4,0]=np.random.uniform(config_parameter.Initial_lowercar2_min, config_parameter.Initial_lowercar2_max)
# initial_car_x[5, 0] = np.random.uniform(config_parameter.Initial_uppercar2_min,
# config_parameter.Initial_uppercar2_max)

location_car_x = np.zeros(shape=(
    num_whole, int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot) + 1))
coordinates_car = np.zeros(shape=(
    num_whole, int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot) + 1))
location_car_x[:, 0] = initial_car_x[:, :].reshape((4,))
real_distance = np.zeros(shape=(
    totalnum_vehicle, int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot) + 1))
real_theta = np.zeros(shape=(
    totalnum_vehicle, int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot) + 1))
for v in range(num_whole):
    for t in range(int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot)):
        location_car_x[v, t + 1] = location_car_x[v, t] + speed_whole_dictionary[
            v, t] * config_parameter.Radar_measure_slot

for v in range(1, num_whole):
    for t in range(int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot) + 1):
        real_distance[v - 1, t] = math.sqrt((location_car_x[0, t] - location_car_x[v, t]) ** 2 + (
                location_car_y[0, t] - location_car_y[v, t]) ** 2)
        real_theta[v - 1, t] = math.atan2(location_car_y[v, t] - location_car_y[0, t],
                                          location_car_x[v, t] - location_car_x[0, t])
        if real_theta[v - 1, t] == 0:
            real_theta[v - 1, t] == 0.1
        print(real_theta[v - 1, t])
steering_vector_whole = np.zeros(shape=(
int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot) + 1, totalnum_vehicle, antenna_size),
                                 dtype=complex)
for v in range(0, totalnum_vehicle):
    for t in range(0, int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot) + 1):
        print("ssssssssssssssssssssssss", real_distance[v, t], real_theta[v, t])
        steering_vector_whole[t, v] = loss.calculate_steer_vector_this(real_theta[v, t])
        input_whole = np.zeros(shape=(int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot) + 1, totalnum_vehicle,
           4 * antenna_size))
        input_whole[:, :, 0:antenna_size] = np.real(steering_vector_whole)
        input_whole[:, :, antenna_size:2 * antenna_size] = np.imag(steering_vector_whole)
for i in range(0, antenna_size):
    input_whole[:, :, 2 * antenna_size + i] = real_theta.T
    input_whole[:, :, 3 * antenna_size + i] = real_distance.T
for epo in range(int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot) + 1 - 9):
    print(input_whole.shape)
    input_single = input_whole[epo:epo + 10, :, :]
    input_single = tf.convert_to_tensor(input_single)
    input_single = tf.expand_dims(input_single, axis=0)
    output = model(input_single)
    antenna_size_f = tf.cast(antenna_size, tf.float32)
    # dont forget here we are inputing a whole batch
    G = tf.math.sqrt(antenna_size_f)
    Analog_matrix, Digital_matrix = loss.tf_Output2PrecodingMatrix(Output=output)
    precoding_matrix = loss.tf_Precoding_matrix_combine(Analog_matrix, Digital_matrix)
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

    # Compute the beam pattern
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