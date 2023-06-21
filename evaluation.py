import loss
import numpy as np
import config_parameter
import math
import tensorflow as tf
from network import DL_method_NN_for_v2x_mod,DL_method_NN_for_v2x_hybrid

Test = "V2V"
Test = "V2I"
if config_parameter.mode == "V2I":
    antenna_size = config_parameter.antenna_size
    num_vehicle = config_parameter.num_vehicle
elif config_parameter.mode == "V2V":
    antenna_size = config_parameter.vehicle_antenna_size
    num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar + config_parameter.num_horizoncar


def load_model_hybrid():
    #model = DL_method_NN_for_v2x_mod()
    model =DL_method_NN_for_v2x_hybrid()
    #model = ResNet()
    #model = ResNetLSTMModel()
    num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar +config_parameter.num_horizoncar

    model.build(input_shape=(config_parameter.batch_size, num_vehicle,40,1))

    model.summary()

    model.load_weights(filepath='Keras_models_hybrid/new_model')
    return model
def load_model_digital():
    model = DL_method_NN_for_v2x_mod()
    #model =DL_method_NN_for_v2x_hybrid()
    #model = ResNet()
    #model = ResNetLSTMModel()
    num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar +config_parameter.num_horizoncar

    model.build(input_shape=(config_parameter.batch_size, num_vehicle,40,1))

    model.summary()

    model.load_weights(filepath='Keras_models_digital/new_model')
    return model
def load_model_only_communication_digital():
    model = DL_method_NN_for_v2x_mod()
    #model =DL_method_NN_for_v2x_hybrid()
    #model = ResNet()
    #model = ResNetLSTMModel()
    num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar +config_parameter.num_horizoncar

    model.build(input_shape=(config_parameter.batch_size, num_vehicle,40,1))

    model.summary()

    model.load_weights(filepath='Keras_models_only_communication_digital/new_model')
    return model
def load_model_only_communication_hybrid():
    #model = DL_method_NN_for_v2x_mod()
    model =DL_method_NN_for_v2x_hybrid()
    #model = ResNet()
    #model = ResNetLSTMModel()
    num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar +config_parameter.num_horizoncar

    model.build(input_shape=(config_parameter.batch_size, num_vehicle,40,1))

    model.summary()

    model.load_weights(filepath='Keras_models_only_communication_hybrid/new_model')
    return model
def generate_input_v2v():
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
    # speed_horizon_car_dictionary = np.zeros(shape=(
    #   config_parameter.num_horizoncar,
    #  int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot)))
    # real_location_horizoncar_x = np.zeros(shape=(
    #   config_parameter.num_horizoncar,
    #  int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot)))
    # real location at every timepoint
    # this one doesn't include the initial location
    # real angle at every timepoint in this iter
    totalnum_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar + config_parameter.num_horizoncar
    real_theta = np.zeros(shape=(
        totalnum_vehicle, int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot)))

    speed_own_dictionary = np.random.uniform(config_parameter.horizonspeed_low,
                                             config_parameter.horizonspeed_high,
                                             size=speed_own_dictionary.shape)
    # speed_horizon_car_dictionary = np.random.uniform(config_parameter.horizonspeed_low,
    #                                                config_parameter.horizonspeed_high,
    #                                               size=speed_horizon_car_dictionary.shape)
    speed_lower_car_dictionary = np.random.uniform(config_parameter.lowerspeed_low,
                                                   config_parameter.lowerspeed_high,
                                                   size=speed_lower_car_dictionary.shape)
    speed_upper_car_dictionary = np.random.uniform(config_parameter.upperspeed_low,
                                                   config_parameter.upperspeed_high,
                                                   size=speed_upper_car_dictionary.shape)
    # speed_whole_dictionary = np.vstack((speed_own_dictionary, speed_lower_car_dictionary,
    #                                   speed_upper_car_dictionary, speed_horizon_car_dictionary))
    speed_whole_dictionary = np.vstack((speed_own_dictionary, speed_lower_car_dictionary,
                                        speed_upper_car_dictionary))
    print(speed_whole_dictionary.shape)
    num_whole = totalnum_vehicle + 1  # 1 is the observer
    initial_car_x = np.zeros(shape=(num_whole, 1))
    initial_car_x[0, 0] = 0
    location_car_y = np.zeros(
        shape=(num_whole, int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot) + 1))
    location_car_y[0, :] = np.full(
        (1, int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot) + 1), 0)

    # initial_car_x[1, 0] = np.random.uniform(config_parameter.Initial_horizoncar1_min,
    #                                       config_parameter.Initial_horizoncar1_max)
    # location_car_y[1, :] = np.full(
    #   (1, int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot) + 1), 0)
    initial_car_x[1, 0] = np.random.uniform(config_parameter.Initial_lowercar1_min,
                                            config_parameter.Initial_lowercar1_max)
    location_car_y[1, :] = np.full(
        (1, int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot) + 1), -50)
    initial_car_x[2, 0] = np.random.uniform(config_parameter.Initial_uppercar1_min,
                                            config_parameter.Initial_uppercar1_max)
    location_car_y[2, :] = np.full(
        (1, int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot) + 1), 50)
    # initial_car_x[4,0]=np.random.uniform(config_parameter.Initial_lowercar2_min, config_parameter.Initial_lowercar2_max)
    # initial_car_x[5, 0] = np.random.uniform(config_parameter.Initial_uppercar2_min,
    # config_parameter.Initial_uppercar2_max)

    location_car_x = np.zeros(shape=(
        num_whole, int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot) + 1))
    coordinates_car = np.zeros(shape=(
        num_whole, int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot) + 1))
    location_car_x[:, 0] = initial_car_x[:, :].reshape((num_vehicle + 1,))
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
    return real_distance,real_theta
def generate_input_v2i():

def eva_v2v():
    real_distance, real_theta = generate_input_v2v()
    combined = loss.Conversion2CSI(real_distance, real_theta)
    # this one output the CSI and zf matrix
    zf_matrix = tf.complex(combined[:,:,2*antenna_size:3*antenna_size],combined[:,:,3*antenna_size:4*antenna_size])
    CSI = tf.complex(combined[:,:,0:antenna_size],combined[:,:,antenna_size:2*antenna_size])
    precoding_matrix =




if Test == "V2V":
    eva_v2v()






