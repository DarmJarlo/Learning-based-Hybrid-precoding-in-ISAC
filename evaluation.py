import loss
import numpy as np
import config_parameter
import math
Test = "V2V"
def generate_input():
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
    steering_vector_whole = np.zeros(shape=(int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot) + 1, totalnum_vehicle, antenna_size),
                                     dtype=complex)

    for v in range(0, totalnum_vehicle):
        for t in range(0, int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot) + 1):
            print("ssssssssssssssssssssssss", real_distance[v, t], real_theta[v, t])
            steering_vector_whole[t, v] = loss.calculate_steer_vector_this(real_theta[v, t])
    print("steering_vector_whole", steering_vector_whole)
    input_whole = np.zeros(
        shape=(int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot) + 1, totalnum_vehicle,
               4 * antenna_size))
    input_whole[:, :, 0:antenna_size] = np.real(steering_vector_whole)
    input_whole[:, :, antenna_size:2 * antenna_size] = np.imag(steering_vector_whole)
    speed_dictionary=np.zeros(shape=(
        num_vehicle, int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot) + 1))
    speed_dictionary[:,:300]=speed_whole_dictionary[1:,]
    for i in range(0, antenna_size):
        input_whole[:, :, 2 * antenna_size + i] = real_theta.T
        input_whole[:, :, 3 * antenna_size + i] = real_distance.T
        #input_whole[:,:,4*antenna_size+i]=speed_dictionary.T
    with open('angleanddistance', "w") as file:
        file.write("real_theta")
        file.write(str(real_theta) + "\n")
        file.write("real_distance")
        file.write(str(real_distance) + "\n")
    return input_whole
def eva_v2v():

if Test == "V2V":
    eva_v2v()






