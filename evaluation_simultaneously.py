import random

import loss
import numpy as np
import config_parameter
import math
import tensorflow as tf
from network import DL_method_NN_for_v2x_mod,DL_method_NN_for_v2x_hybrid
import matplotlib.pyplot as plt
#Test = "V2V"
Test = "V2I"
if config_parameter.mode == "V2I":
    antenna_size = config_parameter.antenna_size
    num_vehicle = config_parameter.num_vehicle
elif config_parameter.mode == "V2V":
    antenna_size = config_parameter.vehicle_antenna_size
    num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar + config_parameter.num_horizoncar
def load_model(digital):
    if digital==True:
        model = DL_method_NN_for_v2x_mod()
    else:
        model = DL_method_NN_for_v2x_hybrid()

    num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar +config_parameter.num_horizoncar

    model.build(input_shape=(None, num_vehicle,144,1))

    model.summary()
    return model
def load_model_hybrid_combine():
    digital = False
    model = load_model(digital)

    model.load_weights(filepath='allmodel1/Keras_models_hybrid_combinefinal/new_model')
    return model
def load_model_digitalwith_combine():
    digital = True
    model = load_model(digital)

    model.load_weights(filepath='allmodel1/Keras_models_digitalwith_combinefinal/new_model')
    return model
def load_model_digitalwithout_combine():
    digital = True
    model = load_model(digital)
    model.load_weights(filepath='allmodel1/Keras_models_digitalwithout_combinefinal/new_model')
    return model

def load_model_only_communication_digitalwith():
    digital = True
    model = load_model(digital)
    model.load_weights(filepath='allmodel1/Keras_models_digitalwith_sumratefinal/new_model')
    return model
def load_model_only_communication_digitalwithout():
    digital = True
    model = load_model(digital)
    model.load_weights(filepath='allmodel1/Keras_models_digitalwithout_sumratefinal/new_model')
    #model.load_weights(filepath='allmodel1/Keras_models_test/new_model')
    return model
def load_model_only_communication_hybrid():
    digital = False
    model = load_model(digital)
    model.load_weights(filepath='allmodel1/Keras_models_hybrid_onlycommfinal/new_model')
    return model
def load_model_only_crbd_hybrid():
    digital = False
    model = load_model(digital)
    #model.load_weights(filepath='allmodel1/Keras_models_hybrid_onlycrbdfinal/new_model')
    model.load_weights(filepath='allmodel1/Keras_models_hybrid_onlycommfinal/new_model')
    return model
def load_model_only_crbd_digitalwith():
    digital = True
    model = load_model(digital)
    #model.load_weights(filepath='allmodel1/Keras_models_digitalwith_crbdfinal/new_model')
    model.load_weights(filepath='allmodel1/Keras_models_digitalwith_sumratefinal/new_model')
    return model
def load_model_only_crbd_digitalwithout():
    digital = True
    model = load_model(digital)
    #model.load_weights(filepath='allmodel1/Keras_models_digitalwithout_crbdfinal/new_model')
    model.load_weights(filepath='allmodel1/Keras_models_digitalwithout_sumratefinal/new_model')
    return model
def load_model_only_crbangle_hybrid():
    digital = False
    model = load_model(digital)
    model.load_weights(filepath='allmodel1/Keras_models_hybrid_onlycrbanglefinal/new_model')
    return model
def load_model_only_crbangle_digitalwith():
    digital = True
    model = load_model(digital)
    model.load_weights(filepath='allmodel1/Keras_models_digitalwith_crbanglefinal/new_model')
    return model
def load_model_only_crbangle_digitalwithout():
    digital = True
    model = load_model(digital)
    model.load_weights(filepath='allmodel1/Keras_models_digitalwithout_crbafinal/new_model')
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
    totalnum_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar + config_parameter.num_horizoncar

    speed_own_dictionary = np.random.uniform(config_parameter.horizonspeed_low,
                                             config_parameter.horizonspeed_high,
                                             size=speed_own_dictionary.shape)
    speed_lower_car_dictionary = np.random.uniform(config_parameter.lowerspeed_low,
                                                   config_parameter.lowerspeed_high,
                                                   size=speed_lower_car_dictionary.shape)
    speed_upper_car_dictionary = np.random.uniform(config_parameter.upperspeed_low,
                                                   config_parameter.upperspeed_high,
                                                   size=speed_upper_car_dictionary.shape)
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
    initial_car_x[1, 0] = np.random.uniform(config_parameter.Initial_location_min[0],
                                            config_parameter.Initial_location_max[0])
    location_car_y[1, :] = np.full(
        (1, int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot) + 1), \
        config_parameter.Initial_lowercar_y)
    initial_car_x[2, 0] = np.random.uniform(config_parameter.Initial_location_min[1],
                                            config_parameter.Initial_location_max[1])
    location_car_y[2, :] = np.full(
        (1, int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot) + 1), \
        config_parameter.Initial_lowercar_y)
    initial_car_x[3, 0] = np.random.uniform(config_parameter.Initial_location_min[2],
                                            config_parameter.Initial_location_max[2])
    location_car_y[3, :] = np.full(
        (1, int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot) + 1), \
        config_parameter.Initial_uppercar_y)
    initial_car_x[4, 0] = np.random.uniform(config_parameter.Initial_location_min[3],
                                            config_parameter.Initial_location_max[3])
    location_car_y[4, :] = np.full(
        (1, int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot) + 1), \
        config_parameter.Initial_uppercar_y)
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
            if real_theta[v - 1, t] < 0:
                real_theta[v - 1, t] = real_theta[v - 1, t] + 2 * math.pi
            print(real_theta[v - 1, t])
    return real_distance,real_theta
def generate_input_v2i():

    initial_location_x = np.zeros(shape=(config_parameter.num_vehicle, 1))
 # speed at every timepoint
    speed_dictionary = np.zeros(shape=(
    config_parameter.num_vehicle, int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot)))
    # real location at every timepoint
    real_location_x = np.zeros(shape=(config_parameter.num_vehicle,
                                      int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot)))  # this one doesn't include the initial location

    real_theta = np.zeros(shape=(
    config_parameter.num_vehicle, int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot)))

    real_distance_list = np.zeros(shape=(
        config_parameter.num_vehicle, int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot)))
    # real x coordinates of target including the initial location
    location = np.zeros(shape=(config_parameter.num_vehicle, (
                int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot) + 1)))  # this one include the initial location
    # real y coordinates of target including the initial location
    location_y = np.zeros(shape=(config_parameter.num_vehicle, (int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot) + 1)))
    # target coordinates combined
    target_coordinates = np.zeros(shape=(config_parameter.num_vehicle, int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot)))
    Initial_location_min = config_parameter.Initial_location_min
    Initial_location_max = config_parameter.Initial_location_max
    for vehicle in range(0, config_parameter.num_vehicle):
        # initial_location_x[vehicle] = []
        speed_dictionary[vehicle] = [np.random.uniform(low=config_parameter.speed_low, high=config_parameter.speed_high) \
                                     for _ in
                                     range(int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot))]
        print(speed_dictionary.shape)
        # initialize location for every car [0,100)
        random_location = int(np.random.uniform(Initial_location_min[vehicle], Initial_location_max[vehicle]))

        print(random_location)

        initial_location_x[vehicle] = random_location
        location[vehicle, 0] = random_location
        # location_y[vehicle] = [0]
        for i in range(int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot)):
            location[vehicle, i + 1] = config_parameter.Radar_measure_slot * speed_dictionary[vehicle, i] + \
                                       location[vehicle, i]

            # location_y[vehicle].append(0)
            real_theta[vehicle, i] = math.atan2(location_y[vehicle, i] - config_parameter.RSU_location[1],location[vehicle, i] - config_parameter.RSU_location[0])
            if real_theta[vehicle,i] < 0:
                real_theta[vehicle, i] = real_theta[vehicle, i] + 2 * math.pi
            #target_coordinates[vehicle, i] = (location[vehicle, i + 1], location_y[vehicle, i + 1])
            real_distance_list[vehicle, i] = math.sqrt((location[vehicle, i] - config_parameter.RSU_location[0]) ** 2 \
                                                       + (location_y[vehicle, i] - config_parameter.RSU_location[1]) ** 2)


        real_location_x[vehicle] = location[vehicle][1:]
    # for vehicle in range(0,config_parameter.num_vehicle):
    #   for time in range(0,int(config_parameter.one_iter_period/config_parameter.Radar_measure_slot)):
    #      sigma_time_delay[v, time] = loss.Sigma_time_delay_square(index=v,distance_list = real_distance[:, time],estimated_theta_list=real_theta[:,time],\
    #                                                            precoding_matrix=precoding_matrix)
    #     sigma_doppler[v, time] = loss.Sigma_time_delay_square(index=vdistance_list = real_distance[:, time],estimated_theta_list=real_theta[:,time].\
    #        precoding_matrix=precoding_matrix)
    print("location while start measuring", real_location_x)

    print("initial location", initial_location_x)
    return real_distance_list,real_theta
def comparison_between_crb_distance(combined):
    #CSI = tf.complex(combined[:,:,0:antenna_size],combined[:,:,antenna_size:2*antenna_size])
    CSI = tf.complex(combined[:, :, 6 * antenna_size:7 * antenna_size, 0],
                     combined[:, :, 7 * antenna_size:8 * antenna_size, 0])
    steering_vector_this_o = tf.complex(combined[:, :, 0:antenna_size, 0],
                                        combined[:, :, antenna_size:2 * antenna_size, 0])
    zf_matrix = tf.complex(combined[:, :, 4 * antenna_size:5 * antenna_size, 0],
                            combined[:, :, 5 * antenna_size:6 * antenna_size, 0])
    distance = combined[:, :, 3 * antenna_size:4*antenna_size, 0]
    beta = loss.Reflection_coefficient(distance)
    model1 = load_model_hybrid_combine()
    model2 = load_model_digitalwith_combine()
    model3 = load_model_digitalwithout_combine()
    model4 = load_model_only_crbd_hybrid()
    model5 = load_model_only_crbd_digitalwith()
    model6 = load_model_only_crbd_digitalwithout()
    output1 = model1(combined)
    output2 = model2(combined)
    output3 = model3(combined)
    output4 = model4(combined)
    output5 = model5(combined)
    output6 = model6(combined)
    analog1, digital1 = loss.tf_Output2PrecodingMatrix_rad(output1)
    precoder1 = loss.tf_Precoding_matrix_combine(analog1, digital1)
    precoder2 = loss.tf_Output2digitalPrecoding(output2, zf_matrix=zf_matrix, distance=None)
    precoder3 = loss.tf_Output2digitalPrecoding(output3, zf_matrix=None, distance=None)
    analog4, digital4 = loss.tf_Output2PrecodingMatrix_rad(output4)
    precoder4 = loss.tf_Precoding_matrix_combine(analog4, digital4)
    precoder5 = loss.tf_Output2digitalPrecoding(output5, zf_matrix=zf_matrix, distance=None)
    precoder6 = loss.tf_Output2digitalPrecoding(output6, zf_matrix=None, distance=None)
    precoder7 = loss.random_beamforming()
    steering_vector_this_o = steering_vector_this_o
    Sigma_time_delay1 = loss.tf_sigma_delay_square(steering_vector_this_o, precoder1, beta)
    Sigma_time_delay2 = loss.tf_sigma_delay_square(steering_vector_this_o, precoder2, beta)
    Sigma_time_delay3 = loss.tf_sigma_delay_square(steering_vector_this_o, precoder3, beta)
    Sigma_time_delay4 = loss.tf_sigma_delay_square(steering_vector_this_o, precoder4, beta)
    Sigma_time_delay5 = loss.tf_sigma_delay_square(steering_vector_this_o, precoder5, beta)
    Sigma_time_delay6 = loss.tf_sigma_delay_square(steering_vector_this_o, precoder6, beta)
    Sigma_time_delay7 = loss.tf_sigma_delay_square(steering_vector_this_o, tf.transpose(zf_matrix,perm=[0,2,1]), beta)
    CRB_d1 = tf.reduce_sum(loss.tf_CRB_distance(Sigma_time_delay1),axis=1)/4
    CRB_d2 = tf.reduce_sum(loss.tf_CRB_distance(Sigma_time_delay2),axis=1)/4
    CRB_d3 = tf.reduce_sum(loss.tf_CRB_distance(Sigma_time_delay3),axis=1)/4
    CRB_d4 = tf.reduce_sum(loss.tf_CRB_distance(Sigma_time_delay4),axis=1)/4
    CRB_d5 = tf.reduce_sum(loss.tf_CRB_distance(Sigma_time_delay5),axis=1)/4
    CRB_d6 = tf.reduce_sum(loss.tf_CRB_distance(Sigma_time_delay6),axis=1)/4
    CRB_d7 = tf.reduce_sum(loss.tf_CRB_distance(Sigma_time_delay7),axis=1)/4
    #shape = tf.shape(CRB_d1)[0]
    #CRB_d7 = loss.tf_CRB_distance(Sigma_time_delay7)
    fig, ax1 = plt.subplots()
    ax1.plot(range(40), CRB_d1, 'b-.', label='hybrid ISAC')

    ax1.plot(range(40), CRB_d2, 'g-', label='digital ISAC with initialization')
    ax1.plot(range(40), CRB_d3, 'r.-', label='digital ISAC,without initialization')
    ax1.plot(range(40), CRB_d4, 'm.', label='hybrid only crb_distance')
    ax1.plot(range(40), CRB_d5, 'r-', label='digital with initialization only CRB_d')
    ax1.plot(range(40), CRB_d6, 'b-', label='digital without initialization only CRB_d')
    ax1.plot(range(40), CRB_d7, 'g.', label='ZF precoder')
    ax1.set_xlabel('time')
    ax1.set_ylabel('CRB distance', color='b')
    ax1.tick_params('y', colors='b')

    #plot.legend(loc='best')
    fig.tight_layout()

    # 添加图例
    lines = [ax1.get_lines()[0], \
             ax1.get_lines()[1], \
            ax1.get_lines()[2], \
             ax1.get_lines()[3], \
             ax1.get_lines()[4],ax1.get_lines()[5],ax1.get_lines()[6]]
    #lines = [ax1.get_lines()[0], \
     #        ax1.get_lines()[1], \
      #       ax1.get_lines()[2]]

    # lines = [ax4.get_lines()[0]]
    labels = [line.get_label() for line in lines]
    #ax1.legend(lines, labels, bbox_to_anchor=(0.5, 0.4),loc='best')
    ax1.legend(lines, labels, loc='best')
    ax1.legend_.get_frame().set_facecolor('white')  # 设置标签框的背景颜色为白色
    ax1.legend_.get_frame().set_linewidth(0.5)  # 设置标签框的边框宽度
    for text in ax1.legend_.get_texts():
        text.set_fontsize(10)

        #ax1.legend(lines, labels, loc='best')
    plt.show()


def comparison_between_crb_angle(combined):
    CSI = tf.complex(combined[:, :, 6 * antenna_size:7 * antenna_size, 0],
                     combined[:, :, 7 * antenna_size:8 * antenna_size, 0])

    steering_vector_this_o = tf.complex(combined[:, :, 0:antenna_size, 0],
                                        combined[:, :, antenna_size:2 * antenna_size, 0])
    zf_matrix = tf.complex(combined[:, :, 4 * antenna_size:5 * antenna_size, 0],
                            combined[:, :, 5 * antenna_size:6 * antenna_size, 0])
    distance = combined[:, :, 3 * antenna_size:4*antenna_size, 0]
    theta = combined[:, :, 2 * antenna_size, 0]
    beta = loss.Reflection_coefficient(distance)
    model1 = load_model_hybrid_combine()
    model2 = load_model_digitalwith_combine()
    model3 = load_model_digitalwithout_combine()
    model4 = load_model_only_crbangle_hybrid()
    model5 = load_model_only_crbangle_digitalwith()
    model6 = load_model_only_crbangle_digitalwithout()
    output1 = model1(combined)
    output2 = model2(combined)
    output3 = model3(combined)
    output4 = model4(combined)
    output5 = model5(combined)
    output6 = model6(combined)
    analog1,digital1 = loss.tf_Output2PrecodingMatrix_rad(output1)
    precoder1 = loss.tf_Precoding_matrix_combine(analog1,digital1)
    precoder2 = loss.tf_Output2digitalPrecoding(output2,zf_matrix=zf_matrix,distance=None)
    precoder3 = loss.tf_Output2digitalPrecoding(output3, zf_matrix=None,distance=None)
    analog4,digital4 = loss.tf_Output2PrecodingMatrix_rad(output4)
    precoder4 = loss.tf_Precoding_matrix_combine(analog4,digital4)
    precoder5 = loss.tf_Output2digitalPrecoding(output5,zf_matrix=zf_matrix,distance=None)
    precoder6 = loss.tf_Output2digitalPrecoding(output6,zf_matrix=None,distance=None)
    precoder7 = loss.random_beamforming()
    CRB_angle1 = tf.reduce_sum(loss.tf_CRB_angle(beta, precoder1, theta),axis=1)/4
    CRB_angle2 = tf.reduce_sum(loss.tf_CRB_angle(beta, precoder2, theta),axis=1)/4
    CRB_angle3 = tf.reduce_sum(loss.tf_CRB_angle(beta, precoder3, theta),axis=1)/4
    CRB_angle4 = tf.reduce_sum(loss.tf_CRB_angle(beta, precoder4, theta),axis=1)/4
    CRB_angle5 = tf.reduce_sum(loss.tf_CRB_angle(beta, precoder5, theta),axis=1)/4
    CRB_angle6 = tf.reduce_sum(loss.tf_CRB_angle(beta, precoder6, theta),axis=1)/4
    CRB_angle7 = tf.reduce_sum(loss.tf_CRB_angle(beta, tf.transpose(zf_matrix,perm=[0,2,1]), theta),axis=1)/4
    fig, ax1 = plt.subplots()
    ax1.plot(range(40), CRB_angle1, 'b-.', label='hybrid ISAC')

    ax1.plot(range(40), CRB_angle2, 'g-', label='digital ISAC,with initial point')
    ax1.plot(range(40), CRB_angle3, 'r.-', label='digital ISAC,without initial point')
    ax1.plot(range(40), CRB_angle4, 'ms', label='hybrid only crb_angle')
    ax1.plot(range(40), CRB_angle5, 'r-', label='digital with initial point only crb_angle')
    ax1.plot(range(40), CRB_angle6, 'b-', label='digital without initial point only crb_angle')
    ax1.plot(range(40), CRB_angle7, 'g.', label='ZF precoder')
    ax1.set_xlabel('time')
    ax1.set_ylabel('CRB angle', color='b')
    ax1.tick_params('y', colors='b')

    # plot.legend(loc='best')
    fig.tight_layout()

    # 添加图例
    lines = [ax1.get_lines()[0], \
             ax1.get_lines()[1], \
             ax1.get_lines()[2], \
             ax1.get_lines()[3], \
             ax1.get_lines()[4], ax1.get_lines()[5],ax1.get_lines()[6]]
    # lines = [ax1.get_lines()[0], \
    #        ax1.get_lines()[1], \
    #       ax1.get_lines()[2]]

    # lines = [ax4.get_lines()[0]]
    labels = [line.get_label() for line in lines]
    #ax1.legend(lines, labels, bbox_to_anchor=(0.5, 0.45),loc='best')
    ax1.legend(lines, labels, loc='best')
    ax1.legend_.get_frame().set_facecolor('white')  # 设置标签框的背景颜色为白色
    ax1.legend_.get_frame().set_linewidth(0.4)  # 设置标签框的边框宽度
    for text in ax1.legend_.get_texts():
        text.set_fontsize(10)
    #ax1.legend(lines, labels, loc='best')
    plt.show()
    idx = np.arange(antenna_size)
    angle_set = np.arange(1, 181) / 180 * np.pi
    Hset = np.exp(1j * np.pi * idx.reshape(-1, 1) * np.cos(angle_set))

    precoder5 = precoder5.numpy()
    print(precoder5.shape)
    r1 = np.matmul(Hset.T, precoder4[6, :, :])
    plt.polar(angle_set, np.abs(r1))
    #plt.show()



def comparison_between_sumrate(combined):
   #CSI = tf.complex(combined[:,:,0:antenna_size],combined[:,:,antenna_size:2*antenna_size])
    CSI = tf.complex(combined[:, :, 6 * antenna_size:7 * antenna_size, 0],
                     combined[:, :, 7 * antenna_size:8 * antenna_size, 0])




    zf_matrix = tf.complex(combined[:, :, 4 * antenna_size:5 * antenna_size, 0],
                       combined[:, :, 5 * antenna_size:6 * antenna_size, 0])
    model1 = load_model_hybrid_combine()
    model2 = load_model_digitalwith_combine()
    model3 = load_model_digitalwithout_combine()
    model4 = load_model_only_communication_hybrid()
    model5 = load_model_only_communication_digitalwith()
    model6 = load_model_only_communication_digitalwithout()
    output1 = model1(combined)
    output2 = model2(combined)
    output3 = model3(combined)
    output4 = model4(combined)
    output5 = model5(combined)
    output6 = model6(combined)
    analog1,digital1 = loss.tf_Output2PrecodingMatrix_rad(output1)
    precoder1 = loss.tf_Precoding_matrix_combine(analog1,digital1)
    precoder2 = loss.tf_Output2digitalPrecoding(output2,zf_matrix=zf_matrix,distance=None)
    precoder3 = loss.tf_Output2digitalPrecoding(output3, zf_matrix=None,distance=None)
    analog4,digital4 = loss.tf_Output2PrecodingMatrix_rad(output4)
    precoder4 = loss.tf_Precoding_matrix_combine(analog4,digital4)
    precoder5 = loss.tf_Output2digitalPrecoding(output5,zf_matrix=zf_matrix,distance=None)
    precoder6 = loss.tf_Output2digitalPrecoding(output6,zf_matrix=None,distance=None)
    precoder7 = loss.random_beamforming()
    sum_rate1,sinr = loss.tf_loss_sumrate(CSI,precoder1)
    sum_rate2,sinr = loss.tf_loss_sumrate(CSI,precoder2)
    sum_rate3,sinr = loss.tf_loss_sumrate(CSI,precoder3)
    sum_rate4,sinr = loss.tf_loss_sumrate(CSI,precoder4)
    sum_rate5,sinr = loss.tf_loss_sumrate(CSI,precoder5)
    sum_rate6,sinr = loss.tf_loss_sumrate(CSI,precoder6)
    sum_rate7,sinr = loss.tf_loss_sumrate(CSI,tf.transpose(zf_matrix,perm=[0,2,1]))
    #sum_rate8,sinr = loss.tf_loss_sumrate(CSI,precoder7)



    fig, ax1 = plt.subplots()
    ax1.plot(range(40), sum_rate1, 'b-.', label='hybrid ISAC')
    #ax2 = ax1.twinx()
    ax1.plot(range(40), sum_rate2, 'r--', label='digital ISAC,with initial point')
    ax1.plot(range(40), sum_rate3, 'r.-', label='digital ISAC,without initial point')
    ax1.plot(range(40), sum_rate4, 'ms', label='hybrid only communication')
    ax1.plot(range(40), sum_rate5, 'b-', label='digital with initial point only communication')
    ax1.plot(range(40), sum_rate6, 'r-', label='digital without initial point only communication')
    ax1.plot(range(40), sum_rate7, 'g.', label='ZF precoder')
    ax1.set_xlabel('time')
    ax1.set_ylabel('sum rate(bits/s/hz)', color='b')
    ax1.tick_params('y', colors='b')

    #plot.legend(loc='best')
    fig.tight_layout()

    # 添加图例
    lines = [ax1.get_lines()[0], \
             ax1.get_lines()[1], \
             ax1.get_lines()[2], \
             ax1.get_lines()[3], \
             ax1.get_lines()[4],ax1.get_lines()[5],ax1.get_lines()[6]]

    # lines = [ax4.get_lines()[0]]
    labels = [line.get_label() for line in lines]



    ax1.legend(lines, labels, bbox_to_anchor=(0.55, 0.65),loc = 'center',ncol=1,fontsize=10)


    #ax1.legend(lines, labels, loc='best', ncol=1, fontsize=10)


    ax1.legend_.get_frame().set_facecolor('white')  # 设置标签框的背景颜色为白色
    ax1.legend_.get_frame().set_linewidth(0.4)  # 设置标签框的边框宽度
    for text in ax1.legend_.get_texts():
        text.set_fontsize(10)
    plt.show()


    idx = np.arange(antenna_size)
    angle_set = np.arange(1, 181) / 180 * np.pi
    Hset = np.exp(1j * np.pi * idx.reshape(-1, 1) * np.cos(angle_set))

    precoder5 = precoder5.numpy()
    print(precoder5.shape)
    r1 = np.matmul(Hset.T, tf.transpose(zf_matrix[30, :, :],perm=[1,0]).numpy())
    r1 = np.matmul(Hset.T, precoder3[10, :, :])
    plt.polar(angle_set, np.abs(r1))
    plt.show()

def eva(Test):
    if Test == "V2V":
        real_distance, real_theta = generate_input_v2v()
    elif Test == "V2I":
        real_distance, real_theta = generate_input_v2i()
    #combined = loss.Conversion2CSI(real_distance, real_theta)
    combined = loss.Conversion2input_small(real_theta, real_distance)
    # this one output the CSI and zf matrix
    zf_matrix = tf.complex(combined[:,:,2*antenna_size:3*antenna_size],combined[:,:,3*antenna_size:4*antenna_size])
    #CSI = tf.complex(combined[:,:,0:antenna_size],combined[:,:,antenna_size:2*antenna_size])
    CSI = tf.complex(input[:, :, 6 * antenna_size:7 * antenna_size, 0],
                     input[:, :, 7 * antenna_size:8 * antenna_size, 0])
    model1 = load_model_hybrid()
    model2 = load_model_digital()
    model3 = load_model_only_communication_hybrid()
    model4 = load_model_only_communication_digital()
    output1 = model1(combined)
    output2 = model2(combined)
    output3 = model3(combined)
    output4 = model4(combined)



    # 创建一个图形窗口
    plt.figure()

    # 绘制第一个模型的输出
    plt.plot(range(100), sum_rate1, label='')

    # 绘制第二个模型的输出
    plt.plot(range(100), sum_rate2, label='Model 2')

    # 绘制第三个模型的输出
    plt.plot(range(100), sum_rate3, label='Model 3')

    # 绘制第四个模型的输出
    plt.plot(range(100), sum_rate4, label='Model 4')

    # 添加图例
    plt.legend()

    # 添加标题和轴标签
    plt.title('Sum rate at every timepoint')
    plt.xlabel('time')
    plt.ylabel('sum rate: bits/s/Hz')

    # 显示图形
    plt.show()


random.seed(2)
if Test == "V2V":
    real_distance, real_theta = generate_input_v2v()
elif Test == "V2I":
    real_distance, real_theta = generate_input_v2i()


#combined = loss.Conversion2CSI(real_distance, real_theta)P
combined = loss.Conversion2input_small(real_theta.T[:40], real_distance.T[:40])

combined = tf.expand_dims(combined, axis=3)
comparison_between_sumrate(combined)
#comparison_between_crb_distance(combined)
#comparison_between_crb_angle(combined)
#print("real_distance",real_distance.T.shape)
#print("real_theta",real_theta.T.shape)






