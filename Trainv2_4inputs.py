"""
In this file, we want to input transmitsteering vector,distance and angle

"""

from __future__ import absolute_import, division, print_function
import tensorflow as tf


import math
import sys
import loss
import config_parameter

sys.path.append("..")
import matplotlib.pyplot as plt

from network import DL_method_NN_for_v2x_mod,ResNetLSTMModel,ResNet
from config_parameter import iters
sys.path.append("..")
import numpy as np
#tf.compat.v1.enable_eager_execution()
def load_model():


    model = ResNet()
    #model = ResNetLSTMModel()
    num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar +config_parameter.num_horizoncar
    model.build(input_shape=(config_parameter.batch_size, 1, num_vehicle,80))
    model.summary()
    if config_parameter.FurtherTrain ==True:
        #model = tf.saved_model.load('Keras_models/new_model')
        model.load_weights(filepath='Keras_models/new_model')
    return model
'''
class Datastorager():
    def __init__(self):
        self.output = None
        self.analog_precoding = None
        self.G = None
'''
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

if __name__ == '__main__':
    writer = tf.summary.create_file_writer("log.txt")
    if config_parameter.mode == "V2I":
        antenna_size = config_parameter.antenna_size
        num_vehicle = config_parameter.num_vehicle
    elif config_parameter.mode == "V2V":
        antenna_size = config_parameter.vehicle_antenna_size
        num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar + config_parameter.num_horizoncar
    print(tf.__version__)

    Antenna_Gain = math.sqrt(antenna_size * config_parameter.receiver_antenna_size)
    c = config_parameter.c
    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    #if gpus:
     #   for gpu in gpus:
      #      tf.config.experimental.set_memory_growth(gpu, True)
    optimizer_1 = tf.keras.optimizers.Adam(learning_rate=0.01)
    #optimizer_1 = tf.keras.optimizers.Adagrad(
     #   learning_rate=0.05,
      #  initial_accumulator_value=0.1,
       # epsilon=1e-07,
        #name="Adagrad"
   # )
    model = load_model()


    @tf.function  # means not eager mode. graph mode
    def train_step(input):
        # input shape(1,10,3,32)
        with tf.GradientTape() as tape:
            if config_parameter.mode == "V2I":
                antenna_size = config_parameter.antenna_size
                num_vehicle = config_parameter.num_vehicle
            elif config_parameter.mode == "V2V":
                antenna_size = config_parameter.vehicle_antenna_size
                num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar + config_parameter.num_horizoncar
            output = model(input)
            num_vehicle_f = tf.cast(num_vehicle, tf.float32)
            antenna_size_f = tf.cast(antenna_size,tf.float32)
            # dont forget here we are inputing a whole batch
            G =tf.math.sqrt(antenna_size_f)


            shape = tf.shape(input)

            batch_size= shape[0]
            Analog_matrix, Digital_matrix = loss.tf_Output2PrecodingMatrix(Output=output)
            precoding_matrix = loss.tf_Precoding_matrix_combine(Analog_matrix, Digital_matrix)


            steering_vector_this = tf.complex(input[:,-1,:,0:antenna_size], input[:,-1,:,antenna_size:2*antenna_size])
            #below is the real right steering vector
            steering_vector_this = tf.transpose(steering_vector_this, perm=[0, 2, 1])
            CSI = tf.complex(input[:,-1,:,2*antenna_size:3*antenna_size], input[:,-1,:,3*antenna_size:4*antenna_size])
            #CSi here shape is (BATCH,NUMVEHICLE,ANTENNAS)
            zf_matrix = tf.complex(input[:,-1,:,4*antenna_size:5*antenna_size], input[:,-1,:,5*antenna_size:6*antenna_size])

            zf_sumrate = loss.tf_loss_sumrate(CSI,tf.transpose(zf_matrix,perm=[0,2,1]))
            #sigma_doppler = loss.tf
            #steering_hermite = tf.transpose(tf.math.conj(steering_vector_this))

            #pathloss=loss.tf_Path_loss(input[:,-1,:,0])
            #pathloss = tf.expand_dims(pathloss, axis=1)
            #pathloss = tf.broadcast_to(pathloss, tf.shape(steering_hermite))
            #CSI = tf.multiply(tf.cast(tf.multiply(G, pathloss),dtype=tf.complex128), steering_hermite)
            #zf_beamformer = loss.tf_zero_forcing(CSI)
            #loss_MSE = loss.tf_matrix_mse(zf_beamformer,precoding_matrix)


            # steering_vector = [loss.calculate_steer_vector(predict_theta_list[v] for v in range(config_parameter.num_vehicle)
            sum_rate_this = loss.tf_loss_sumrate(CSI, precoding_matrix)
            sum_rate_this = tf.cast(sum_rate_this, tf.float32)
            zf_sumrate = tf.cast(zf_sumrate, tf.float32)
            batch_size = tf.cast(batch_size, tf.float32)
            #communication_loss = -sum_rate_this
            #communication_loss = -sum_rate_this+loss_MSE/(tf.stop_gradient(loss_MSE/(sum_rate_this)))
            communication_loss = tf.reduce_sum(zf_sumrate-sum_rate_this)/batch_size
            #Sigma_time_delay =
            #CRB_d = loss.tf_CRB_distance()
            #communication_loss = communication_loss/input.shape[0]
            #crb_loss =
            #communication_loss = tf.math.divide(1.0, sum_rate_this)
        if config_parameter.loss_mode == "Upper_sum_rate":
            gradients = tape.gradient(communication_loss, model.trainable_variables)
        elif config_parameter.loss_mode == "lower_bound_crb":
            gradients = tape.gradient(crb_combined_loss, model.trainable_variables)
        elif config_parameter.loss_mode == "combined_loss":
            gradients = tape.gradient(combined_loss, model.trainable_variables)
        '''
        with writer.as_default():
            tf.summary.histogram("CSIIMAG", tf.math.imag(CSI))
            tf.summary.scalar("G", G)
            tf.summary.histogram("CSIREAL", tf.math.real(CSI))
            tf.summary.scalar("Digital", Digital_matrix)
            tf.summary.scalar("Pathloss", pathloss)
        optimizer_1.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))
        '''
        return communication_loss,precoding_matrix,CSI,gradients


    crb_d_sum_list = []  # the crb distance sum at all timepoints in this list
    crb_angle_sum_list = []  # the crb angle sum at all timepoints in this list

    sum_rate_list_reciprocal = []  # the sum rate at all timepoints in this list
    sum_rate_list = []

    for iter in range(0, config_parameter.iters):

        # the sum rate at all timepoints in this list
        sigma_time_delay = np.zeros(
            shape=(
            num_vehicle, int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot) + 1),
            dtype=complex)
        sigma_doppler = np.zeros(
            shape=(
            num_vehicle, int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot) + 1),
            dtype=complex)
        # Reference_Signal = loss.Chirp_signal()
        print("1")

        input_whole = loss.generate_random_sample()
        for epo in range(0,45):

            #print(input_whole.shape)
            communication_loss = 0
            input_single = input_whole[4*epo:4*epo + config_parameter.batch_size, :, :]
            input_single = tf.convert_to_tensor(input_single)
            input_single=tf.expand_dims(input_single, axis=0)
            input_single = tf.transpose(input_single,perm = [1,0,2,3])

            #communication_loss,crb_dThis,crb_angelTHis=train_step(input_single)
            communication_loss,precoding_matrix,CSI,gradients = train_step(input_single)

            print("Epoch: {}/{}, step: {},loss: {}".format(iter + 1,config_parameter.iters, epo,communication_loss.numpy()
                                                                                     ))
            file_path = "precoding_matrix.txt"
            with open(file_path, "a") as file:

                file.write("precoding")
                file.write(str(precoding_matrix.numpy()) + "\n")
                file.write("theta")
                file.write(str(input_single[0,-1,0:num_vehicle,2*antenna_size]) + "\n")
                file.write("distance")
                file.write(str(input_single[0,-1,0:num_vehicle,3*antenna_size]) + "\n")
                file.write("CSI")
                file.write(str(CSI.numpy()) + "\n")
                file.write("gradients")
                file.write(str(gradients) + "\n")
        sum_rate_list.append(communication_loss)
        timestep = list(range(1, len(sum_rate_list) + 1))
        plt.plot(timestep, sum_rate_list, 'b-o')
        plt.xlabel('Timestep')
        plt.ylabel('Loss')
        plt.title('Loss vs Timestep')
        plt.grid(True)
        plt.show()
            #tf.saved_model.save(model, 'Keras_models/new_model')
        model.save_weights(filepath='Keras_models/new_model', save_format='tf')
        '''checkpointer = ModelCheckpoint(filepath="Keras_models/weights.{epoch:02d}-{val_accuracy:.2f}.hdf5",
                                               monitor='val_accuracy',
                                               save_weights_only=False, period=1, verbose=1, save_best_only=False)'''
        #tf.saved_model.save(model, )
    model.save_weights(filepath='Keras_models/new_model', save_format='tf')