from __future__ import absolute_import, division, print_function
import tensorflow as tf


import math
import sys
import loss
import config_parameter

sys.path.append("..")

from network import DL_method_NN_for_v2x_mod,ResNetLSTMModel
from config_parameter import iters
sys.path.append("..")
import numpy as np
#tf.compat.v1.enable_eager_execution()
def load_model():


    model = ResNetLSTMModel()
    num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar +config_parameter.num_horizoncar
    model.build(input_shape=(1, 10, num_vehicle,16))
    model.summary()
    if config_parameter.FurtherTrain ==True:
        model = tf.saved_model.load('Keras_models/new_model')
    return model

#consider a situation we have observer on the road and lower is basestation, upper is car,
# priority is the consideration only contains car without base station.
#for example
#------car---------car--------
#-------observer---------car--
#---------base station--------
if __name__ == '__main__':
    writer = tf.summary.create_file_writer('log.txt')
    if config_parameter.mode == "V2I":
        antenna_size = config_parameter.antenna_size
        num_vehicle = config_parameter.num_vehicle
    elif config_parameter.mode == "V2V":
        antenna_size = config_parameter.vehicle_antenna_size
        num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar + config_parameter.num_horizoncar
    #f.compat.v1.enable_eager_execution()
    print()
    print(tf.__version__)


    Antenna_Gain = math.sqrt(antenna_size * config_parameter.receiver_antenna_size)
    c = config_parameter.c
    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    optimizer_1 = tf.keras.optimizers.Adagrad(
        learning_rate=0.001,
        initial_accumulator_value=0.1,
        epsilon=1e-07,
        name="Adagrad"
    )
    optimizer_1 = tf.keras.optimizers.Adadelta()
    optimizer_1 = tf.keras.optimizers.Adam(learning_rate=0.001)
    model = load_model()
    #model.compile
    """          
                   input_whole[:,:,index,0] = real_distance[index]
                   input_whole[:,:,index,1] = real_theta[index]
                   input_whole[:,:,index,2] = own_velocity[index]
                    maybe : input_whole[:,:,index,3] = target_velocity[index]
                   """

    """
    #@tf.function
    def train_step(input, time):
        #input shape(1,11,5,4)
        with tf.GradientTape() as tape:

            #input = tf.convert_to_tensor(input)
            #input = tf.expand_dims(input_tensor, axis=0)
            output = model(input[:,:,:,:])  # dont forget here we are inputing a whole batch

            print(input[:,0:10,:,:])
            input1 = input.numpy()
            print('oooooooooooo', time, output)
            real_theta_this_list=input1[0,-1,:,0]
            real_distance_this_list = input1[0,-1,:,1]
            own_velocity = input1[0,-1,0,2]
            #target_velocity_list = input[0,-1,:,3]
            output = output.numpy()
            Analog_matrix, Digital_matrix = loss.Output2PrecodingMatrix(Output=output)
            precoding_matrix = loss.Precoding_matrix_combine(Analog_matrix, Digital_matrix)
            print("precoding_matrix")
            # steering_vector = [loss.calculate_steer_vector(predict_theta_list[v] for v in range(config_parameter.num_vehicle)
            communication_loss = loss.tensor_Sum_rate(input[0,-1,:,1], precoding_matrix, input[0,-1,:,0])

            #communication_loss= sum(input[0,-1,:,0])**2 + sum(input[0,-1,:,1])**2+input[0,-1,0,2]**2
            with open("sum_rate.txt", "w") as file:
                # Write the value of the variable to the file
                file.write(str(communication_loss))
            sum_rate_list_reciprocal.append(1 / communication_loss)
            sum_rate_list.append(communication_loss)

            CRB_d_list = []
            CRB_angle_list = []

            for v in range(num_vehicle):
                sigma_time_delay[v, time] = loss.Sigma_time_delay_square(index=v,
                                                                         distance_list=real_distance_this_list,
                                                                         estimated_theta_list=real_theta_this_list, \
                                                                         precoding_matrix=precoding_matrix)
                sigma_doppler[v, time] = loss.Sigma_time_delay_square(index=v, \
                                                                      distance_list=real_distance_this_list, \
                                                                      estimated_theta_list=real_distance_this_list, \
                                                                      precoding_matrix=precoding_matrix)
                # cos_theta = real_distance_list[v]**2 - last_real_distance_list[v]**2
                CRB_d = loss.CRB_distance(sigma_time_delay[v, time])
                CRB_d_list.append(CRB_d)
                CRB_angle = loss.CRB_angle(index=v, distance_list=real_distance_this_list, estimated_theta_list=real_theta_this_list,
                                           precoding_matrix=precoding_matrix)
                CRB_angle_list.append(CRB_angle)

            CRB_d_thisT_sum = loss.CRB_sum(CRB_d_list)
            crb_d_sum_list.append(CRB_d_thisT_sum)
            with open("CRB_d_this.txt", "a") as file:
                # Write the value of the variable to the file
                file.write(str(CRB_d_thisT_sum))
            CRB_angle_thisT_sum = loss.CRB_sum((CRB_angle_list))
            crb_angle_sum_list.append(CRB_angle_thisT_sum)
            with open("CRBtheta_this.txt", "a") as file:
                # Write the value of the variable to the file
                file.write(str(CRB_angle_thisT_sum))
            crb_combined_loss = loss.loss_CRB_combined(CRB_d_list=crb_d_sum_list, CRB_thet_list=crb_angle_sum_list, \
                                                       CRB_d_this_sum=CRB_d_thisT_sum,
                                                       CRB_thet_this_sum=CRB_angle_thisT_sum)
            combined_loss = loss.loss_combined(CRB_d_list=crb_d_sum_list, CRB_thet_list=crb_angle_sum_list,
                                               sumrate_inv_list=sum_rate_list_reciprocal, \
                                               CRB_d_this_sum=CRB_d_thisT_sum, CRB_thet_this_sum=CRB_angle_thisT_sum,
                                               sumrate_inv_this=communication_loss)
            #communication_loss = tf.convert_to_tensor(communication_loss)
            print(communication_loss,time)
            trainable_vars = [tf.Variable(var) for var in model.trainable_variables]
            #combined_loss = tf.convert_to_tensor(combined_loss)
        if config_parameter.loss_mode == "Upper_sum_rate":
            gradients = tape.gradient(communication_loss, model.trainable_variables)
        elif config_parameter.loss_mode == "lower_bound_crb":
            gradients = tape.gradient(crb_combined_loss, model.trainable_variables)
        elif config_parameter.loss_mode == "combined_loss":
            gradients = tape.gradient(combined_loss, model.trainable_variables)

        optimizer_1.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        return gradients
        """
    @tf.function #means not eager mode. graph mode
    def train_step(input):
        #input shape(1,11,5,4)
        with tf.GradientTape() as tape:

            #input = tf.convert_to_tensor(input)
            #input = tf.expand_dims(input_tensor, axis=0)
            output = model(input[:,:,:,:])  # dont forget here we are inputing a whole batch


            print('oooooooooooo', output)

            Analog_matrix, Digital_matrix = loss.tf_Output2PrecodingMatrix(Output=output)
            precoding_matrix = loss.tf_Precoding_matrix_combine(Analog_matrix, Digital_matrix)
            print("precoding_matrix",precoding_matrix)
            CSI = tf.complex(input[0,-1,:,0:antenna_size], input[0,-1,:,antenna_size:2*antenna_size])

            print("csi",CSI)
            # steering_vector = [loss.calculate_steer_vector(predict_theta_list[v] for v in range(config_parameter.num_vehicle)
            communication_loss = loss.tf_loss_sumrate(CSI, precoding_matrix)

            #def get_loss_value(loss_tensor):
             #   return np.array(loss_tensor)
            #loss_value = tf.py_function(get_loss_value, [communication_loss], tf.float32)
            print("loss_value",communication_loss)
            with writer.as_default():

                tf.summary.scalar("CSI", CSI, step=step)
                tf.summary.scalar("G", G, step=step)
                tf.summary.scalar("ANALOG", Analog_matrix, step=step)
                tf.summary.scalar("Digital", Digital_matrix, step=step)
                tf.summary.scalar("Pathloss", Pathloss, step=step)
            #communication_loss = loss.tensor_Sum_rate(CSI, precoding_matrix)
            '''
            #communication_loss= sum(input[0,-1,:,0])**2 + sum(input[0,-1,:,1])**2+input[0,-1,0,2]**2
            with open("sum_rate.txt", "w") as file:
                # Write the value of the variable to the file
                file.write(str(communication_loss))
            sum_rate_list_reciprocal.append(1 / communication_loss)
            sum_rate_list.append(communication_loss)

            CRB_d_list = []
            CRB_angle_list = []

            for v in range(num_vehicle):
                sigma_time_delay[v, time] = loss.Sigma_time_delay_square(index=v,
                                                                         distance_list=real_distance_this_list,
                                                                         estimated_theta_list=real_theta_this_list, \
                                                                         precoding_matrix=precoding_matrix)
                sigma_doppler[v, time] = loss.Sigma_time_delay_square(index=v, \
                                                                      distance_list=real_distance_this_list, \
                                                                      estimated_theta_list=real_distance_this_list, \
                                                                      precoding_matrix=precoding_matrix)
                # cos_theta = real_distance_list[v]**2 - last_real_distance_list[v]**2
                CRB_d = loss.CRB_distance(sigma_time_delay[v, time])
                CRB_d_list.append(CRB_d)
                CRB_angle = loss.CRB_angle(index=v, distance_list=real_distance_this_list, estimated_theta_list=real_theta_this_list,
                                           precoding_matrix=precoding_matrix)
                CRB_angle_list.append(CRB_angle)

            CRB_d_thisT_sum = loss.CRB_sum(CRB_d_list)
            crb_d_sum_list.append(CRB_d_thisT_sum)
            with open("CRB_d_this.txt", "a") as file:
                # Write the value of the variable to the file
                file.write(str(CRB_d_thisT_sum))
            CRB_angle_thisT_sum = loss.CRB_sum((CRB_angle_list))
            crb_angle_sum_list.append(CRB_angle_thisT_sum)
            with open("CRBtheta_this.txt", "a") as file:
                # Write the value of the variable to the file
                file.write(str(CRB_angle_thisT_sum))
            crb_combined_loss = loss.loss_CRB_combined(CRB_d_list=crb_d_sum_list, CRB_thet_list=crb_angle_sum_list, \
                                                       CRB_d_this_sum=CRB_d_thisT_sum,
                                                       CRB_thet_this_sum=CRB_angle_thisT_sum)
            combined_loss = loss.loss_combined(CRB_d_list=crb_d_sum_list, CRB_thet_list=crb_angle_sum_list,
                                               sumrate_inv_list=sum_rate_list_reciprocal, \
                                               CRB_d_this_sum=CRB_d_thisT_sum, CRB_thet_this_sum=CRB_angle_thisT_sum,
                                               sumrate_inv_this=communication_loss)
            #communication_loss = tf.convert_to_tensor(communication_loss)
            print(communication_loss,time)
            trainable_vars = [tf.Variable(var) for var in model.trainable_variables]
            #combined_loss = tf.convert_to_tensor(combined_loss)
            '''
        if config_parameter.loss_mode == "Upper_sum_rate":
            gradients = tape.gradient(communication_loss, model.trainable_variables)
        elif config_parameter.loss_mode == "lower_bound_crb":
            gradients = tape.gradient(crb_combined_loss, model.trainable_variables)
        elif config_parameter.loss_mode == "combined_loss":
            gradients = tape.gradient(combined_loss, model.trainable_variables)

        optimizer_1.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        return communication_loss

    for i in range(0, config_parameter.iters):

        crb_d_sum_list = []  # the crb distance sum at all timepoints in this list
        crb_angle_sum_list = []  # the crb angle sum at all timepoints in this list

        sum_rate_list_reciprocal = []  # the sum rate at all timepoints in this list
        sum_rate_list = []  # the sum rate at all timepoints in this list
        sigma_time_delay = np.zeros(
            shape=(num_vehicle, int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot) + 1),
            dtype=complex)
        sigma_doppler = np.zeros(
            shape=(num_vehicle, int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot) + 1),
            dtype=complex)
        # Reference_Signal = loss.Chirp_signal()
        print("1")
        speed_own_dictionary=np.zeros(shape=(
        1, int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot)))
        speed_upper_car_dictionary = np.zeros(shape=(
        config_parameter.num_uppercar, int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot)))
        real_location_uppercar_x = np.zeros(shape=(
        config_parameter.num_uppercar, int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot)))

        speed_lower_car_dictionary = np.zeros(shape=(
        config_parameter.num_lowercar, int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot)))
        real_location_lowercar_x = np.zeros(shape=(
            config_parameter.num_lowercar, int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot)))
        speed_horizon_car_dictionary = np.zeros(shape=(
            config_parameter.num_horizoncar, int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot)))
        real_location_horizoncar_x = np.zeros(shape=(
            config_parameter.num_horizoncar, int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot)))
        # real location at every timepoint
      # this one doesn't include the initial location
        # real angle at every timepoint in this iter
        totalnum_vehicle = config_parameter.num_uppercar+config_parameter.num_lowercar+config_parameter.num_horizoncar
        real_theta = np.zeros(shape=(totalnum_vehicle, int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot)))



        speed_own_dictionary = np.random.uniform(config_parameter.horizonspeed_low,config_parameter.horizonspeed_high, size=speed_own_dictionary.shape)
        speed_horizon_car_dictionary = np.random.uniform(config_parameter.horizonspeed_low,config_parameter.horizonspeed_high, size=speed_horizon_car_dictionary.shape)
        speed_lower_car_dictionary = np.random.uniform(config_parameter.lowerspeed_low,config_parameter.lowerspeed_high, size=speed_lower_car_dictionary.shape)
        speed_upper_car_dictionary = np.random.uniform(config_parameter.upperspeed_low, config_parameter.upperspeed_high,
                                                       size=speed_upper_car_dictionary.shape)
        speed_whole_dictionary = np.vstack((speed_own_dictionary,speed_lower_car_dictionary,speed_upper_car_dictionary,speed_horizon_car_dictionary))
        print(speed_whole_dictionary.shape)
        num_whole = totalnum_vehicle+1 #1 is the observer
        initial_car_x = np.zeros(shape=(num_whole,1))
        initial_car_x[0,0]=0
        location_car_y = np.zeros(shape=(num_whole,int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot)+1))
        location_car_y[0, :] = np.full(
            (1, int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot) + 1), 0)

        initial_car_x[1,0]=np.random.uniform(config_parameter.Initial_horizoncar1_min, config_parameter.Initial_horizoncar1_max)
        location_car_y[1,:] = np.full((1, int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot)+1), 0)
        initial_car_x[2,0]=np.random.uniform(config_parameter.Initial_lowercar1_min, config_parameter.Initial_lowercar1_max)
        location_car_y[2, :] = np.full(
            (1, int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot) + 1), -20)
        initial_car_x[3, 0] = np.random.uniform(config_parameter.Initial_uppercar1_min,
                                                config_parameter.Initial_uppercar1_max)
        location_car_y[3, :] = np.full(
            (1, int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot) + 1), 20)
        #initial_car_x[4,0]=np.random.uniform(config_parameter.Initial_lowercar2_min, config_parameter.Initial_lowercar2_max)
        #initial_car_x[5, 0] = np.random.uniform(config_parameter.Initial_uppercar2_min,
                                               # config_parameter.Initial_uppercar2_max)


        location_car_x = np.zeros(shape=(
        num_whole, int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot)+1))
        coordinates_car = np.zeros(shape=(
        num_whole, int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot)+1))
        location_car_x[:,0]=initial_car_x[:,:].reshape((4,))
        real_distance = np.zeros(shape=(
        totalnum_vehicle, int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot)+1))
        real_theta = np.zeros(shape=(
        totalnum_vehicle, int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot)+1))
        for v in range(num_whole):
            for t in range(int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot)):
                location_car_x[v,t+1] = location_car_x[v,t]+speed_whole_dictionary[v,t]*config_parameter.Radar_measure_slot

        for v in range(1,num_whole):
            for t in range(int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot)+1):
                real_distance[v-1,t]=math.sqrt((location_car_x[0, t] - location_car_x[v,t]) ** 2 + (location_car_y[0,t] - location_car_y[v,t]) ** 2)
                real_theta[v-1,t]=math.atan2(location_car_y[v,t]-location_car_y[0,t],location_car_x[v,t]-location_car_x[0,t])
                if real_theta[v-1,t]==0:
                    real_theta[v-1,t]==0.1
                print(real_theta[v-1,t])



        #set input as (1,5,11,3) containing angle distance and velocity
        if config_parameter.mode == "V2I":
            antenna_size = config_parameter.antenna_size
            #num_vehicle = config_parameter.num_vehicle
        elif config_parameter.mode == "V2V":
            antenna_size = config_parameter.vehicle_antenna_size
            #num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar + config_parameter.num_horizoncar

        CSI_whole = np.zeros(shape=(int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot)+1,totalnum_vehicle,antenna_size),dtype=complex)

        for v in range(0,totalnum_vehicle):
            for t in range(0,int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot)+1):
                print("ssssssssssssssssssssssss",real_distance[v,t],real_theta[v,t])
                CSI_whole[t,v,:]=loss.calculate_CSI(real_distance[v,t],real_theta[v,t])#based on the 1-10 data create a precoding matrix to make a precoding matrix and calculate the precoding matrix for 11.
        input_whole = np.zeros(shape=(int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot)+1,totalnum_vehicle,2*antenna_size))
        input_whole[:,:,0:antenna_size] = np.real(CSI_whole)
        input_whole[:,:,antenna_size:2*antenna_size] = np.imag(CSI_whole)
        #input_whole[:,:,antenna_size:2antenna_size] = velocity
        #input_whole[:, :, 0] = np.transpose(real_theta)
        #input_whole[:-1,:,2]=np.tile(np.transpose(speed_own_dictionary), (1,totalnum_vehicle))
        #print("obsever velocity",input_whole[:,:,2])
        #input_whole[:,:,1]=np.transpose(real_distance)
        #print("real distance",input_whole[:,:,1])
        #input_whole[:,:,0]=np.transpose(real_theta)
        #print("real angle", input_whole[:, :, 0])

        for epo in range(int(config_parameter.one_iter_period / config_parameter.Radar_measure_slot)+1-9):

            print(input_whole.shape)
            communication_loss = 0
            input_single = input_whole[epo:epo + 10, :, :]
            input_single = tf.convert_to_tensor(input_single)
            input_single=tf.expand_dims(input_single, axis=0)
            communication_loss,=train_step(input_single)
            print("Epoch: {}/{}, step: {},loss: {}".format(i + 1,config_parameter.iters, epo,communication_loss.numpy()
                                                                                     ))

            #tf.saved_model.save(model, 'Keras_models/new_model')
            model.save_weights(filepath='Keras_models/new_model', save_format='tf')
            '''checkpointer = ModelCheckpoint(filepath="Keras_models/weights.{epoch:02d}-{val_accuracy:.2f}.hdf5",
                                               monitor='val_accuracy',
                                               save_weights_only=False, period=1, verbose=1, save_best_only=False)'''
    #tf.saved_model.save(model, )
    model.save_weights(filepath='Keras_models/new_model', save_format='tf')






