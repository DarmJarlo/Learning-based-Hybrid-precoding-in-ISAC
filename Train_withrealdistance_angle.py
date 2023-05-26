from __future__ import absolute_import, division, print_function
import tensorflow as tf


import math
import sys
import loss
import config_parameter

sys.path.append("..")

from network import DL_method_NN
from config_parameter import iters
sys.path.append("..")
import numpy as np
#from loss import Estimate_delay_and_doppler
Nt=8
Nr=8




def load_model():


    model = DL_method_NN()
    model.build(input_shape=(None, int(config_parameter.train_data_period/config_parameter.Radar_measure_slot), \
                             config_parameter.num_vehicle,2))
    model.summary()
    if config_parameter.FurtherTrain ==True:
        model = tf.saved_model.load('Keras_models/new_model')
    return model


if __name__ == '__main__':
    model = load_model()
    Antenna_Gain = math.sqrt(config_parameter.antenna_size * config_parameter.receiver_antenna_size)
    c = 3e8
    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    optimizer_1 = tf.keras.optimizers.Adagrad(
        learning_rate=0.001,
        initial_accumulator_value=0.1,
        epsilon=1e-07,
        name="Adagrad"
    )
    crb_d_sum_list = []
    crb_angle_sum_list = []
    @tf.function
    def train_step(input,real_distance_list,last_real_distance_list,step):
        with tf.GradientTape() as tape:
            output = model(input)#dont forget here we are inputing a whole batch
            print('oooooooooooo',step,output)

            Analog_matrix,Digital_matrix= loss.Output2PrecodingMatrix(Output=output)
            precoding_matrix = loss.Precoding_matrix_combine(Analog_matrix,Digital_matrix)
            #print(predictions)
            #estimated_theta_list=input[0,step,:,1]
            print("theta_list_shape",estimated_theta_list.shape)
            #steering_vector = [loss.calculate_steer_vector(predict_theta_list[v] for v in range(config_parameter.num_vehicle)
            communication_loss = loss.loss_Sumrate(real_distance,precoding_matrix,predict_theta_list)
            CRB_d_list = []
            CRB_angle_list = []
            for v in range(config_parameter.num_vehicle):
                cos_theta = real_distance_list[v]**2 - last_real_distance_list[v]**2
                CRB_d = loss.CRB_distance(index=v,distance_list=real_distance,estimated_theta_list,precoding_matrix)
                CRB_d_list.append(CRB_d)
                CRB_angle =loss.CRB_distance(index=v,real_distance,estimated_theta_list,precoding_matrix)
                CRB_angle_list.append(CRB_angle)
            combined_loss = loss.loss_combined(c
        gradients = tape.gradient(communication_loss, model.trainable_variables)


        optimizer_1.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        return gradients




    for i in range(0,config_parameter.iters):
        Reference_Signal = loss.Chirp_signal()
        print("1")
        communication_loss = 0
        initial_location_x = {}
        #speed_dictionary = np.random.uniform(low=config_parameter.speed_low, high=config_parameter.speed_high, size=(config_parameter.num_vehicle,\
                                                                    config_parameter.one_iter_period/(config_parameter.Radar_measure_slot)))
        speed_dictionary = {}
        real_location_x = {} #this one doesn't include the initial location
        real_theta = {}
        location = {} #this one include the initial location
        for vehicle in range(0,config_parameter.num_vehicle):
            initial_location_x[vehicle] = []
            speed_dictionary[vehicle] = [np.random.uniform(low=config_parameter.speed_low,high=config_parameter.speed_high)\
                                         for _ in range(config_parameter.one_iter_period/config_parameter.Radar_measure_slot)]
            #initialize location for every car [0,100)
            random_location = (np.random.rand(1))*100
            print(random_location)
            initial_location_x[vehicle]= random_location
            location[vehicle] = [random_location]
            for i in range(len(speed_dictionary[vehicle])):

                location[vehicle][i+1] = config_parameter.Radar_measure_slot*speed_dictionary[vehicle][i]+\
                    location[vehicle][i]

            real_location_x[vehicle]=location[vehicle][1:]
        print("location while start measuring",real_location_x)


        print("initial location",initial_location_x)

        #step_index = 0
        #dist_index = 1
        estimated_time_delay = {}
        estimated_doppler_frequency = {}
        estimated_distance = {}
        estimated_velocity = {}
        estimated_theta = {}
        real_distance = {}
        for v in range(0,config_parameter.num_vehicle):
            real_distance[v]=[]
            estimated_distance[v] = []
            estimated_velocity[v] = []
            estimated_time_delay[v] = []
            estimated_doppler_frequency[v] = []

            #time_array = np.arange(0,config_parameter.one_iter_period,config_parameter.Radar_measure_slot)
            #print(time_array)
            for time in np.arange(0,config_parameter.one_iter_period/config_parameter.Radar_measure_slot):
                #calculate the real distance between rsu and vehicles,initialize speed for every timeslot
                real_distances[v].append(math.sqrt((config_parameter.RSU_location[1]) ** 2 + (location[v][time]-config_parameter.RSU_location[0]) ** 2))

        #

                #if time > 0:
                    #calculate the new location
                #dist_new = initial_location_x[v]+\
                 #              config_parameter.Radar_measure_slot*speed_dictionary[v][step_index]
                #initial_location_x[v].append(dist_new)

                #dist_index +=1
                #step_index +=1
                target_coordinates=(location[v][time],0)
                tx = loss.Received_Signal(Reference_Signal,speed_dictionary[v][time],\
                                          target_coordinates,real_distances[v][time])
                latency,estimated_dist_this,estimated_velocity_this,doppler_frequency_shift = loss.Matched_filter(Reference_Signal,tx,)
                estimated_latency[v].append(latency)
                estimated_doppler_frequency[v].append(doppler_frequency_shift)
                estimated_distance[v].append(estimated_dist_this)
                estimated_velocity[v].append(estimated_velocity_this)
                #if time >0:
                 #   latency,estimated_distance,estimated_velocity,doppler_frequency_shift = loss.Matched_filter(reference,echo,last_location_y=)



                  #  estimated_distance[v][step_index] = estimated_time_delay[v]*c/2
                   # estimated_velocity[v][step_index] = c/(1/2+(config_parameter.Frequency_original/(estimated_doppler_frequency[veh]-\
                                                                                   config_parameter.Frequency_original)))
                # assume vehicle approaching RSU has the positive V

                #estimated_theta[v][step_index] = math.asin(config_parameter.RSU_location_y/estimated_distance[v][step_index])

            input_whole = np.zeros(shape=(1,config_parameter.one_iter_period/config_parameter.Radar_measure_slot,\
                                          config_parameter.num_vehicle,3))
            real_distance_np = np.zeros((config_parameter.one_iter_period/config_parameter.Radar_measure_slot,\
                                          config_parameter.num_vehicle))
            for index in range(0,config_parameter.num_vehicle):
                input_whole[:,:,index,0] = estimated_distance[index]
                input_whole[:,:,index,1] = estimated_theta[index]
                input_whole[:,:,index,2] = estimated_velocity[index]
                real_distance_np[:,index] = real_distance[index]
                print(real_distance_np)
            num_datapoint = input_whole.shape[1]


            #count how many states in total

            for epo in range(0,num_datapoint-8): #10 points 1 step, 11 points 2 steps
                input_single = input_whole[:,epo:epo+10,:,:]
                real_distance_slice = real_distance_np[epo:epo+10,:]
                input_single = tf.data.Dataset.from_tensor_slices(input_single)
                real_distance_slice = tf.data.Dataset.from_tensor_slices(real_distance_slice)
                train_step(input_single,real_distance_slice,epo)
            tf.saved_model.save(model, 'Keras_models/new_model')
            '''checkpointer = ModelCheckpoint(filepath="Keras_models/weights.{epoch:02d}-{val_accuracy:.2f}.hdf5",
                                           monitor='val_accuracy',
                                           save_weights_only=False, period=1, verbose=1, save_best_only=False)'''
        tf.saved_model.save(model, 'Keras_models/new_model')








            #



    # assume the whole length of observable highway is  540m
    # take 20s as one round.
    '''
    for i in range(0,iters):
        Speed = []
        initial_location_x = []




        real_distance= []
        real_angle = []
        for i in range(0,5):
            Speed.append(np.random(20, 22))

            real_distance.append(np.sqrt(np.square(initial_location_y[i]-RSU_location_x) + np.square(RSU_location_y)))
            real_angle.append(np.sqrt())

    '''



    #input = np.zeros(shape=())
    #input = np.vstack(estimated_distance,estimated_theta)



    # start training
    #for epoch in range(config_parameter.iters):

        'adaptive learning rate for adadelta'

     #   if epoch < 6:
      #      alpha = epoch*0.2+0.2
       # elif epoch > 15 and accu < 0.85:
        #    alpha = 0.2+(epoch-16*0.2)
        #else:
         #   alpha = alpha/5
        #print("alpha",alpha)

        #optimizer_1 = tf.keras.optimizers.Adam(learning_rate=alpha)

        #tf.saved_model.save(model, 'Keras_models/new_model')
    '''checkpointer = ModelCheckpoint(filepath="Keras_models/weights.{epoch:02d}-{val_accuracy:.2f}.hdf5",
                                   monitor='val_accuracy',
                                   save_weights_only=False, period=1, verbose=1, save_best_only=False)'''
    #tf.saved_model.save(model, 'Keras_models/new_model')