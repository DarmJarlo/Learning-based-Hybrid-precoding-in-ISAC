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

    """          
                input_whole[:,:,index,0] = estimated_distance[index]
                input_whole[:,:,index,1] = estimated_theta[index]
                input_whole[:,:,index,2] = estimated_velocity[index]
                """
    @tf.function
    def train_step(input,real_distance_list,last_real_distance_list,time):
        with tf.GradientTape() as tape:
            output = model(input)#dont forget here we are inputing a whole batch
            input = input.numpy()
            print('oooooooooooo',time,output)

            Analog_matrix,Digital_matrix = loss.Output2PrecodingMatrix(Output=output)
            precoding_matrix = loss.Precoding_matrix_combine(Analog_matrix,Digital_matrix)
            #print(predictions)
            #estimated_theta_list=input[0,step,:,1]
            real_theta = []# this list is for the real



            predict_theta_list = real_theta # in the future can be considered to be replaced by the prediction model of theta velocity and distance
            print("theta_list_shape",real_theta.shape)
            #steering_vector = [loss.calculate_steer_vector(predict_theta_list[v] for v in range(config_parameter.num_vehicle)
            communication_loss = loss.loss_Sumrate(input[0,9,:,0],precoding_matrix,input[0,9,:,1])
            with open("sum_rate.txt", "w") as file:
                # Write the value of the variable to the file
                file.write(communication_loss)
            sum_rate_list_reciprocal.append(1/communication_loss)
            sum_rate_list.append(communication_loss)

            CRB_d_list = []
            CRB_angle_list = []
            for v in range(config_parameter.num_vehicle):
                sigma_time_delay[v, time] = loss.Sigma_time_delay_square(index=v,
                                                                         distance_list=real_distance[:, time],
                                                                         estimated_theta_list=input, \
                                                                         precoding_matrix=precoding_matrix)
                sigma_doppler[v, time] = loss.Sigma_time_delay_square(index=v,\
                                                                      distance_list = real_distance[:,time], \
                                                                      estimated_theta_list = real_theta[:,time], \
                                                                      precoding_matrix = precoding_matrix)
                #cos_theta = real_distance_list[v]**2 - last_real_distance_list[v]**2
                CRB_d = loss.CRB_distance(sigma_time_delay[v,time])
                CRB_d_list.append(CRB_d)
                CRB_angle =loss.CRB_angle(index=v,real_distance=real_distance_list,estimated_theta_list=real_theta,precoding_matrix=precoding_matrix)
                CRB_angle_list.append(CRB_angle)

                estimated_time_delay[v,time+1] = real_time_delay[v,time+1]+ np.random.normal(0, sigma_time_delay[v,time], 1)
                estimated_doppler_frequency[v, time + 1] = real_doppler_frequency[v, time + 1] + np.random.normal(0, sigma_doppler[v, time], 1)
                estimated_distance[v,time+1] = 0.5*config_parameter.c*estimated_time_delay[v,time+1] # precoding matrix for this time is for the estimation_next time
                estimated_velocity_between_norm = (estimated_distance[v,time+1]-estimated_distance[v,time])/config_parameter.Radar_measure_slot
            crb_d_sum_list.append(CRB_d_list)
            CRB_d_thisT_sum = loss.CRB_sum(CRB_d_list)
            CRB_angle_thisT_sum = loss.CRB_sum((CRB_angle_list))
            crb_angle_sum_list.append(CRB_angle_list)
            crb_combined_loss = loss.loss_CRB_combined(CRB_d_list=crb_d_sum_list,CRB_thet_list=crb_angle_sum_list,\
                                          CRB_d_this_sum=CRB_d_thisT_sum,CRB_thet_this_sum=CRB_angle_thisT_sum)
            combined_loss = loss.loss_combined(CRB_d_list=crb_d_sum_list,CRB_thet_list=crb_angle_sum_list,sumrate_list=sum_rate_list,\
                                          CRB_d_this_sum=CRB_d_thisT_sum,CRB_thet_this_sum=CRB_angle_thisT_sum,sumrate_this=communication_loss)

            if config_parameter.loss_mode == "Upper_sum_rate":
                gradients = tape.gradient(communication_loss, model.trainable_variables)
            elif config_parameter.loss_mode == "lower_bound_crb":
                gradients = tape.gradient(crb_combined_loss, model.trainable_variables)
            elif config_parameter.loss_mode == "combined_loss":
                gradients = tape.gradient(combined_loss, model.trainable_variables)



        optimizer_1.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        return gradients




    for i in range(0,config_parameter.iters):
        crb_d_sum_list = []  # the crb distance sum at all timepoints in this list
        crb_angle_sum_list = []  # the crb angle sum at all timepoints in this list

        sum_rate_list_reciprocal = []  # the sum rate at all timepoints in this list
        sum_rate_list = []  # the sum rate at all timepoints in this list
        #Reference_Signal = loss.Chirp_signal()
        print("1")
        #communication_loss = 0


        #generate dataset
        initial_location_x = {}
        #speed_dictionary = np.random.uniform(low=config_parameter.speed_low, high=config_parameter.speed_high, size=(config_parameter.num_vehicle,\
                         #                                           config_parameter.one_iter_period/(config_parameter.Radar_measure_slot)))
        #speed at every timepoint
        speed_dictionary = np.zeros(shape=(config_parameter.num_vehicle,int(config_parameter.one_iter_period/config_parameter.Radar_measure_slot)))
        #real location at every timepoint
        real_location_x = np.zeros(shape=(config_parameter.num_vehicle,int(config_parameter.one_iter_period/config_parameter.Radar_measure_slot))) #this one doesn't include the initial location
        #real angle at every timepoint in this iter
        real_theta = np.zeros(shape=(config_parameter.num_vehicle,int(config_parameter.one_iter_period/config_parameter.Radar_measure_slot)))
        #real distance between target and base station
        real_distance_list = np.zeros(shape=(
        config_parameter.num_vehicle, int(config_parameter.one_iter_period/config_parameter.Radar_measure_slot)))
        #real distance between
        last_real_distance_list = np.zeros(shape=(
        config_parameter.num_vehicle, int(config_parameter.one_iter_period/config_parameter.Radar_measure_slot)))
        #real x coordinates of target including the initial location
        location = np.zeros(shape=(config_parameter.num_vehicle,(int(config_parameter.one_iter_period/config_parameter.Radar_measure_slot)+1))) #this one include the initial location
        #real y coordinates of target including the initial location
        location_y = np.zeros(shape=(config_parameter.num_vehicle,(int(config_parameter.one_iter_period/config_parameter.Radar_measure_slot)+1)))
        #target coordinates combined
        target_coordinates = np.zeros(shape=(config_parameter.num_vehicle,int(config_parameter.one_iter_period/config_parameter.Radar_measure_slot)))
        real_time_delay=np.zeros(shape=(config_parameter.num_vehicle,int(config_parameter.one_iter_period/config_parameter.Radar_measure_slot)))
        real_doppler_frequency = np.zeros(shape=(config_parameter.num_vehicle,int(config_parameter.one_iter_period/config_parameter.Radar_measure_slot)))

        estimated_time_delay = np.zeros(shape=(config_parameter.num_vehicle,int(config_parameter.one_iter_period/config_parameter.Radar_measure_slot)))
        estimated_doppler_frequency = np.zeros(shape=(config_parameter.num_vehicle,int(config_parameter.one_iter_period/config_parameter.Radar_measure_slot)))
        estimated_distance = np.zeros(shape=(config_parameter.num_vehicle,int(config_parameter.one_iter_period/config_parameter.Radar_measure_slot)))
        estimated_velocity = np.zeros(shape=(config_parameter.num_vehicle,int(config_parameter.one_iter_period/config_parameter.Radar_measure_slot)))
        estimated_theta = real_theta
        for vehicle in range(0,config_parameter.num_vehicle):
            #initial_location_x[vehicle] = []
            speed_dictionary[vehicle] = [np.random.uniform(low=config_parameter.speed_low,high=config_parameter.speed_high)\
                                         for _ in range(int(config_parameter.one_iter_period/config_parameter.Radar_measure_slot))]
            print(speed_dictionary.shape)
            #initialize location for every car [0,100)
            random_location = (np.random.rand(1))*100
            print(random_location)

            initial_location_x[vehicle]= random_location
            location[vehicle,0] = random_location
            #location_y[vehicle] = [0]
            for i in range(int(config_parameter.one_iter_period/config_parameter.Radar_measure_slot)):



                location[vehicle,i+1] = config_parameter.Radar_measure_slot*speed_dictionary[vehicle,i]+\
                    location[vehicle,i]

                #location_y[vehicle].append(0)
                real_theta[vehicle,i]=math.atan2(location[vehicle,i] - config_parameter.RSU_location[0],
                                             location_y[vehicle,i] - config_parameter.RSU_location[1])
                target_coordinates[vehicle,i] = (location[v,i+1], location_y[v,i+1])
                real_distance_list[vehicle, i] = math.sqrt((location[vehicle,i] - config_parameter.RSU_location[0])**2\
                                                               +(location_y[vehicle,i] - config_parameter.RSU_location[1])**2)

                real_time_delay[vehicle,i] = 2 * real_distance_list[v,time] / config_parameter.c
                real_doppler_frequency[vehicle,i] =
            real_location_x[vehicle]=location[vehicle][1:]
        #for vehicle in range(0,config_parameter.num_vehicle):
         #   for time in range(0,int(config_parameter.one_iter_period/config_parameter.Radar_measure_slot)):
          #      sigma_time_delay[v, time] = loss.Sigma_time_delay_square(index=v,distance_list = real_distance[:, time],estimated_theta_list=real_theta[:,time],\
             #                                                            precoding_matrix=precoding_matrix)
           #     sigma_doppler[v, time] = loss.Sigma_time_delay_square(index=vdistance_list = real_distance[:, time],estimated_theta_list=real_theta[:,time].\
            #        precoding_matrix=precoding_matrix)
        print("location while start measuring",real_location_x)


        print("initial location",initial_location_x)

        #step_index = 0
        #dist_index = 1


        #real_distance = np.zeros(shape=(config_parameter.num_vehicle,int(config_parameter.one_iter_period/config_parameter.Radar_measure_slot)))
        #sigma_time_delay[v, time] = loss.Sigma_time_delay_square(index=vdistance_list = real_distance[:, time],
        '''
        for v in range(0,config_parameter.num_vehicle):

            for time in np.arange(0,int(config_parameter.one_iter_period/config_parameter.Radar_measure_slot)):
                #calculate the real distance between rsu and vehicles,initialize speed for every timeslot


                #estimated_time_delay[v,time] = real_time_delay[v,time] + np.random.normal(0, np.sqrt(config_parameter.Signal_noise_power))
        #

                #if time > 0:
                    #calculate the new location
                #dist_new = initial_location_x[v]+\
                 #              config_parameter.Radar_measure_slot*speed_dictionary[v][step_index]
                #initial_location_x[v].append(dist_new)

                #dist_index +=1
                #step_index +=1
                target_coordinates=(location[v][time],0)
                #tx = loss.Received_Signal(Reference_Signal,speed_dictionary[v][time],\
                 #                         target_coordinates,real_distances[v][time])
                #latency,estimated_dist_this,estimated_velocity_this,doppler_frequency_shift = loss.Matched_filter(Reference_Signal,tx,)
                estimated_latency[v].append(latency)
                estimated_doppler_frequency[v].append(doppler_frequency_shift)
                estimated_distance[v].append(estimated_dist_this)
                estimated_velocity[v].append(estimated_velocity_this)
                #if time >0:
                 #   latency,estimated_distance,estimated_velocity,doppler_frequency_shift = loss.Matched_filter(reference,echo,last_location_y=)



                  #  estimated_distance[v][step_index] = estimated_time_delay[v]*c/2
                   # estimated_velocity[v][step_index] = c/(1/2+(config_parameter.Frequency_original/(estimated_doppler_frequency[veh]-\
                                                                                   #config_parameter.Frequency_original)))
                # assume vehicle approaching RSU has the positive V

                #estimated_theta[v][step_index] = math.asin(config_parameter.RSU_location_y/estimated_distance[v][step_index])
'''
            estimated_distance[0:10,:]=real_distance_list[0:10,:]
            estimated_theta[0:10,:]=real_theta[0:10,:]

            input_whole = np.zeros(shape=(1,int(config_parameter.one_iter_period/config_parameter.Radar_measure_slot),\
                                          config_parameter.num_vehicle,3))
            real_distance_np = np.zeros((int(config_parameter.one_iter_period/config_parameter.Radar_measure_slot),\
                                          config_parameter.num_vehicle))


            for index in range(0,config_parameter.num_vehicle):
                input_whole[:,:,index,0] = estimated_distance[index]
                input_whole[:,:,index,1] = estimated_theta[index]
                input_whole[:,:,index,2] = estimated_velocity[index]
                real_distance_np[:,index] = real_distance[index]
                print(real_distance_np)
            num_datapoint = input_whole.shape[1]


            #count how many states in total

            for epo in range(0,num_datapoint-9): #10 points 1 step, 11 points 2 steps
                input_single = input_whole[:,epo:epo+10,:,:]
                real_distance_slice = real_distance_np[epo:epo+10,:]
                input_single = tf.data.Dataset.from_tensor_slices(input_single)
                real_distance_slice = tf.data.Dataset.from_tensor_slices(real_distance_slice)
                real_distance_this_list = real_distances[:,epo+9]#real distances at this timepoint

                train_step(input_single,real_distance_slice,time=epo)
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

     #   'adaptive learning rate for adadelta'

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