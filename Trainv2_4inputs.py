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

from network import DL_method_NN_for_v2x_hybrid,DL_method_NN_for_v2x_mod
from config_parameter import iters
sys.path.append("..")
import numpy as np
#tf.compat.v1.enable_eager_execution()
def load_model():

    model = DL_method_NN_for_v2x_hybrid()
    #model = ResNet()
    #model = ResNetLSTMModel()
    num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar +config_parameter.num_horizoncar

    model.build(input_shape=(config_parameter.batch_size, num_vehicle,48,1))

    model.summary()
    if config_parameter.FurtherTrain ==True:
        #model = tf.saved_model.load('Keras_models/new_model')
        model.load_weights(filepath='Keras_models_hybrid/new_model')
    return model


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
    #optimizer_1 = tf.keras.optimizers.Adam(learning_rate=0.003,beta_1 = 0.91,beta_2 = 0.99)
    #optimizer_1 = tf.keras.optimizers.Adagrad(learning_rate=0.004)
    '''
    optimizer_1 = tf.keras.optimizers.Adagrad(
        learning_rate=0.01,
        initial_accumulator_value=0.1,
        epsilon=1e-07,
        name="Adagrad"
    )
    '''
    model = load_model()

    #profiler = tf.profiler.experimental.Profiler(tf.profiler.experimental.ProfilerOptions(host_tracer_level=2))
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
            analog_rad = input[:,0:config_parameter.rf_size,6*antenna_size:7*antenna_size,0]
            digital_ref = tf.complex(input[:,0:config_parameter.rf_size,\
                                   2*antenna_size:2*antenna_size+num_vehicle,0], \
                                   input[:,:,3*antenna_size:3*antenna_size+num_vehicle,0])
            distance = input[:, :, 5 * antenna_size:6*antenna_size, 0]
            distance = tf.multiply(distance, 100)
            distance = tf.cast(distance, tf.float64)
            theta = input[:,:, 4 * antenna_size, 0]
            steering_vector_this_o = tf.complex(input[:, :, 0:antenna_size, 0],
                                                input[:, :, antenna_size:2 * antenna_size, 0])
            #steering_vector_this = tf.transpose(steering_vector_this_o, perm=[0, 2, 1])
            pathloss = loss.tf_Path_loss(distance)
            beta = loss.Reflection_coefficient(distance)

            num_vehicle_f = tf.cast(num_vehicle, tf.float32)
            antenna_size_f = tf.cast(antenna_size,tf.float32)
            # dont forget here we are inputing a whole batch
            G =tf.math.sqrt(antenna_size_f)
            CSI = tf.multiply(tf.cast(tf.multiply(G, pathloss),dtype=tf.complex128), steering_vector_this_o)



            shape = tf.shape(input)

            batch_size= shape[0]
            #Analog_matrix, Digital_matrix = loss.tf_Output2PrecodingMatrix_rad(Output=output)
            Analog_matrix, Digital_matrix = loss.tf_Output2PrecodingMatrix_rad_mod(Output=output,\
                                                                                   analog_ref=analog_rad,\
                                                                                   digital_ref=digital_ref)


            #precoding_matrix = loss.tf_Precoding_matrix_combine(Analog_matrix, Digital_matrix)
            precoding_matrix = loss.tf_Precoding_matrix_comb_Powerallocated(Analog_matrix, Digital_matrix,distance[:,:,0])
            #precoding_matrix = loss.tf_Precoding_comb_no_powerconstarint(Analog_matrix, Digital_matrix)

            pathloss = loss.tf_Path_loss(input[:,:,2*antenna_size,0])
            #below is the real right steering vector
            steering_vector_this = tf.transpose(steering_vector_this_o, perm=[0, 2, 1])
            #CSI = tf.complex(input[:,:,2*antenna_size:3*antenna_size,0], input[:,:,3*antenna_size:4*antenna_size,0])
            #CSi here shape is (BATCH,NUMVEHICLE,ANTENNAS)
            zf_matrix = tf.complex(input[:,:,2*antenna_size:3*antenna_size,0], input[:,:,3*antenna_size:4*antenna_size,0])
            #Analog_matrix, Digital_matrix = loss.tf_Output2PrecodingMatrix_rad(Output=output,zf_matrix=zf_matrix)
            #precoding_matrix = loss.tf_Precoding_matrix_combine(Analog_matrix, Digital_matrix)
            zf_sumrate,zf_sinr = loss.tf_loss_sumrate(CSI,tf.transpose(zf_matrix,perm=[0,2,1]))
            #sigma_doppler = loss.tf
            #steering_hermite = tf.transpose(tf.math.conj(steering_vector_this))

            #pathloss=loss.tf_Path_loss(input[:,-1,:,0])
            #pathloss = tf.expand_dims(pathloss, axis=1)
            #pathloss = tf.broadcast_to(pathloss, tf.shape(steering_hermite))
            #CSI = tf.multiply(tf.cast(tf.multiply(G, pathloss),dtype=tf.complex128), steering_hermite)
            #zf_beamformer = loss.tf_zero_forcing(CSI)
            #loss_MSE = loss.tf_matrix_mse(zf_beamformer,precoding_matrix)



            # steering_vector = [loss.calculate_steer_vector(predict_theta_list[v] for v in range(config_parameter.num_vehicle)
            sum_rate_this,sinr = loss.tf_loss_sumrate(CSI, precoding_matrix)
            mean_sinr = tf.reduce_mean(sinr,axis=1,keepdims=True)
            shape = tf.shape(sinr)
            sum_rate_this = tf.cast(sum_rate_this, tf.float32)
            mean_sinr = tf.tile(mean_sinr,[1,shape[1]])
            mse_sinr = tf.reduce_mean(tf.square(sinr-mean_sinr),axis=1)
            mse_sinr = tf.cast(mse_sinr, tf.float32)
            coeffi = tf.constant(0.05,tf.float32)
            sum_rate_this = tf.cast(sum_rate_this, tf.float32)
            zf_sumrate = tf.cast(zf_sumrate, tf.float32)
            batch_size = tf.cast(batch_size, tf.float32)
            communication_loss = tf.reduce_sum(- sum_rate_this) / batch_size
            #communication_loss = tf.reduce_sum(mse_sinr*coeffi-sum_rate_this)/batch_size
            #communication_loss = -sum_rate_this+loss_MSE/(tf.stop_gradient(loss_MSE/(sum_rate_this)))
            #communication_loss = tf.reduce_sum(zf_sumrate-sum_rate_this)/batch_size
            #beta = tf.complex(input[:,:,6*antenna_size:7*antenna_size,0], input[:,:,7*antenna_size:8*antenna_size,0])
            random_precoding = loss.random_beamforming()
            random_sumrate,random_sinr = loss.tf_loss_sumrate(CSI,random_precoding)
            random_sumrate = tf.cast(random_sumrate, tf.float32)
            communication_random = tf.reduce_sum(-random_sumrate) / batch_size
            Sigma_time_delay_random = loss.tf_sigma_delay_square(steering_vector_this_o,random_precoding,beta)
            Sigma_time_delay = loss.tf_sigma_delay_square(steering_vector_this_o,precoding_matrix,beta)
            Sigma_time_delay_zf = loss.tf_sigma_delay_square(steering_vector_this_o,tf.transpose(zf_matrix,perm=[0,2,1]),beta)
            #Sigma_doppler = loss.tf_sigma_doppler_square(steering_vector_this,precoding_matrix,beta)
            CRB_random = loss.tf_CRB_distance(Sigma_time_delay_random)
            CRB_d = loss.tf_CRB_distance(Sigma_time_delay)
            CRB_d_zf = loss.tf_CRB_distance(Sigma_time_delay_zf)
            #CRB_d = 0.1
            #theta = tf.reduce_mean(input[:,:,8*antenna_size:9*antenna_size,0])
            CRB_angle =0
            #CRB_angle = loss.tf_CRB_angle(beta,precoding_matrix,theta)
            crb_combined_loss = CRB_d*1 +CRB_angle*10e3
            #CRB_d = loss.tf_CRB_distance()
            #communication_loss = communication_loss/input.shape[0]
            precoding_matrix = tf.cast(precoding_matrix, tf.complex128)
            crb_combined_loss = tf.cast(crb_combined_loss, tf.float32)
            #crb_combined_loss = -1000/(tf.reduce_sum(crb_combined_loss)/(batch_size*num_vehicle))
            crb_combined_loss = tf.reduce_sum(crb_combined_loss) / (batch_size * num_vehicle)
            combined_loss = tf.reduce_sum(crb_combined_loss -sum_rate_this,axis=0)/batch_size
            #combined_loss = 1e15*mse_value
            #crb_loss =

        #communication_loss = #communication_loss = tf.math.divide(1.0, sum_rate_this)
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
        
        '''
        optimizer_1.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        return communication_loss,output,CSI,gradients,tf.reduce_sum(CRB_d,axis=0)/batch_size


    crb_d_sum_list = []  # the crb distance sum at all timepoints in this list
    crb_angle_sum_list = []  # the crb angle sum at all timepoints in this list

    sum_rate_list_reciprocal = []  # the sum rate at all timepoints in this list
    sum_rate_list = []
    angle, distance = loss.load_data()
    #input_whole = loss.Conversion2input_small(angle, distance)
    input_whole = loss.Conversion2input_small2(angle, distance)
    input_whole = np.expand_dims(input_whole, axis=3)
    #normalized_data = (data - np.mean(data)) / np.std(data)

    #input_mean = np.mean(input_whole, axis=0)
    #input_var = np.var(input_whole, axis=0)
    #input_whole_norm = (input_whole - input_mean)/
    tf_dataset = tf.data.Dataset.from_tensor_slices(input_whole)
    tf_dataset = tf_dataset.batch(config_parameter.batch_size)
    #tf_dataset = tf_dataset.map(lambda x: tf.expand_dims(x, axis=1))
    #tf_dataset = tf.expand_dims(tf_dataset, axis=0)
    #dataset = tf.transpose(tf_dataset, perm=[1, 0, 2, 3])
    #profiler.start()
    for iter in range(0, config_parameter.iters):
        #iter += 7
        if iter < 4:
            #optimizer_1 = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
            optimizer_1 = tf.keras.optimizers.Adam(learning_rate=0.002, beta_1=0.9, beta_2=0.99)
        #optimizer_1 = tf.keras.optimizers.Adam(learning_rate=0.003, beta_1=0.91, beta_2=0.99)
            #optimizer_1 = tf.keras.optimizers.Adagrad(learning_rate=0.003)
        elif iter <6:
            optimizer_1 = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.99)
            #optimizer_1 = tf.keras.optimizers.RMSprop(learning_rate=0.00005, rho=0.9)
        else:
            optimizer_1 = tf.keras.optimizers.Adam(learning_rate=0.0003, beta_1=0.9, beta_2=0.99)
            #optimizer_1 = tf.keras.optimizers.RMSprop(learning_rate=0.00003, rho=0.9)
        print(iter)
        tf_dataset = tf_dataset.shuffle(9600)
        #tf_dataset = tf_dataset.batch(config_parameter.batch_size)
        for batch in tf_dataset:
            crb_d_this = []
            communication_loss_list =[]
            print(tf.shape(batch))
            input_single = batch
            communication_loss, output, CSI, gradients,crb_d = train_step(input_single)

            print("Epoch: {}/{},loss: {}".format(iter + 1, config_parameter.iters,
                                                           communication_loss.numpy()))

            file_path = "precoding_matrix.txt"
            crb_d_this.append(crb_d)
            communication_loss_list.append(communication_loss)
            with open(file_path, "w") as file:
                file.write("output")
                file.write(str(output.numpy()) + "\n")
                file.write("CRB_d")
                file.write(str(crb_d_this) + "\n")
                #file.write()
                #file.write("theta")
                #file.write(str(input_single[0, 0:num_vehicle, 2 * antenna_size]) + "\n")
                #file.write("distance")
                #file.write(str(input_single[0, 0:num_vehicle, 3 * antenna_size]) + "\n")
                file.write("sumrate")
                file.write(str(communication_loss.numpy()) + "\n")
                file.write("gradients")
                file.write(str(gradients) + "\n")
        sorted_data = tf.sort(communication_loss_list, axis=0)

        # 计算中位数索引
        n = tf.shape(sorted_data)[0]
        median_index = n // 2

        # 获取中位数值
        communication_loss_median = sorted_data[median_index]
        crb_d_sum_list.append(tf.math.reduce_mean(crb_d_this,axis=0))
        sum_rate_list.append(communication_loss_median)
        timestep = list(range(1, len(sum_rate_list) + 1))
        plt.plot(timestep, sum_rate_list, 'b-o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss vs Epoch')
        plt.grid(True)
        plt.show()
        #plt.plot(timestep, crb_d_sum_list, 'b-o')
        #plt.xlabel('Epoch')
        #plt.ylabel('Loss')
        #plt.title('Loss vs Epoch')
        #plt.grid(True)
        #plt.show()
        # tf.saved_model.save(model, 'Keras_models/new_model')
        model.save_weights(filepath='Keras_models_hybrid/new_model', save_format='tf')
        '''checkpointer = ModelCheckpoint(filepath="Keras_models/weights.{epoch:02d}-{val_accuracy:.2f}.hdf5",
                                               monitor='val_accuracy',
                                               save_weights_only=False, period=1, verbose=1, save_best_only=False)'''
        # tf.saved_model.save(model, )
    model.save_weights(filepath='Keras_models_hybrid/new_model', save_format='tf')
    #profiler.stop()


    #profiler.print_stats()
'''
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

        #input_whole = loss.generate_random_sample()


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
        '''
'''
        checkpointer = ModelCheckpoint(filepath="Keras_models/weights.{epoch:02d}-{val_accuracy:.2f}.hdf5",
                                               monitor='val_accuracy',
                                               save_weights_only=False, period=1, verbose=1, save_best_only=False)
        #tf.saved_model.save(model, )
    model.save_weights(filepath='Keras_models/new_model', save_format='tf')
    '''