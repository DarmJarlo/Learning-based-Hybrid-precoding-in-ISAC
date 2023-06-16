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
    model = DL_method_NN_for_v2x_mod()
    #model = ResNet()
    #model = ResNetLSTMModel()
    num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar +config_parameter.num_horizoncar

    model.build(input_shape=(config_parameter.batch_size, num_vehicle,32,1))

    model.summary()
    if config_parameter.FurtherTrain ==True:
        #model = tf.saved_model.load('Keras_models/new_model')
        model.load_weights(filepath='Keras_models/new_model')
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
    optimizer_1 = tf.keras.optimizers.Adam(learning_rate=0.002)
    #optimizer_1 = tf.keras.optimizers.Adagrad(learning_rate=0.01)
    #optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)

    #optimizer_1 = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)

    '''
    optimizer_1 = tf.keras.optimizers.Adagrad(
        learning_rate=0.01,
        initial_accumulator_value=0.1,
        epsilon=1e-07,
        name="Adagrad"
    )
    '''
    model = load_model()
    @tf.function  # means not eager mode. graph mode
    def train_step(input):
        # input shape(1,10,3,32)
        with tf.GradientTape() as tape:
            tape.watch(input)
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
            #Analog_matrix, Digital_matrix = loss.tf_Output2PrecodingMatrix_rad(Output=output)
            #precoding_matrix = loss.tf_Precoding_matrix_combine(Analog_matrix, Digital_matrix)
            ZF_matrix = tf.complex(input[:, :, 2 * antenna_size:3 * antenna_size, 0],
                                   input[:, :, 3 * antenna_size:4 * antenna_size, 0])
            precoding_matrix = loss.tf_Output2digitalPrecoding(Output=output,zf_matrix=ZF_matrix)

            CSI = tf.complex(input[:,:,0:1*antenna_size,0], input[:,:,1*antenna_size:2*antenna_size,0])
            sum_rate_this = loss.tf_loss_sumrate(CSI, precoding_matrix)

            sum_rate_this = tf.cast(sum_rate_this, tf.float64)
            #zf_sumrate = tf.cast(zf_sumrate, tf.float32)
            batch_size = tf.cast(batch_size, tf.float64)
            power = tf.constant(config_parameter.power, dtype=tf.float64)
            power_error = tf.reduce_sum(tf.abs(precoding_matrix),axis= (1,2))-power
            #communication_loss = (tf.reduce_sum(power_error,axis=0)-\
             #                     tf.reduce_sum(-sum_rate_this, axis=0)) / batch_size
            communication_loss = tf.reduce_sum(-sum_rate_this, axis=0) / batch_size

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
        '''
        #clipped_gradients = [tf.clip_by_value(grad, 0, float('inf')) for grad in gradients]
        optimizer_1.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        return sum_rate_this,communication_loss,CSI,gradients,precoding_matrix,output,ZF_matrix


    sum_rate_list = []
    angle, distance = loss.load_data()
    input_whole = loss.Conversion2CSI(angle, distance)
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
    for iter in range(0, config_parameter.iters):
        tf_dataset = tf_dataset.shuffle(3200)
        #tf_dataset = tf_dataset.batch(config_parameter.batch_size)
        for batch in tf_dataset:
            print(tf.shape(batch))
            input_single = batch
            sum_rate, communication_loss, CSI, gradients,precoding_matrix,output,ZF_matrix= train_step(input_single)

            print("Epoch: {}/{},loss: {}".format(iter + 1, config_parameter.iters,
                                                           communication_loss
                                                           ))
            file_path = "losssum_rate.txt"
            #for i, (grad, var) in enumerate(zip(gradients, model.trainable_variables)):
             #   print("Layer:", i + 1)
              #  print("Gradient shape:", grad.shape)
               # print("Grad:", grad.numpy())
               # print("Variable shape:", var.shape)
                #print("==============================")
            with open(file_path, "w") as file:
                file.write("Output")
                file.write(str(output.numpy()) + "\n")
                file.write("ZF_matrix")
                file.write(str(ZF_matrix.numpy()) + "\n")
                file.write("sum_rate")


                file.write(str(sum_rate.numpy()) + "\n")
                file.write("CSI")
                file.write(str(CSI.numpy()) + "\n")
                file.write("precoding_matrix")
                file.write(str(precoding_matrix.numpy()) + "\n")

            file_path1 = "gradients.txt"
            with open(file_path1, "a") as file1:
                file1.write("gradients")
                file1.write(str(gradients) + "\n")


        sum_rate_list.append(communication_loss)
        timestep = list(range(1, len(sum_rate_list) + 1))
        plt.plot(timestep, sum_rate_list, 'b-o')
        plt.xlabel('Timestep')
        plt.ylabel('Loss')
        plt.title('Loss vs Timestep')
        plt.grid(True)
        plt.show()
        # tf.saved_model.save(model, 'Keras_models/new_model')
        model.save_weights(filepath='Keras_models/new_model', save_format='tf')
        '''checkpointer = ModelCheckpoint(filepath="Keras_models/weights.{epoch:02d}-{val_accuracy:.2f}.hdf5",
                                               monitor='val_accuracy',
                                               save_weights_only=False, period=1, verbose=1, save_best_only=False)'''
        # tf.saved_model.save(model, )
    model.save_weights(filepath='Keras_models/new_model', save_format='tf')
