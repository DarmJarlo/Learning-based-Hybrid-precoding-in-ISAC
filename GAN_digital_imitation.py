'''
this file is planned to generate hybrid precoding to
imitate digital precoding with GAN'''


import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


# Define the GAN model
import config_parameter


def build_gan_model():
    input_shape = (1, 10, config_parameter.num_vehicle, 2)
    output_shape1 = (16, 8)
    output_shape2 = (8, 4)

    # Generator
    generator_input = keras.Input(shape=input_shape)
    x = layers.Flatten()(generator_input)
    x = layers.Dense(128)(x)
    x = layers.Dense(64)(x)
    generator_output = layers.Dense(np.prod(output_shape1) + np.prod(output_shape2))(x)

    # Discriminator
    discriminator_input = keras.Input(shape=output_shape1)
    x = layers.Flatten()(discriminator_input)
    discriminator_output = layers.Dense(1, activation='sigmoid')(x)

    # Build and compile the GAN model
    gan_model = keras.Model(inputs=generator_input, outputs=[generator_output, discriminator_output])
    gan_model.compile(optimizer='adam', loss=['mse', 'binary_crossentropy'])

    return gan_model


# Prepare the training data
input_data = np.random.random((1000, 1, 10, 5, 2))
target_data = np.random.random((1000, 16, 4))

# Normalize the input data
input_data = input_data / np.max(input_data)

# Create an instance of the GAN model
gan_model = build_gan_model()

# Train the GAN model
gan_model.fit(input_data, [target_data.reshape(-1, np.prod(target_data.shape[1:])),
                           target_data.reshape(-1, np.prod(target_data.shape[1:]))], batch_size=32, epochs=100)

# Generate output similar to target data
latent_input = np.random.random((1, 1, 10, 5, 2))
generated_output, _ = gan_model.predict(latent_input)

# Reshape the generated output into two matrices
matrix1 = generated_output[:, :np.prod(output_shape1)].reshape((16, 8))
matrix2 = generated_output[:, np.prod(output_shape1):].reshape((8, 4))

# Multiply the matrices to imitate target data
result_matrix = np.matmul(matrix1, matrix2)

# Print the result matrix
print("Result Matrix:")
print(result_matrix)
