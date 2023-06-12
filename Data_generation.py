import random
import tensorflow as tf
import numpy as np
import config_parameter
if config_parameter.mode == "V2I":
    antenna_size = config_parameter.antenna_size
    num_vehicle = config_parameter.num_vehicle
elif config_parameter.mode == "V2V":
    antenna_size = config_parameter.vehicle_antenna_size
    num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar + config_parameter.num_horizoncar
# Set the number of samples and batch size
num_samples = 3200
batch_size = 32

# Generate random samples
angles_1 = np.random.uniform(0, 0.15*np.pi, size=(num_samples, 1))
angles_2 = np.random.uniform(0.15*np.pi, 0.3*np.pi, size=(num_samples, 1))
angles_3 = np.random.uniform(0.7*np.pi, 0.85*np.pi, size=(num_samples, 1))
angles_4 = np.random.uniform(0.85*np.pi, np.pi, size=(num_samples, 1))
angles = np.concatenate((angles_1, angles_2, angles_3, angles_4), axis=1)
distances = np.random.uniform(50, 200, size=(num_samples, num_vehicle))

# Combine angles and distances into a single array
data = np.concatenate((angles, distances), axis=1)

# Shuffle the data
#np.random.shuffle(data)

# Save data to a file
filename = "dataset.npy"
np.save(filename, data)

# Create a TensorFlow Dataset from the file with shuffling
tf_dataset = tf.data.Dataset.from_tensor_slices(data)
tf_dataset = tf_dataset.shuffle(num_samples)
tf_dataset = tf_dataset.batch(batch_size)

# Read 32 samples as separate lists for each batch
for batch in tf_dataset.take(5):  # Print 5 batches
    angles_batch = batch[:, :4].numpy().tolist()
    distances_batch = batch[:, 4:].numpy().tolist()
    print("Angles:", angles_batch)
    print("Distances:", distances_batch)
