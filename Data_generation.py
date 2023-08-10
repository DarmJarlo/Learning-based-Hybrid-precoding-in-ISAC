import random
import tensorflow as tf
import numpy as np
import config_parameter
import evaluation
if config_parameter.mode == "V2I":
    antenna_size = config_parameter.antenna_size
    num_vehicle = config_parameter.num_vehicle
elif config_parameter.mode == "V2V":
    antenna_size = config_parameter.vehicle_antenna_size
    num_vehicle = config_parameter.num_uppercar + config_parameter.num_lowercar + config_parameter.num_horizoncar
# Set the number of samples and batch size
num_samples = 9600
batch_size = 32

# Generate random samples
angles_1 = np.random.uniform(0.2*np.pi, 0.3*np.pi, size=(num_samples, 1))
angles_2 = np.random.uniform(0.3*np.pi, 0.4*np.pi, size=(num_samples, 1))
angles_3 = np.random.uniform(0.6*np.pi, 0.7*np.pi, size=(num_samples, 1))
angles_4 = np.random.uniform(0.7*np.pi, 0.8*np.pi, size=(num_samples, 1))
angles = np.concatenate((angles_1, angles_2, angles_3,angles_4), axis=1)
distances = np.random.uniform(1500, 2000, size=(num_samples, num_vehicle))
angles = np.random.uniform(0.2*np.pi, 0.8*np.pi, size=(num_samples, num_vehicle))
# Combine angles and distances into a single array
angles = np.sort(angles, axis=1)
data = np.concatenate((angles, distances), axis=1)

distances = None
angles = None
for i in range(96):
    distance,angle = evaluation.generate_input_v2i()
    angle = angle.T
    distance = distance.T
    if distances is not None:
        distances = np.concatenate((distances, distance), axis=0)
        angles = np.concatenate((angles, angle), axis=0)
    else:
        distances = distance
        angles = angle



data = np.concatenate((np.array(angles), np.array(distances)), axis=1)
print(data.shape)
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
for batch in tf_dataset.take(10):  # Print 5 batches
    angles_batch = batch[:, :num_vehicle].numpy().tolist()
    distances_batch = batch[:, num_vehicle:].numpy().tolist()
    print("Angles:", angles_batch)
    print("Distances:", distances_batch)
