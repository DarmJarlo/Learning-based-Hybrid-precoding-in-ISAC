# some training parameters
iters = 10
rf_size = 8
antenna_size = 32
num_vehicle =  10
highway_length = 540
RSU_location_x = 270
RSU_location_y = 20
Initial_location_min = 50
Initial_location_max = 100
Radar_measure_slot = 0.2

# some training parameters
EPOCHS = 10
BATCH_SIZE = 32
NUM_CLASSES = 4
image_height = 90
image_width = 100
channels = 1
FurtherTrain = False
save_model_dir = "saved_model/model"
dataset_dir = "dataset/"
train_dir = dataset_dir + "train"
valid_dir = dataset_dir + "valid"
test_dir = dataset_dir + "test"
Oned = True
model = "resnet50"


