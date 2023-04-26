import numpy
# some training parameters
#imagine the vertical distance between RSU and highway is 20
iters = 10
one_iter_period = 5
train_data_period = 1
num_vehicle =  5
highway_length = 540
RSU_location = numpy.array([270,20])
Initial_location_min = 50
Initial_location_max = 100
Radar_measure_slot = 0.2
RSU_power = 1000
rou_timedelay = 0.000002
fading_coefficient = 1 + 1j
#calculation for doppler_frequency
Frequency_original = 30e9 # carrier frequency in Hz
FurtherTrain = False



#setup for metrics
rf_size = 8
antenna_size = 32
receiver_antenna_size = 1
sigma_k = 5
sigma_z = 5


#path loss parameters
d0 = 1
alpha = -70 #path_loss alpha at reference distance d0 UNIT: dB
path_loss_exponent = 2.55


#chirp signal parameters
c = 3e8  # speed of light in m/s

bandwidth = 1e6  # bandwidth of the chirp signal in Hz
pulse_duration = 10e-6  # pulse duration in seconds
R_max = 200  # maximum range in meters


