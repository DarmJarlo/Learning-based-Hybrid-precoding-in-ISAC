import numpy

mode = "V2V"
#mode = "V2I"
# some training parameters
#imagine the vertical distance between RSU and highway is 20
iters = 100
one_iter_period = 30#s
train_data_period = 1 #s
num_vehicle =  5
batch_size = 16
matched_filtering_gain =10

#parameter for v2v
################################################################3
num_uppercar =1
num_lowercar =1
num_horizoncar = 0
observer_car_init_loca = numpy.array([0,0])
Initial_uppercar1_min = 50
Initial_uppercar1_max = 60
Initial_uppercar2_min = -20
Initial_uppercar2_max = -30
Initial_horizoncar1_min = 50
Initial_horizoncar1_max = 60
Initial_lowercar1_min = 40
Initial_lowercar1_max = 50
Initial_lowercar2_min = -40
Initial_lowercar2_max = -50
lowerspeed_low = -21
lowerspeed_high = -20
upperspeed_low = 20
upperspeed_high = 21
horizonspeed_low = 20
horizonspeed_high = 21
#########################################################



highway_length = 540
RSU_location = numpy.array([270,-20])
Initial_location_min = 50
Initial_location_max = 100
Radar_measure_slot = 0.1 #s
length_echo = 0.005  # length of echo ms
power = 10

fading_coefficient = 10 + 10j
#calculation for doppler_frequency
Frequency_original = 30e9 # carrier frequency in Hz
FurtherTrain = False

#these are for the simulation test
speed_low = 20
speed_high = 21

Initial_location_min = 50
Initial_location_max = 100



train_speed_low = 18
train_speed_high = 25
train_initial_location_min = 30
train_initial_location_max = 120

#setup for metrics
rf_size = 1
antenna_size = 32
vehicle_antenna_size = 8
receiver_antenna_size = 1
sigma_k = 1
sigma_z = 1


#path loss parameters
d0 = 10
alpha = 70 #path_loss alpha at reference distance d0 UNIT: dB
path_loss_exponent = -2.55


#chirp signal parameters
c = 3e8  # speed of light in m/s
sampling_rate = 1000

bandwidth = 1e6  # bandwidth of the chirp signal in Hz
pulse_duration = 10e-6  # pulse duration in seconds
R_max = 200  # maximum range in meters
Signal_noise_power = 0.1#noise for echo signal
sigma_rk = Signal_noise_power
rou_timedelay = 2e-6
rou_dopplershift = 2e-6




loss_mode = "Upper_sum_rate"   # three mode
#loss_mode = "lower_bound_crb"
#loss_mode = "combined_loss"


#below are the parameter for v2v
