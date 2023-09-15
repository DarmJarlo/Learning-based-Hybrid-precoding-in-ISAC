import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

digital = False
#mode = "V2V"
mode = "V2I"
# some training parameters
#imagine the vertical distance between RSU and highway is 20

iters = 1000
one_iter_period = 0.4#s
train_data_period = 1 #s
num_vehicle =  4
batch_size = 32
matched_filtering_gain =10


#parameter for v2v
################################################################3
num_uppercar =2
num_lowercar =2
num_horizoncar = 0
observer_car_init_loca = np.array([0,0])
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
lowerspeed_low = -10
lowerspeed_high = -5
upperspeed_low = 5
upperspeed_high = 10
horizonspeed_low = 5
horizonspeed_high = 10
#########################################################



highway_length = 540
RSU_location = np.array([0,-1000])

Radar_measure_slot = 0.01 #s
length_echo = 0.005  # length of echo ms
power = 100

fading_coefficient = 0.5 + 0.5*1j
"""這裏越小，zf的crb越大，因爲zf的幹擾永遠爲0，這個參數越小的話，sinr數量級越小，zf的crb越大"""
"""同樣的，pathloss越小，zf的sumrate越小，因爲sinr的數量級越小，zf的sumrate越小"""
#calculation for doppler_frequency
Frequency_original = 30e9 # carrier frequency in Hz
FurtherTrain = True

#these are for the simulation test
speed_low = 5
speed_high = 10

Initial_location_min = np.array([1000,650,-450,-850])
Initial_location_max = np.array([1050,700,-400,-800])
print(Initial_location_min)
print(Initial_location_max)
#[ 1089.81379201   487.37954435 -1089.81379201 -2064.57288071]
#[ 2064.57288071  1089.81379201  -487.37954435 -1089.81379201]
Initial_uppercar_y = 1000
Initial_lowercar_y = 900
#Initial_location1_min = 100*np.sin(0.4*np.pi)
#Initial_location1_min = 100*np.cos(0.3*np.pi)
#print(Initial_location1_min)#58
#Initial_location1_max = 100*np.cos(0.35*np.pi)
#print(Initial_location1_max)45



train_speed_low = 18
train_speed_high = 25
train_initial_location_min = 30
train_initial_location_max = 120

#setup for metrics
rf_size = 6
antenna_size = 16
vehicle_antenna_size = 16
receiver_antenna_size = 1
sigma_k = 1e-7
sigma_z = 1e-7


#path loss parameters

d0 = 100
alpha = 1e-3#path_loss alpha at reference distance d0 UNIT: dB
#alpha = 1e-6#path_loss alpha at reference distance d0 UNIT: dB
path_loss_exponent = -2.55


#chirp signal parameters
c = 3e8  # speed of light in m/s
sampling_rate = 1000

bandwidth = 1e6  # bandwidth of the chirp signal in Hz
pulse_duration = 10e-6  # pulse duration in seconds
R_max = 200  # maximum range in meters
Signal_noise_power = 1e-6#noise for echo signal
sigma_rk = Signal_noise_power
rou_timedelay = 2e-8
rou_dopplershift = 2e-8




#loss_mode = "Upper_sum_rate"   # three mode
loss_mode = "lower_bound_crb"

#loss_mode = "combined_loss"


