from math import cos, sqrt
import numpy as np
from scipy.optimize import fmin_bfgs
"This is the loss of every link quality"
Nt=8
Nr=8
Antenna_Gain = sqrt(Nt*Nr)
def calculate_steer_vector(Nt):
    steering_vector = []
    for n1 in range(0, Nt):
        steering_vector = steering_vector.append(np.exp(complex(-pi * n1 * cos(theta))
                                                            for n2 in range(0, Nr):
        receive_steering = receive_steering.append(np.exp(-j * pi * n2 * cos(theta)))
        transmit_steering = transmit_steering / sqrt(Nt)
        receive_steering = receive_steering / sqrt(Nr)
        Attenna_Gain = sqrt(Nt * Nr)
    return steering_vector

def Echo2RSU(self, time_delay, doppler_frequncy, theta, time):
    Reflection_coefficient = RCS / (2 * dist[self.index])
    exp_doppler = np.exp(j * 2 * pi * t * doppler_frequency)
    phase = np.exp(complex(0, 2 * pi * time_delay * time))
    signal_umformiert = self.signal_umform(singal, RSU.precoding_matrix, t - time_delay)
    for n1 in range(0, Nt):
        transmit_steering = transmit_steering.append(np.exp(complex(-pi * n1 * cos(theta))
                                                            for n2 in range(0, Nr):
        receive_steering = receive_steering.append(np.exp(-j * pi * n2 * cos(theta)))
        transmit_steering = transmit_steering / sqrt(Nt)
        receive_steering = receive_steering / sqrt(Nr)
        Attenna_Gain = sqrt(Nt * Nr)

        noise = CSCG
        noise
        # how to represent this CSCG noise
        echo = Attenna_Gain * Reflection_Coefficient * exp_doppler * transmit_steering.T.dot(signal_umformiert) + noise
    return echo
def Matched_filter(echo,Nt,sigma):
    Noise_Zkn = np.random.normal(0,sigma)
    for n1 in range(0, Nt):
        transmit_steering = transmit_steering.append(np.exp(-j * pi * n1 * cos(theta)))
    transmit_steering = transmit_steering / sqrt(Nt)

def link_sinr(alpha,precoding_matrix):
    pathloss = alpha * ((distance / d0) ** coefficient_loss)
    gain = sqrt(Nt)
    w_k = precoding_matrix[:, index]
    for n1 in range(0, Nt):
        transmit_steering = transmit_steering.append(np.exp(-j * pi * n1 * cos(theta)))
    transmit_steering = transmit_steering / sqrt(Nt)
    for i in range(0, count_vehicle):
        w_k = precoding_matrix[:, index]
        w_k = precoding_matrix[:, i]
        link = abs(gain * pathloss * transmit_steering * w_k)

        link = link ** 2
        if i == index:
            self.link_own = link
        else:
            self.link_inter += link

    sinr = self.link_own / (self.link_inter + deltaK_Square)  # deltaK_square is set manually by himself
    return sinr



def loss_Sumrate():
    for vehicle in vehicles:
        Sumrate += np.log2(1+vehicle.sinr)

    return Sum_rate


def loss_CRB():

