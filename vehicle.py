
import numpy as np
from math import pi,sqrt,cos
class Vehicle:
    def __init__(self,index):
        self.real_location == 0
        self.real_speed == 0
        self.DistanceToRsu==0
        self.index = index
    def Echo2RSU(self,time_delay,doppler_frequncy,theta,time):

        Reflection_coefficient = RCS / (2 * dist[self.index])
        exp_doppler= np.exp(j*2*pi*t*doppler_frequency)
        phase = np.exp(complex(0,2*pi*time_delay*time))
        signal_umformiert = self.signal_umform(singal,RSU.precoding_matrix,t-time_delay)
        for n1 in range(0,Nt):
            transmit_steering = transmit_steering.append(np.exp(complex(-pi*n1*cos(theta))
        for n2 in range(0,Nr):
            receive_steering = receive_steering.append(np.exp(-j * pi * n2 * cos(theta)))
        transmit_steering = transmit_steering/sqrt(Nt)
        receive_steering = receive_steering/sqrt(Nr)
        Attenna_Gain= sqrt(Nt*Nr)

        noise = CSCG noise
        #how to represent this CSCG noise
        echo = Attenna_Gain*Reflection_Coefficient*exp_doppler*transmit_steering.T.dot(signal_umformiert)  + noise
        return echo
    def signal_umform(self,RSU.precoding_matrix,t):
        signal1 = self.signal(t)
        return signal1.dot(RSU.precoding_matrix)

    def ReceivedSignalfromRSU(self):
        gain = sqrt(Nt)
        exp_doppler = np.exp(j * 2 * pi * t * doppler_frequency)
        pathloss = alpha*((distance/d0)**coefficient_loss)
        for n1 in range(0,Nt):
            transmit_steering = transmit_steering.append(np.exp(-j*pi*n1*cos(theta)))
        transmit_steering = transmit_steering / sqrt(Nt)
        w_k= precoding_matrix[:,index]
        signal1 = self.signal_umform(w_k)
        ReceivedSignal =




    def signal(self,t):

        return signal1
    def move(self,timeslot):
        self.real_location= timeslot*self.real_speed
        self.DistanceToRsu = sqrt(self.real_location**2 + 500*500)
        return self.real_location self.DistanceToRsu



    def link_sinr(self,alpha):
        pathloss = alpha * ((distance / d0) ** coefficient_loss)
        gain = sqrt(Nt)
        w_k = precoding_matrix[:, index]
        for n1 in range(0, Nt):
            transmit_steering = transmit_steering.append(np.exp(-j * pi * n1 * cos(theta)))
        transmit_steering = transmit_steering / sqrt(Nt)
        for i in range(0,count_vehicle):
            w_k = precoding_matrix[:, index]
            w_k = precoding_matrix[:, i]
            link = abs(gain*pathloss*transmit_steering*w_k)

            link = link**2
            if i == index:
                self.link_own = link
            else:
                self.link_inter += link

        sinr = self.link_own/(self.link_inter+deltaK_Square)  #deltaK_square is set manually by himself
        return sinr


