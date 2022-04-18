from math import dist
import numpy as np
from numpy import math as m
import scipy.stats as s
import random

np.random.seed(123)
c = 3E8

class MBSChannel:
	def __init__(self):
		self.frequency = 2.4
		self.bandwidth = 20E6
		self.alpha = 0.5
		self.shadow_mean = 0
		self.shadow_std = 4
		self.beta = 3

	def get_path_loss(self, position_BS, position_UE):
		# Positions should be a tuple (x,y) with each x,y being numpy arrays of Kx1 where K is the 
		# number of either base stations or the number of user equipment
		# BS_x: Nx1, BS_y: Nx1 vectors
		# UE_x = UE_y = 1x1 scalar
		BS_x, BS_y = position_BS
		UE_x, UE_y = position_UE
		num_BS = BS_x.shape[0]
		# shadowing: num_UEx1 vector
		shadowing = s.lognorm.rvs(s=self.shadow_std, size=num_BS)
		# m.pow(BS_x-UE_x,2) = m.pow(BS_y-UE_y, 2): Nx1 vectors
		# distance: Nx1 vector: Distance of the user in question from each 
		# base station
		distance = m.sqrt(m.pow(BS_x-UE_x,2) - m.pow(BS_y-UE_y,2))
		# path_loss: Nx1 vector (hopefully)
		path_loss = 10*m.log10((4*m.pi*self.frequency)/c) + 10*self.beta*m.log10(distance)+shadowing

		return path_loss

	def get_received_power(self, power_transmitted, gain, fading, path_loss, bias):
		# power_transmitted: Nx1
		# gain: Nx1
		# bias: Nx1
		# fading: Nx1
		# path_loss: Nx1
		return (power_transmitted + gain - path_loss + bias + fading)


class mmWaveChannel:
	def __init__(self, prob_LOS, prob_NLOS):
		self.frequency = 73 # GHz
		self.bandwidth = 2E9
		self.alphaLOS = 2
		self.alphaNLOS = 3.3
		self.shadow_L_mean = 0
		self.shadow_N_mean = 0
		self.shadow_L_std = 5.2
		self.shadow_N_std = 7.2
		self.prob_LOS = prob_LOS
		self.prob_NLOS = prob_NLOS
		self.fixed_path_loss = 32.4 + 20*m.log10(self.frequency)

	def get_path_loss(self, position_BS, position_UE):
		BS_x, BS_y = position_BS
		UE_x, UE_y = position_UE
		num_BS = BS_x.shape[0]
		shadowing_LOS = s.lognorm.rvs(s=self.shadow_std_LOS, size=num_BS)
		shadowing_NLOS = s.lognorm.rvs(s=self.shadow_std_NLOS, size=num_BS)
		distance = m.sqrt(m.pow(BS_x-UE_x,2) - m.pow(BS_y-UE_y,2))
		path_loss_LOS = self.fixed_path_loss + 10*self.alphaLOS*m.log10(distance) + shadowing_LOS
		path_loss_NLOS = self.fixed_path_loss + 10*self.alphaNLOS*m.log10(distance) + shadowing_NLOS

		# net path loss is the weighted sum of LOS and NLOS path losses.
		path_loss = self.prob_LOS*path_loss_LOS + self.prob_NLOS*path_loss_NLOS
		return path_loss

	def get_received_power(self, power_transmitted, gain, fading, path_loss, bias):
		return (power_transmitted + gain - path_loss + bias + fading)


class THzChannel:
	def __init__(self):
		self.frequency = 3E11
		self.bandwidth = 10E9
		self.kF = 0.0033 # some coefficient that rn i'm too tired to read about. the units are m-1

	def path_loss_absorption(self, distance):
		return self.kF*distance*10*m.log10(m.exp(1))

	def path_loss_spread(self, distance):
		return 20*m.log10((4*m.pi*self.frequency*distance)/c)

	def get_path_loss(self, position_BS, position_UE):
		# Positions should be a tuple (x,y) with each x,y being numpy arrays of Kx1 where K is the 
		# number of either base stations or the number of user equipment
		BS_x, BS_y = position_BS
		UE_x, UE_y = position_UE
		
		distance = m.sqrt(m.pow(BS_x-UE_x,2) - m.pow(BS_y-UE_y,2))
		path_loss = self.path_loss_absorption(distance) + self.path_loss_spread(distance)

		return path_loss

	def get_received_power(self, power_transmitted, gain, fading, path_loss, bias):
		return (power_transmitted + gain - path_loss + bias + fading)