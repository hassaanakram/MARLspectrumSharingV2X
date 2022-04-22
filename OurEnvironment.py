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
		# Positions should be a tuple (x,y) with each x,y being numpy arrays of Kx1 or Nx1 where K is the 
		# number of UE and N is the number of BS
		# BS_x: Nx1, BS_y: Nx1 vectors
		# UE_x = UE_y = Nx1 vectors
		BS_x, BS_y = position_BS
		UE_x, UE_y = position_UE
		num_BS = BS_x.shape[0]
		num_UE = UE_x.shape[0]
		# shadowing: num_BSxnum_UE (NxK) matrix
		shadowing = s.lognorm.rvs(s=self.shadow_std, size=(num_BS, num_UE))
		# m.pow(BS_x-UE_x,2) = m.pow(BS_y-UE_y,2): NxK vectors
		# distance: NxK matrix: Distance of each BS from each user
		distance = m.sqrt(m.pow(BS_x[:, None]-UE_x,2) - m.pow(BS_y[:, None]-UE_y,2))
		# path_loss: NxK matrix (hopefully)
		path_loss = 10*m.log10((4*m.pi*self.frequency)/c) + 10*self.beta*m.log10(distance)+shadowing

		return path_loss

	def get_received_power(self, power_transmitted, gain, fading, path_loss, bias):
		# power_transmitted: Nx1
		# gain: Nx1
		# bias: Nx1
		# fading: NxK
		# path_loss: NxK
		return (power_transmitted[:,None] + gain[:,None] - path_loss + bias[:,None] + fading)


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
		num_UE = UE_x.shape[0]
		shadowing_LOS = s.lognorm.rvs(s=self.shadow_std_LOS, size=(num_BS,num_UE))
		shadowing_NLOS = s.lognorm.rvs(s=self.shadow_std_NLOS, size=(num_BS,num_UE))
		distance = m.sqrt(m.pow(BS_x[:, None]-UE_x,2) - m.pow(BS_y[:, None]-UE_y,2))
		path_loss_LOS = self.fixed_path_loss + 10*self.alphaLOS*m.log10(distance) + shadowing_LOS
		path_loss_NLOS = self.fixed_path_loss + 10*self.alphaNLOS*m.log10(distance) + shadowing_NLOS

		# net path loss is the weighted sum of LOS and NLOS path losses.
		path_loss = self.prob_LOS*path_loss_LOS + self.prob_NLOS*path_loss_NLOS
		return path_loss

	def get_received_power(self, power_transmitted, gain, fading, path_loss, bias):
		return (power_transmitted[:,None] + gain[:,None] - path_loss + bias[:,None] + fading)


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
		
		distance = m.sqrt(m.pow(BS_x[:, None]-UE_x,2) - m.pow(BS_y[:, None]-UE_y,2))
		path_loss = self.path_loss_absorption(distance) + self.path_loss_spread(distance)

		return path_loss

	def get_received_power(self, power_transmitted, gain, fading, path_loss, bias):
		return (power_transmitted[:,None] + gain[:,None] - path_loss + bias[:,None] + fading)


class BaseStation:
	def __init__(self, power_transmitted, gain, channel):
		self.power_transmitted = power_transmitted
		self.gain = gain
		self.channel = channel

class UserEquipment:
	def __init__(self, channel):
		self.channel = channel

class Environment:
	def __init__(self):
		self.MBSChannel = MBSChannel()
		self.mmWaveChannel = mmWaveChannel()
		self.THzChannel = THzChannel()
		self.BS_MBS = []
		self.BS_mmWave = []
		self.BS_THz = []
		self.UE_MBS = []
		self.UE_mmWave = []
		self.UE_THz = []

		self.area = 500*500
		self.lambda_MBS = 4e-6
		self.lambda_mmWave = self.lambda_MBS*2
		self.lambda_THz = self.lambda_MBS*6
		self.lambda_UE = 0.0012 # Should give around 30 users

		

		 




