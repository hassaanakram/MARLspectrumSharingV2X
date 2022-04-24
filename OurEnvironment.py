from math import dist
import numpy as np
from numpy import math as m
import scipy.stats as s
import random

np.random.seed(123)
c = 3E8

class MBSChannel:
	def __init__(self, alpha):
		self.frequency = 2.4
		self.bandwidth = 20E6
		self.alpha = alpha
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
		distance = np.power(np.power(BS_x[:, None]-UE_x,2) + np.power(BS_y[:, None]-UE_y,2), 1/2)
		# path_loss: NxK matrix (hopefully)
		path_loss = 10*np.log10((4*m.pi*self.frequency)/c) + 10*self.beta*np.log10(distance)+shadowing

		return path_loss

	def get_received_power(self, power_transmitted, gain, fading, path_loss, bias):
		# power_transmitted: Nx1
		# gain: Nx1
		# bias: Nx1
		# fading: NxK
		# path_loss: NxK
		return (np.array(power_transmitted)[:,None] + np.array(gain)[:,None] - path_loss + bias + fading)


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
		distance = np.power(np.power(BS_x[:, None]-UE_x,2) + np.power(BS_y[:, None]-UE_y,2), 1/2)
		path_loss_LOS = self.fixed_path_loss + 10*self.alphaLOS*m.log10(distance) + shadowing_LOS
		path_loss_NLOS = self.fixed_path_loss + 10*self.alphaNLOS*m.log10(distance) + shadowing_NLOS

		# net path loss is the weighted sum of LOS and NLOS path losses.
		path_loss = self.prob_LOS*path_loss_LOS + self.prob_NLOS*path_loss_NLOS
		return path_loss

	def get_received_power(self, power_transmitted, gain, fading, path_loss, bias):
		return (power_transmitted[:,None] + gain[:,None] - path_loss + bias + fading)


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
		
		distance = np.power(np.power(BS_x[:, None]-UE_x,2) + np.power(BS_y[:, None]-UE_y,2), 1/2)
		path_loss = self.path_loss_absorption(distance) + self.path_loss_spread(distance)

		return path_loss

	def get_received_power(self, power_transmitted, gain, fading, path_loss, bias):
		return (power_transmitted[:,None] + gain[:,None] - path_loss + bias + fading)


class BaseStation:
	def __init__(self, power_transmitted, gain, channel, x, y):
		self.power_transmitted = power_transmitted
		self.gain = gain
		self.channel = channel
		self.x_coord = x
		self.y_coord = y

class UserEquipment:
	def __init__(self, channel, x, y):
		self.channel = channel
		self.received_power = -80
		self.x_coord = x
		self.y_coord = y

class Environment:
	def __init__(self, alpha):
		self.MBSChannel = MBSChannel(alpha)
		#self.mmWaveChannel = mmWaveChannel(prob_LOS ,prob_NLOS)
		#self.THzChannel = THzChannel()
		self.BS_MBS = []
		#self.BS_mmWave = []
		#self.BS_THz = []
		self.UE = []
		self.n_BS_MBS = 0
		self.n_UE = 0
		#self.UE_MBS = []
		#self.UE_mmWave = []
		#self.UE_THz = []
		self.transmit_powers = []
		self.area = 500*500
		self.lambda_MBS = 2
		self.lambda_UE = 30
		#self.lambda_mmWave = self.lambda_MBS*2
		#self.lambda_THz = self.lambda_MBS*6
		#self.lambda_UE = 0.0012 # Should give around 30 users

		self.MBS_powers = [40, 30, 20, 5, -100]
		#self.mmWave_powers = [43, 33, 23, 7, -100]
		#self.THz_powers = [46, 36, 26, 10, -100]

		
	# State change is triggered by a change in LOS/NLOS Probabilites
	def renew_channel(self, alpha):
		self.MBSChannel = MBSChannel(alpha)
	
	def add_UE(self):
		self.n_UE = s.poisson(self.lambda_UE).rvs(size=(1,1))
		self.n_UE = self.n_UE[0,0]
		if self.n_UE == 0:
			self.n_UE = 1
		for i in range(self.n_UE):
			x = s.norm.rvs((1,1))[0]
			y = s.norm.rvs((1,1))[0]
			self.UE.append(UserEquipment(self.MBSChannel,x,y))
	

	def add_BS(self):
		self.n_BS_MBS = s.poisson(self.lambda_MBS).rvs(size=(1,1))
		self.n_BS_MBS = self.n_BS_MBS[0,0]
		if self.n_BS_MBS == 0:
			self.n_BS_MBS = 1
		#self.n_BS_mmWave = s.poisson(self.lambda_mmWave).rvs((1,1))
		#self.n_BS_THz = s.poisson(self.lambda_THz).rvs((1,1))

		for i in range(self.n_BS_MBS):
			x = s.norm.rvs((1,1))[0]
			y = s.norm.rvs((1,1))[0]
			self.BS_MBS.append(BaseStation(60, 0, self.MBSChannel,x,y))

		#for i in range(self.n_BS_mmWave):
		#	self.BS_mmWave.append(BaseStation(-100, 7, channels[1]))

		#for i in range(self.n_BS_THz):
		#	self.BS_THz.append(BaseStation(-100, 10, channels[2]))


	def reset(self):
		self.UE = []
		#self.UE_MBS = []
		#self.UE_mmWave = []
		#self.UE_THz = []
		self.BS_MBS = []
		#self.BS_mmWave = []
		#self.BS_THz = []

		self.add_UE()
		self.add_BS()		

		transmit_powers = [BS.power_transmitted for BS in self.BS_MBS]
		return transmit_powers

	def step(self, prev_received_powers, prev_transmit_powers):
		# One step of our simulation: Calculate received powers for a 
		# given state.
		#! Actions: Values of optimization variables -> Transmit powers of BSs
		#! action: (Nx1) vector where N is the number of BS (ONLY MBS FOR NOW)
		#! AGENTS: BASE STATIONS -> MAXIMIZE GLOBAL RECEIVED POWER AT UE WHILE
		#! MINIMIZING BS TRANSMIT POWERS
		#TODO: Get recieved powers -> Compare with previous received powers stats ->
		#TODO: give rewards based on new received powers

		num_UE = len(self.UE)
		num_BS = len(self.BS_MBS)
		rewards = np.zeros((num_BS,1))

		position_BS_x = np.array([BS.x_coord for BS in self.BS_MBS])
		position_BS_y = np.array([BS.y_coord for BS in self.BS_MBS])
		position_BS = (position_BS_x, position_BS_y)
		position_UE_x = np.array([UE.x_coord for UE in self.UE])
		position_UE_y = np.array([UE.y_coord for UE in self.UE])
		position_UE = (position_UE_x, position_UE_y)

		gains = [BS.gain for BS in self.BS_MBS]
		transmit_powers = [BS.power_transmitted for BS in self.BS_MBS]
		current_avg_transmit_power = np.nanmean(transmit_powers)
		prev_avg_transmit_power = np.nanmean(prev_transmit_powers)
		fading = s.nakagami(nu=3).rvs((num_BS, num_UE))
		path_loss = self.MBSChannel.get_path_loss(position_BS, position_UE)

		#* received_powers: numBS x numUE
		received_powers = self.MBSChannel.get_received_power(
						  transmit_powers,
						  gains,
						  fading,
						  path_loss, 0
		)
		# Associate users with max recvd power BS
		received_powers = np.nanmax(received_powers, axis=0)
		for idx in range(num_UE):
			self.UE[idx].received_power = received_powers[idx]

		#! REWARD POLICY: AVERAGE RECEIVED POWER SHOULD INCREASE & AVG TRANSMIT POWER SHOULD DEC
		#! GOING WITH A GLOBAL REWARD TO PROMOTE OVERALL BETTER SYSTEM
		prev_avg_received_power = np.nanmean(prev_received_powers)
		current_avg_received_power = np.nanmean(received_powers)
		if prev_avg_received_power < current_avg_received_power:
			rewards += 1
		else:
			rewards -= 1
		
		if prev_avg_transmit_power < current_avg_transmit_power:
			rewards -= 1
		else:
			rewards += 1

		state = transmit_powers

		return rewards, state










		 




