from math import dist
import re
import numpy as np
from numpy import math as m
from pyparsing import Dict
import scipy.stats as s
import random

np.random.seed(123)
c = 3E8

class MBSChannel:
	def __init__(self, alpha):
		self.name = 'MBS'
		self.frequency = 2.4
		self.bandwidth = 20E6
		self.alpha = alpha
		self.shadow_mean = 0
		self.shadow_std = 4
		self.fading_mu = 3
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
		return (np.array(power_transmitted)[:,None] + gain[:,None] - path_loss + bias[:,None] + fading)


class mmWaveChannel:
	def __init__(self, prob_LOS, prob_NLOS):
		self.name = 'mmWave'
		self.frequency = 73 # GHz
		self.bandwidth = 2E9
		self.alphaLOS = 2
		self.alphaNLOS = 3.3
		self.shadow_L_mean = 0
		self.shadow_N_mean = 0
		self.shadow_L_std = 5.2
		self.shadow_N_std = 7.2
		self.fading_mu = 4
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
		return (power_transmitted[:,None] + gain[:,None] - path_loss + bias[:,None] + fading)


class THzChannel:
	def __init__(self):
		self.name = 'THz'
		self.frequency = 3E11
		self.bandwidth = 10E9
		self.fading_mu = 5.2
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
		return (power_transmitted[:,None] + gain[:,None] - path_loss + bias[:,None] + fading)


class BaseStation:
	def __init__(self, power_transmitted, gain, bias, bandwidth, channel, x, y):
		self.power_transmitted = power_transmitted
		self.gain = gain
		self.bias = bias
		self.bandwidth = bandwidth
		self.channel = channel
		self.x_coord = x
		self.y_coord = y
		self.associated_UE = 0

class UserEquipment:
	def __init__(self, received_power, bandwidth, x, y):
		self.base_station = None
		self.received_power = received_power
		self.bandwidth = bandwidth
		self.data_rate = 0
		self.x_coord = x
		self.y_coord = y

class Environment:
	def __init__(self, alpha, prob_LOS, prob_NLOS, area, lambdas: Dict):
		self.channels = {'MBS': MBSChannel(alpha),
						 'mmWave': mmWaveChannel(prob_LOS, prob_NLOS),
						 'THz': THzChannel()}
		self.BS = {'MBS': [],
				   'mmWave': [],
				   'THz': []}
		#self.UE = {'MBS': [],
		#		   'mmWave': [],
		#		   'THz': [],
		#		   'Init': []}
		self.UE = []
		# Wrapper functions to get some D (D for data)
		self.n_BS = {'MBS': lambda: len(self.BS['MBS']),
					 'mmWave': lambda: len(self.BS['mmWave']),
					 'THz': lambda: len(self.BS['THz'])}
		#self.n_UE = {'MBS': lambda: len(self.UE['MBS']),
		#			 'mmWave': lambda: len(self.UE['mmWave']),
		#			 'THz': lambda: len(self.UE['THz']),
		#			 'Init': lambda: len(self.UE['Init'])}
		self.n_UE = lambda: len(self.UE)
		self.transmit_powers = {'MBS': lambda: [bs.power_transmitted for bs in self.BS['MBS']],
								'mmWave': lambda: [bs.power_transmitted for bs in self.BS['mmWave']],
								'THz': lambda: [bs.power_transmitted for bs in self.BS['THz']]}
		self.area = area
		self.lambdas = lambdas
		#self.lambda_mmWave = self.lambda_MBS*2
		#self.lambda_THz = self.lambda_MBS*6
		#self.lambda_UE = 0.0012 # Should give around 30 users

		
	# State change is triggered by a change in LOS/NLOS Probabilites or alpha for MBS channel
	def renew_channel(self, alpha, prob_LOS, prob_NLOS):
		self.channels['MBS'] = MBSChannel(alpha)
		self.channels['mmWave'] = mmWaveChannel(prob_LOS, prob_NLOS)
	
	def add_UE(self):
		n_UE = s.poisson(self.lambdas['UE']).rvs(size=(1,1))
		n_UE = n_UE[0,0]
		if n_UE == 0:
			n_UE = 1
		for i in range(n_UE):
			x = s.norm.rvs((1,1))[0]
			y = s.norm.rvs((1,1))[0]
			# Initializing a UE with a received power of -100dBm and a bandwidth of 1Hz
			self.UE.append(UserEquipment(-100, 1, x, y))
	

	def add_BS(self):
		n_BS = {}
		transmit_power = {'MBS': 20, 'mmWave': 21, 'THz': 23}
		gain = {'MBS': 0, 'mmWave': 20, 'THz': 24}
		bias = {'MBS': 0, 'mmWave': 5, 'THz': 7}

		for tier in self.lambdas:
			n_BS[tier] = (s.poisson(self.lambdas[tier]).rvs(size=(1,1)))[0,0]
			if n_BS[tier] == 0:
				n_BS[tier] = 1

		for tier in n_BS:
			bandwidth_per_BS = self.channels[tier].bandwidth / n_BS[tier]
			for i in range(n_BS[tier]):
				x = s.norm.rvs((1,1))[0]
				y = s.norm.rvs((1,1))[0]
				self.BS[tier].append(BaseStation(transmit_power[tier], gain[tier], bias[tier], bandwidth_per_BS, self.channels[tier],x,y))


	def reset(self):
		pass
		#self.UE = []
		#self.UE_MBS = []
		#self.UE_mmWave = []
		#self.UE_THz = []
		#self.BS_MBS = []
		#self.BS_mmWave = []
		#self.BS_THz = []

		#self.add_UE()
		#self.add_BS()		

		#transmit_powers = [BS.power_transmitted for BS in self.BS_MBS]
		#return transmit_powers

	def step(self, prev_state):
		#! One step: - Calculate received powers, associate, get DR, get DR Coverage, get PE
		#! Actions: Values of optimization variables -> Transmit powers of BSs
		#! AGENTS: BASE STATIONS -> Maximize DR Coverage and Power Efficiency while minimizing Transmit powers
		#! Rewards: Dict with a list for each tier
		# Unpacking prev state
		#* prev_transmit_power is a dict
		prev_transmit_power, prev_received_power, prev_dr_coverage, prev_pe = prev_state

		rewards = {}
		position_BS = {}
		gains = {}
		bias = {}
		transmit_powers = {}
		current_avg_transmit_power = {}
		prev_avg_transmit_power = {}
		fading = {}
		path_loss = {}
		received_powers = {}
		best_BS = {} # To store index of max recvd power for each tier. index will be used to figure out 
								 # the BS giving max power in each tier. 

		position_UE = (np.array([UE.x_coord for UE in self.UE]), np.array([UE.y_coord for UE in self.UE]))
		for tier in ['MBS', 'mmWave', 'THz']:
			rewards[tier] = np.zeros((self.n_BS[tier],1))
			position_BS[tier] = (np.array([BS.x_coord for BS in self.BS[tier]]), np.array([BS.y_coord for BS in self.BS[tier]]))
			gains[tier] = np.array([BS.gain for BS in self.BS[tier]])
			bias[tier] = np.array([BS.bias for BS in self.BS[tier]])
			transmit_powers[tier] = [BS.power_transmitted for BS in self.BS[tier]]
			current_avg_transmit_power[tier] = np.nanmean(transmit_powers[tier])
			prev_avg_transmit_power[tier] = np.nanmean(prev_transmit_power[tier])
			fading[tier] = s.nakagami(self.channels[tier].fading_mu).rvs((self.n_BS[tier], self.n_UE))
			path_loss[tier] = self.channels[tier].get_path_loss(position_BS[tier], position_UE)

			#* received_powers[tier]: n_BS[tier] x n_UE matrix
			received_powers[tier] = self.channels[tier].get_received_power(
									transmit_powers[tier],
									gains[tier],
									fading[tier],
									path_loss[tier],
									bias[tier]
			)
			#* received_powers_max[tier] = 1 x n_UE vector
			best_BS[tier] = np.argmax(received_powers[tier], axis=0)
			
		# Associate UE with BS
		best_BS_ue = {}
		for idx in range(self.n_UE):
			# all this spaghetti to figure out the best BS
			best_BS_ue['MBS'] = best_BS['MBS'][idx]
			best_BS_ue['mmWave'] = best_BS['mmWave'][idx]
			best_BS_ue['THz'] = best_BS['THz'][idx]
			best_BS_tier = max(zip(best_BS_ue.values(), best_BS_ue.keys()))[1]
			BS_number = best_BS[tier][idx]
			self.UE[idx].base_station = self.BS[best_BS_tier][BS_number]
			self.BS[best_BS_tier][BS_number].associated_UE += 1

			# now the powers
			self.UE[idx].received_power = (received_powers[best_BS_tier])[BS_number][idx]

		# need a new loop for bandwidth
		for idx in range(self.n_UE):
			self.UE[idx].bandwidth = self.UE[idx].base_station.bandwidth / self.UE[idx].base_station.associated_UE

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










		 




