import numpy as np
from numpy import math as m
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

	def get_path_loss(self, position_BS, position_UE, shadowing):
		BS_x, BS_y = position_BS
		UE_x, UE_y = position_UE

		distance = m.sqrt(m.pow(BS_x-UE_x,2) - m.pow(BS_y-UE_y,2))
		path_loss = 10*m.log10((4*m.pi*self.fMBS)/c) + 10*self.beta*m.log10(distance)+shadowing


class mmWaveChannel:
	def __init__(self):
		self.frequency = 73 # GHz
		self.bandwidth = 2E9
		self.alphaLOS = 2
		self.alphaNLOS = 3.3
		self.shadow_L_mean = 0
		self.shadow_N_mean = 0
		self.shadow_L_std = 5.2
		self.shadow_N_std = 7.2

	def get_path_loss(self, position_BS, position_UE, shadowing):
		BS_x, BS_y = position_BS
		UE_x, UE_y = position_UE

		distance = m.sqrt(m.pow(BS_x-UE_x,2) - m.pow(BS_y-UE_y,2))
		path_loss = 10*m.log10((4*m.pi*self.fMBS)/c) + 10*self.beta*m.log10(distance)+shadowing