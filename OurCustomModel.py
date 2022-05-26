from OurEnvironment import Environment
import random
import scipy.stats as s
import numpy as np
from tensorflow.keras import Input, Model
from collections import deque
from tensorflow.keras.layers import Dense, concatenate, Flatten
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

class CustomModel:
	def __init__(self, 
				inputs: list,
				hidden_units: list,
				outputs: list,
				output_activation: str,
				upper_bound: float):

		self.inputs = []
		self.hidden = []
		self.outputs = []
		self.upper_bound = upper_bound

		for input_units in inputs:
			self.inputs.append(Input(shape=input_units))

		for hidden_unit in hidden_units:
			self.hidden.append(Dense(hidden_unit, activation='leaky_relu'))

		# outputs: 3. each output: Dense(n_BS[tier])
		for n_output in outputs:
			self.outputs.append(Dense(n_output, activation=output_activation))
	
	def build_model(self):
		outputs = []
		if len(self.inputs) == 1:
			x = self.hidden[0](self.inputs[0])
			for hidden_layer in self.hidden[1:]:
				x = hidden_layer(x)
		else:
			x = concatenate([hidden_layer(input_unit) for hidden_layer, input_unit in zip(self.hidden, self.inputs)],axis=1)

		for output_layer in self.outputs:
			outputs.append(output_layer(x)*self.upper_bound)
		
		return Model(inputs=self.inputs, outputs=outputs)

	
if __name__ == '__main__':
	# Creating a single input (4x1), 2 hidden layer, 3 output (mbs,mm,thz) model for the actor
	actor = CustomModel([(4,1)], [32,16], [2,3,4], 'tanh', 40.0).build_model()
	print(actor.summary())