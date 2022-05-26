from OUActionNoise import OUActionNoise
from OurACAgent import ACAgent
from OurBuffer import Buffer
from OurEnvironment import Environment
from OurCustomModel import CustomModel
import tensorflow as tf
import numpy as np
import scipy.stats as s
from tqdm import tqdm
import matplotlib.pyplot as plt
np.seterr(all='raise')

def actor_loss(y_true, y_pred):
	return -tf.math.reduce_sum(y_true*tf.math.log(y_pred))

def train_loop(total_episodes: int, 
			   steps_per_ep: int, 
			   env: Environment, 
			   agent: ACAgent,
			   init_state: list):
	ep_reward_list = []
	avg_received_power_list = []
	avg_transmit_power_list = []
	#* Sim params
	betas = s.uniform(loc=2.9, scale=1.2).rvs(size=(total_episodes))
	prob_LOS = s.uniform().rvs(size=(total_episodes))
	prob_NLOS = 1 - prob_LOS
	
	state = init_state


	for e in tqdm(range(total_episodes)):
		env.renew_channel(betas[e],prob_LOS[e],prob_NLOS[e])

		episodic_reward = 0
		avg_received_power_episodic = []
		avg_transmit_power_episodic = []
		for i in range(steps_per_ep):
			actions = agent.act(np.array(state)[None,:])
			transmit_powers = dict(zip(env.tier_list, actions))
			prev_transmit_powers = {'MBS':[], 'mmWave':[], 'THz':[]}
			for tier in env.tier_list:
				for idx in range(len(env.BS[tier])):
					prev_transmit_powers[tier].append(env.BS[tier][idx].power_transmitted)
					env.BS[tier][idx].power_transmitted = transmit_powers[tier][idx]
			
			reward, next_state, avg_received_power, avg_transmit_power = env.step([state[0], np.mean(state[1:])])
			agent.replay_buffer.record((state, actions, reward, next_state))
			state = next_state
			episodic_reward +=  reward
			agent.learn()
			agent.update_target()
			avg_transmit_power_episodic.append(avg_transmit_power)
			avg_received_power_episodic.append(avg_received_power)
		ep_reward_list.append(episodic_reward)
		avg_transmit_power_list.append(np.nanmean(avg_transmit_power_episodic))
		avg_received_power_list.append(np.nanmean(avg_received_power_episodic))
	
	return ep_reward_list, avg_received_power_list, avg_transmit_power_list

def main():
	# Define env initially
	env_params = {'alpha':0.5, 'prob_LOS':0.7,
				'prob_NLOS':0.3, 'area':700*700,
				'lambdas':{'MBS':4E-6,'mmWave':8E-6,
							'THz':24E-6,'UE':0.0012}}
	env = Environment(**env_params)
	init_state = env.reset()

	#* Defining models
	# actor inputs: (avg Pr, avg Pt mbs, avg Pt mm, avg Pt THz)
	actor_inputs = [(4,)] 
	actor_hidden_layers = [64,32,16]
	actor_outputs = [env.n_BS['MBS'](),
					 env.n_BS['mmWave'](),
					 env.n_BS['THz']()]
	actor_output_activ = 'tanh'
	actor_upper_bound = 40.0
	actor_model = CustomModel(actor_inputs,
							  actor_hidden_layers,
							  actor_outputs,
							  actor_output_activ,
							  actor_upper_bound).build_model()
	target_actor = CustomModel(actor_inputs,
							   actor_hidden_layers,
							   actor_outputs,
							   actor_output_activ,
							   actor_upper_bound).build_model()

	critic_inputs = [(4,),
					(env.n_BS['MBS']()+env.n_BS['mmWave']()+env.n_BS['THz'](),)]
	critic_hidden_layers = [32,32]
	critic_outputs = [env.n_BS['MBS'](),
					 env.n_BS['mmWave'](),
					 env.n_BS['THz']()]
	critic_output_activ = None
	critic_upper_bound = 1.0
	critic_model = CustomModel(critic_inputs,
							   critic_hidden_layers,
							   critic_outputs,
							   critic_output_activ,
							   critic_upper_bound).build_model()
	target_critic = CustomModel(critic_inputs,
							   critic_hidden_layers,
							   critic_outputs,
							   critic_output_activ,
							   critic_upper_bound).build_model()
	
	#* Making model weights equal initially
	target_actor.set_weights(actor_model.get_weights())
	target_critic.set_weights(critic_model.get_weights())

	#* Compile models
	actor_optimizer = tf.keras.optimizers.Adam(0.001)
	critic_optimizer = tf.keras.optimizers.Adam(0.002)
	actor_model.compile(optimizer=actor_optimizer,
						loss=actor_loss)
	target_actor.compile(optimizer=actor_optimizer,
						 loss=actor_loss)
	critic_model.compile(optimizer=critic_optimizer,
						 loss='mse')
	target_critic.compile(optimizer=critic_optimizer,
						  loss='mse')
	
	#* Define AC Agent
	gamma = 0.99
	tau = 0.005
	buffer = Buffer(4, 3)
	noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(0.2) * np.ones(1))
	agent = ACAgent(actor_model, critic_model,
					target_actor, target_critic,
					buffer, gamma, tau,
					0.0, 40.0, noise)
	#* Training loop params
	total_episodes = 200
	steps_per_ep = 30
	ep_reward_list, avg_received_power_list, avg_transmit_power_list = train_loop(total_episodes, steps_per_ep, env, agent, init_state)
	
	plt.figure()
	plt.plot([i for i in range(total_episodes)], ep_reward_list)
	plt.xlabel('Episodes')
	plt.ylabel('Reward')
	plt.figure()

	plt.plot([i for i in range(total_episodes)], avg_transmit_power_list)
	plt.plot([i for i in range(total_episodes)], avg_received_power_list)
	plt.legend(["Average transmit power", "Average received power"])
	plt.xlabel('Episodes')
	

	plt.show()

if __name__ == '__main__':
	main()







