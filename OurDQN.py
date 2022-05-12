from OurEnvironment import Environment

import random
import scipy.stats as s
import numpy as np
from tensorflow.keras import Sequential, Input, Model
from collections import deque
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

env_params = {'alpha':0.5, 'prob_LOS':0.7,
              'prob_NLOS':0.3, 'area':500*500,
              'lambdas':{'MBS':4E-6,'mmWave':8E-6,
                         'THz':24E-6,'UE':0.0012}}
env = Environment(**env_params)

np.random.seed(0)


class DQN:

    """ Implementation of deep q learning algorithm """

    def __init__(self, action_space, state_space):

        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = 1
        self.gamma = .95
        self.batch_size = 64
        self.epsilon_min = .01
        self.epsilon_decay = .995
        self.learning_rate = 0.001
        self.memory = deque(maxlen=100000)
        self.model = self.build_model()

    def build_model(self, n_BS: dict):
        
        input_MBS_power = Input(shape=(n_BS['MBS'],))
        input_MBS_bandwidth = Input(shape=(n_BS['MBS'],))
        input_mmWave_power = Input(shape=(n_BS['mmWave'],))
        input_mmWave_bandwidth = Input(shape=(n_BS['mmWave'],))
        input_THz_power = Input(shape=(n_BS['THz'],))
        input_THz_bandwidth = Input(shape=(n_BS['THz'],))
        
        #* Dealing with powers and bandwidths separately 
        x_MBS_power = Dense(64, activation='relu')(input_MBS_power)
        x_mmWave_power = Dense(64, activation='relu')(input_mmWave_power)
        x_THz_power = Dense(64, activation='relu')(input_THz_power)

        x_MBS_bandwidth = Dense(64, activation='leaky_relu')(input_MBS_bandwidth)
        x_mmWave_bandwidth = Dense(64, activation='leaky_relu')(input_mmWave_bandwidth)
        x_THz_bandwidth = Dense(64, activation='leaky_relu')(input_THz_bandwidth)

        x_powers = x_MBS_power + x_mmWave_power + x_THz_power
        x_bandwidths = x_MBS_bandwidth + x_mmWave_bandwidth + x_THz_bandwidth

        x_powers = Dense(32, activation='relu')(x_powers)
        x_bandwidths = Dense(32, activation='leaky_relu')(x_bandwidths)

        output_MBS_power = Dense(n_BS['MBS'], activation='linear')(x_powers)
        output_mmWave_power = Dense(n_BS['mmWave'], activation='linear')(x_powers)
        output_THz_power = Dense(n_BS['THz'])(x_powers, activation='linear')

        output_MBS_bandwidth = Dense(n_BS['MBS'], activation='linear')(x_bandwidths)
        output_mmWave_bandwidth = Dense(n_BS['mmWave'], activation='linear')(x_bandwidths)
        output_THz_bandwidth = Dense(n_BS['THz'], activation='linear')(x_bandwidths)

        model = Model(inputs=[input_MBS_power,input_mmWave_power,input_THz_power,
                              input_MBS_bandwidth, input_mmWave_bandwidth, input_THz_bandwidth],
                      outputs=[output_MBS_power, output_mmWave_power, output_THz_power,
                               output_MBS_bandwidth, output_mmWave_bandwidth, output_THz_bandwidth])
        
        
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def act(self, state):

        if np.random.rand() <= self.epsilon:
            out_MBS_power = s.randint(20,40,size=(env.n_BS['MBS']))
            out_mmWave_power = s.randint(22,40,size=(env.n_BS['mmWave']))
            out_THz_power = s.randint(23,39,size=(env.n_BS['THz']))
            
            out_MBS_bandwidth = s.randint(1E6,10E6,size=(env.n_BS['MBS']))
            out_mmWave_bandwidth = s.randint(1E9,2E9,size=(env.n_BS['mmWave']))
            out_THz_bandwidth = s.randint(3E9,8E9,size=(env.n_BS['THz']))

            random_outs = [out_MBS_power, out_mmWave_power, out_THz_power,
                           out_MBS_bandwidth, out_mmWave_bandwidth, out_THz_bandwidth]

            return random_outs
        act_values = self.model.predict(state)
        return act_values

    def replay(self):

        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.squeeze(np.array([i[0] for i in minibatch]))
        actions = np.squeeze(np.array([i[1] for i in minibatch]))
        rewards = np.squeeze(np.array([i[2] for i in minibatch]))
        next_states = np.squeeze(np.array([i[3] for i in minibatch]))
        #dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + self.gamma*(self.model.predict_on_batch(next_states))
        #targets_full = self.model.predict_on_batch(states)

        #ind = np.array([i for i in range(self.batch_size)])
        #targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_dqn(episode):

    loss = []
    
    #action_space = 3
    #state_space = 5
    max_steps = 1000
    state = env.reset()
    action_space = sum(env.n_BS.values())
    state_space = action_space
    agent = DQN(action_space, state_space)
    for e in range(1):
        state = np.reshape(state, (1, state_space))
        score = 0
        for i in tqdm(range(max_steps)):
            prev_received_powers = [UE.received_power for UE in env.UE]
            action = agent.act(state)
            prev_transmit_powers = [BS.power_transmitted for BS in env.BS_MBS]
            #! action is the new transmit powers so changing it in 
            #! our env

            for idx in range(len(env.BS_MBS)):
                env.BS_MBS[idx].power_transmitted = action[idx]
            reward, next_state = env.step(prev_received_powers, prev_transmit_powers)
            score += reward
            next_state = np.reshape(next_state, (1, state_space))
            agent.remember(state, action, reward, next_state)
            state = next_state
            agent.replay()
            #if done:
            #    print("episode: {}/{}, score: {}".format(e, episode, score))
            #    break
        loss.append(score)
    return loss


if __name__ == '__main__':

    ep = 100
    loss = train_dqn(ep)
    #plt.plot([i for i in range(ep)], loss)
    plt.plot([i for i in range(6)], np.squeeze(loss))
    plt.xlabel('episodes')
    plt.ylabel('reward')
    plt.show()
