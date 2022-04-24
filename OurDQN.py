from OurEnvironment import Environment

import random
import scipy.stats as s
import numpy as np
from tensorflow.keras import Sequential
from collections import deque
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

env = Environment(0.5)

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

    def build_model(self):

        model = Sequential()
        model.add(Dense(64, input_shape=(self.state_space,), activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def act(self, state):

        if np.random.rand() <= self.epsilon:
            return s.norm.rvs(size=(env.n_BS_MBS))
        act_values = self.model.predict(state)
        return act_values[0]

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
    action_space = env.n_BS_MBS
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
