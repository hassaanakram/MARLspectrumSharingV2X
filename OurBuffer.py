import numpy as np
np.seterr(all='raise')

class Buffer:
    def __init__(self, 
                 num_states: int,
                 num_actions: int,
                 buffer_capacity=100000,
                 batch_size=10):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_counter = 0

        self.memory = []
        #self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        #self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        #self.reward_buffer = np.zeros((self.buffer_capacity,1))
        #self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    
    def record(self, observation_tuple):
        # Make sure that the buffer capacity does not exceed
        index = self.buffer_counter % self.buffer_capacity

        self.memory[index] = observation_tuple
        #self.state_buffer[index,:] = observation_tuple[0]
        #self.action_buffer[index,:] = observation_tuple[1]
        #self.reward_buffer[index,:] = observation_tuple[2]
        #self.next_state_buffer[index,:] = observation_tuple[3]

        self.buffer_counter += 1

    