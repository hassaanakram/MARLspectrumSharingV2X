import tensorflow as tf
import numpy as np
import random
np.seterr(all='raise')

class ACAgent:
    def __init__(self,
                 actor_model,
                 critic_model,
                 target_actor,
                 target_critic,
                 replay_buffer,
                 gamma,
                 tau,
                 lower_bound,
                 upper_bound,
                 noise
                 ):

        self.actor_model = actor_model
        self.critic_model = critic_model
        self.target_actor = target_actor
        self.target_critic = target_critic
        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.tau = tau
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.noise_object = noise()
    
    def update_models(self, state_batch, action_batch, reward_batch, next_state_batch):
        # Train and update the critic model
        target_actions = self.target_actor(np.array(next_state_batch), training=True)
        critic_value = self.target_critic([np.array(next_state_batch), np.concatenate(target_actions, axis=1)], training=True)
        critic_value_old_state = self.target_critic([np.array(state_batch), np.concatenate(target_actions, axis=1)], training=True)
        y = [np.array(reward_batch)[:,None] + self.gamma*out_new - out_old for out_new, out_old in zip(critic_value, critic_value_old_state)]
        aaaaaaaaaa = [None, None, None]
        for idx in range(3):
            aaaaaaaaaa[idx] = [action_batch[batch_idx][idx] for batch_idx in range(len(action_batch))]
        self.critic_model.fit([np.array(state_batch), np.concatenate(aaaaaaaaaa, axis=1)], 
                               y, epochs=1, verbose=0)
        
        # update the actor model
        actions = self.actor_model(np.array(state_batch), training=True)
        critic_value = self.critic_model([np.array(state_batch), np.concatenate(actions, axis=1)], training=True)
        self.actor_model.fit(np.array(state_batch), critic_value, epochs=1, verbose=0)
    
    def learn(self):
        # sample from the buffer
        if len(self.replay_buffer.memory) < self.replay_buffer.batch_size:
            return
        #record_range = min(self.replay_buffer.buffer_counter, self.replay_buffer.buffer_capacity)
        observations = random.sample(self.replay_buffer.memory, self.replay_buffer.batch_size)

        state_batch = [obs[0] for obs in observations]
        action_batch = [obs[1] for obs in observations]
        reward_batch = [obs[2] for obs in observations]
        next_state_batch = [obs[3] for obs in observations]

        self.update_models(state_batch, action_batch, reward_batch, next_state_batch)

    # Function to update target network weights
    @tf.function
    def update_target(self):
        target_actor_weights = self.target_actor.variables
        actor_model_weights = self.actor_model.variables
        target_critic_weights = self.target_critic.variables
        critic_model_weights = self.critic_model.variables

        for (a, b) in zip(target_actor_weights, actor_model_weights):
            a.assign(b * self.tau + a * (1 - self.tau))
        
        for (a, b) in zip(target_critic_weights, critic_model_weights):
            a.assign(b * self.tau + a * (1 - self.tau))

    def act(self, state):
        sampled_actions = self.actor_model(state)
        sampled_actions = [tf.squeeze(act) for act in sampled_actions]
        sampled_actions = [np.clip(act.numpy() + self.noise_object, self.lower_bound, self.upper_bound) for act in sampled_actions]
        #legal_actions = np.clip(sampled_actions, self.lower_bound, self.upper_bound)
        return sampled_actions

    

    
    
