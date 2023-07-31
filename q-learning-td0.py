#!/usr/bin/env python3
"""
Q-Learning with Cartpole Problem of OpenAI Gym
Author: Magi Chen <chenmagi@gmail.com>

This script implements the Q-Learning algorithm to solve the Cartpole problem from the OpenAI Gym environment.
The goal is to teach an agent to balance a pole on a moving cart for as long as possible.

Environment:
- Action Space: The agent can take two actions - pushing the cart to the left or the right.
- Observation Space: The observations include the cart's position, velocity, pole angle, and pole angular velocity.

Key Components:
1. MappingConfig: A class that helps to map and discretize the continuous observation values to discrete states.

2. LearningRateControl: A class that controls the exploration and learning rates during training.

3. Q_Learning: The main class that encapsulates the Q-Learning algorithm. It includes methods to select actions,
   update Q-values, and run the Q-Learning process.

Usage:
1. Ensure you have the necessary dependencies installed: gym, numpy, matplotlib.
   You can install them using pip:
   pip install -r requirements.txt

"""
import gym
from collections import defaultdict
import math
import numpy as np
import random
import time
import matplotlib
import matplotlib.pyplot as plt
import os



"""CartPole environment
    1. action space: two actions of left push and right push
    2. observation space: position, velocity, pole angle, pole angular velocity
"""


    

class MappingConfig(defaultdict):
    def __init__(self, range, nslots) -> None:
        super().__init__(lambda: 0)
        self['min'] = range[0]
        self['max'] = range[1]
        self['nslots'] = nslots
        self['offset'] = (nslots-1)*range[0]/(range[1]-range[0])
        self['scaling'] = (nslots-1)/(range[1]-range[0])
        return
    
    def mapping(self,value) -> int:
        if value <= self['min']: return 0
        if value >= self['max']: return self['nslots']-1
        return int(round(self['scaling']*value-self['offset']))
    
class LearningRateControl:
    def __init__(self, bound):
        self.__lower_bound = bound[0]
        self.__upper_bound = bound[1]
        self.__rate = bound[1]
       
    
    @property
    def value(self):
        return self.__rate
    
    def simple_update(self, elapse):
        elapse+=1
        p = math.log10(elapse/25)
        self.__rate = max(self.__lower_bound,min(self.__upper_bound,1.0- p))
    
    def reset(self):
        self.__rate = self.__upper_bound


            

def encode_observation(observation, encoders) -> tuple:
    ''' encode the input observation to a discrete state represented by tuple'''
    state = []
    for i in range(len(observation)):
        value = observation[i]
        st = encoders[i].mapping(value)
        state.append(st)
    
    #assert len(state) == 4, 'len(observation) and len(encoders) are equal to 4'
    return tuple(state)


class Q_Learning:
    
    def __init__(self, gym_env, state_shape=(1,1,6,3), action_shape=(2,)) -> None:
        self.state_encoder = self.__initial_state_mapping(state_shape)
        self.q_table = None 
        self.explore_ctrl = LearningRateControl([0.01,1.0])
        self.learning_ctrl = LearningRateControl([0.1,0.5])
        self.discount_rate = 0.9
        self.env = gym_env
        self.__state_shape = state_shape
        self.__action_shape = action_shape
        self.isreplay = False
        return
    
    def __initial_state_mapping(self,shape=(1,1,6,3)) -> list:
        config = []
        config.append(MappingConfig((-4.8,4.8),shape[0])) #cart position
        config.append(MappingConfig((-0.5,0.5),shape[1])) #cart velocity
        config.append(MappingConfig((-0.48,0.48),shape[2])) # pole angle
        config.append(MappingConfig((-math.radians(70),math.radians(70)),shape[3])) # pole angular velocity
        return config        
    
    def select_action(self,state,epsilon):
        if self.isreplay or random.random() > epsilon: # do greedy selection
            return np.argmax(self.q_table[state])
        else:
            return self.env.action_space.sample()
    
    def save_trained_model(self,q_table):
        fname=time.strftime("%Y%m%d-%H%M%S")
        np.save(fname,q_table)
        return fname
    
    def __reset(self, path):
        isreplay=False
        if os.path.isfile(path):
            self.q_table = np.load(path,  allow_pickle=True)
            isreplay=True
        else:
            self.q_table = np.zeros(self.__state_shape+self.__action_shape)
        self.explore_ctrl.reset()
        self.learning_ctrl.reset()
        return isreplay
    
    def run(self, num_episodes, max_steps, replay=""):
        self.isreplay = self.__reset(replay)
        convergence=0
        hist=[]
      
        for episode in range(num_episodes):
            observation = self.env.reset(seed=0)
            #print(observation[0])
            state = encode_observation(observation[0], self.state_encoder)
            step = 0
            for step in range(max_steps):
                self.env.render()
                
                action = self.select_action(state, self.explore_ctrl.value)
                observation, reward, terminated, truncated, info = self.env.step(action)
                state_prime = encode_observation(observation,self.state_encoder)
                q_of_state_prime = np.amax(self.q_table[state_prime])
                td_error = reward+ self.discount_rate*q_of_state_prime - self.q_table[state+(action,)]
                self.q_table[state+(action,)] += self.learning_ctrl.value*td_error
                #print('action={},obv={}, reward={}, done={}, state={}, state\'={}'.format(action,observation, reward,terminated, state, state_prime))
                state = state_prime
                if terminated:  
                    if convergence >0: convergence-=1
                    print('episode={}, keep time={}'.format(episode,step))
                    break
            hist.append(step)
            if step >= max_steps-1:
                print('episode={}, keep time={}'.format(episode,step))
                convergence+=1
                if convergence > 40:
                    fname=self.save_trained_model(self.q_table)
                    print("Converged.spent episode count={}".format(episode))
                    return hist,fname
                
            self.explore_ctrl.simple_update(episode)
            self.learning_ctrl.simple_update(episode)
            
        return hist, None
    
    def plot(self,hist):
        x_range=list(range(len(hist)))
        plt.plot(x_range, hist)
        plt.legend()
        plt.title('cartpole-v1 with q-learning')
        plt.show()
            
            
                
                
            
    
def main():
    random.seed(0)
    env = gym.make('CartPole-v1', render_mode="human")
    q_learner = Q_Learning(env)
    hist,fname = q_learner.run(num_episodes=2000, max_steps=475)
    env.close()
    q_learner.plot(hist)
    return

    

    

if __name__ == '__main__':
    main()
