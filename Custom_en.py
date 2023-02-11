import os
import numpy as np
import tensorflow as tf
import gym
from gym import Env
from gym.spaces import Tuple,Discrete,Box,Dict,MultiDiscrete,MultiBinary
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
import random

from weight_normalise import weight_normalizer, points_initializer, state_initializer, fisher_information


class ShowerEnv(Env):
    def __init__(self):
        # Actions we can take, down, stay, up
        self.action_space = Discrete(9)
        self.weights = weight_normalizer()
        self.points=Box(0,100,shape=(3,1)).sample()
        # Temperature array
        self.observation_space = Box(0,100,shape=(1,1))
        # Set start temp
        self.state = Box(0, 100, shape=(1,1)).sample()
        # Set shower length
        self.episod_length = 1000
        self.final_state = Box(0, 100, shape=(1, 1)).sample()
        self.final_weights=weight_normalizer()




    def step(self, action):
        # Apply action
        # 0 -1 = -1 temperature
        # 1 -1 = 0
        # 2 -1 = 1 temperature
        tmp=fisher_information(self.points, self.state)
        if action==0 :
            self.weights[0]=self.weights[0]* 1.1
            self.weights[1] = self.weights[1]* 1.1
        elif action==1 :
            self.weights[0] = self.weights[0] * 1.1
            self.weights[1] = self.weights[1] * 0.9
        elif action==2 :
            self.weights[0] = self.weights[0] * 0.9
            self.weights[1] = self.weights[1] * 1.1
        elif action==3 :
            self.weights[0] = self.weights[0] * 0.9
            self.weights[1] = self.weights[1] * 0.9
        elif action==4 :
            self.weights[1] = self.weights[1] * 1.1
        elif action==5 :
            self.weights[1] = self.weights[1] * 0.9
        elif action==6 :
            self.weights[0] = self.weights[0] * 1.1
        elif action==7 :
            self.weights[0] = self.weights[0] * 0.9
        else :
            self.weights[0] = self.weights[0]
            self.weights[1] = self.weights[1]



        self.weights[2]=1-(self.weights[0]+self.weights[1])

        if sum(self.weights)>1 or self.weights[0]>1 or self.weights[0]<0 or self.weights[1]>1 or self.weights[1]<0 or self.weights[2]>1 or self.weights[2]<0:
            self.weights=Box(0,1,shape=(3,)).sample()
            self.weights=self.weights/sum(self.weights)

        self.state = state_initializer(self.points, self.weights)
        self.episod_length -= 1
        # Calculate reward
        if self.weights[0]<1 and self.weights[0]>0 and self.weights[1]<1 and self.weights[1]>0 and self.weights[2]<1 and self.weights[2]>0 :
            if fisher_information(self.points, self.state) < tmp:
                if fisher_information(self.points, self.state) < fisher_information(self.points, self.state):
                    reward=10
                    self.final_state=self.state
                    self.final_weights=self.weights
                else:
                    reward=1

            elif fisher_information(self.points, self.state) == tmp:
                reward=0
            else :
                reward = -1
        else :
            reward=-100


            # Check if shower is done
        if self.episod_length <= 0:
            done = True

        else:
            done = False

        # Apply temperature noise
        # self.state += random.randint(-1,1)
        # Set placeholder for info
        info = {}

        # Return step information
        return self.state, reward, done,info

    def render(self,mode='human'):
        # Implement viz
         if mode == 'human':
           pass

         elif mode == 'rgb_array':
            pass
         else:
             raise ValueError("Invalid render mode")

    def reset(self):
        # Reset shower temperature
        self.weights = weight_normalizer()
        self.state = state_initializer(self.points,self.weights)

        # Reset shower time
        self.episod_length = 1000
        return self.state

    def update(self,points):
        self.points=points_initializer(points)

    def state_update(self):
        self.state[0][0] = (self.points[0][0] * self.weights[0] + self.points[1][0] * self.weights[1] + self.points[2][0] * self.weights[2])
        #self.state[0][1] = (self.points[0][1] * self.weights[0] + self.points[1][1] * self.weights[1] + self.points[2][1] * self.weights[2])
    def final_state_update(self):
        self.final_state=self.state
