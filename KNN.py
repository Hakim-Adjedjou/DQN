import numpy as np
import tensorflow as tf
import gym
from gym import Env
from gym.spaces import Tuple,Discrete,Box,Dict,MultiDiscrete,MultiBinary
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
import pandas as pd

df=pd.read_csv("C:/Users/Dual Computer/Desktop/Nouveau dossier/noise_0.5/aggregated_0.5_noise.csv",header=None)
df_test=df[1000:1099]
df_train=df[0:999]
print(df.iloc[0])
print(df_train.iloc[0])