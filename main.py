# This is a sample Python script.
import os

import numpy
import numpy as np
import tensorflow as tf
import gym
from gym import Env
from gym.spaces import Tuple,Discrete,Box,Dict,MultiDiscrete,MultiBinary
from sklearn.ensemble import RandomForestRegressor
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from Custom_en import ShowerEnv

df_knn=pd.read_csv('Knn_results.csv',header=0)
df_knn.drop(columns=["Unnamed: 0"], axis=1, inplace=True)


df_NN=pd.read_csv('NN_results.csv',header=0)
df_NN.drop(columns=["Unnamed: 0"], axis=1, inplace=True)

df_random_forest=pd.read_csv('random_forest_results.csv',header=0)
df_random_forest.drop(columns=["Unnamed: 0"], axis=1, inplace=True)

main_df=pd.concat([df_knn,df_NN,df_random_forest],axis=1)
df_results=pd.DataFrame()
df_weights=pd.DataFrame()

########trainning###################################

env=ShowerEnv()
env.update(main_df.iloc[10])
env.state_update()

log_path = os.path.join('Training', 'Logs')
model_path=os.path.join('Training','Saved Models','PPO_model')
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500000)
model.save(model_path)
env.close()


##########exploitation###############################

model_path=os.path.join('Training','Saved Models','PPO_model')
#for i in range(100) :
env_1=ShowerEnv()
env_1.update(main_df.iloc[99])
env_1.state_update()
env_1.final_state_update()
model=PPO.load(model_path,env_1)
evaluate_policy(model, env_1, n_eval_episodes=10, render=True)
print(env_1.final_state)
print(env_1.final_weights)
#new_row = pd.DataFrame(env_1.final_state, columns=['X','Y'])
#df_results = pd.concat([df_results, new_row], ignore_index=True)
env_1.close()



####################### results########################

#df_results.to_csv("RL_results_noise_0.5_3tech_fusion.csv")

"""

df=pd.read_csv("C:/Users/Dual Computer/Desktop/Nouveau dossier/noise_0.5/aggregated_0.5_noise.csv",header=None)
dt=pd.read_csv("C:/Users/Dual Computer/Desktop/Nouveau dossier/noise_0.5/positions_0.5_noise.csv",header=None)

df_test=pd.concat([df[1000:1100],dt[1000:1100]],axis=1)
df_train=pd.concat([df[0:1000],dt[0:1000]],axis=1)
df_test.set_axis(range(12),axis="columns",inplace=True)
df_train.set_axis(range(12),axis="columns",inplace=True)

################ parameter tuning for KNN #############################
K_values=[i for i in range(1,50)]
K_scores=[]
for k in K_values:
 clf=KNeighborsRegressor(k)
 cv_scores=cross_val_score(clf,df_train.iloc[:,0:10],df_train.iloc[:,10:12],cv=5,scoring="neg_mean_squared_error")
 K_scores.append(cv_scores.mean())
 #clf.fit(df_train.iloc[:,0:10],df_train.iloc[:,10:12])
 #knn_pred=clf.predict(df_test.iloc[:,0:10])

optimal_k=K_values[K_scores.index(max(K_scores))]
print(optimal_k)
print(K_scores)

#################### parameter tuning for random forest #################
rf_values=[i for i in range(50,1000,10)]
rf_scores=[]
for k in rf_values:
 clf=RandomForestRegressor(n_estimators=k)
 cv_scores=cross_val_score(clf,df_train.iloc[:,0:10],df_train.iloc[:,10:12],cv=5,scoring="neg_mean_squared_error")
 rf_scores.append(cv_scores.mean())
 #clf.fit(df_train.iloc[:,0:10],df_train.iloc[:,10:12])
 #knn_pred=clf.predict(df_test.iloc[:,0:10])

optimal_k=rf_values[rf_scores.index(max(rf_scores))]
print(optimal_k)
print(rf_scores)
"""
############## KNN prediction ###############################
"""
clf=KNeighborsRegressor(2)
clf.fit(df_train.iloc[:,0:10],df_train.iloc[:,10:12])
knn_pred=clf.predict(df_test.iloc[:,0:10])
print(mean_squared_error(df_test.iloc[:,10:12],knn_pred))
df_knn_results=pd.DataFrame()
df_random_forest=pd.DataFrame()
new_row = pd.DataFrame(knn_pred, columns=['X','Y'])
df_knn_results = pd.concat([df_knn_results, new_row], ignore_index=True)
df_knn_results.to_csv("Knn_results.csv")

################# random forest prediction ########################
clf=RandomForestRegressor(n_estimators=500)
clf.fit(df_train.iloc[:,0:10],df_train.iloc[:,10:12])
rf_pred=clf.predict(df_test.iloc[:,0:10])
print(mean_squared_error(df_test.iloc[:,10:12],rf_pred))
df_random_forest=pd.DataFrame()
new_row = pd.DataFrame(rf_pred, columns=['X','Y'])
df_random_forest = pd.concat([df_random_forest, new_row], ignore_index=True)
df_random_forest.to_csv("random_forest_results.csv")


############ neural networks prediction ########################

df_NN=pd.DataFrame()

model = Sequential()
model.add(Dense(200, input_dim=10, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(2, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(df_train.iloc[:,0:10], df_train.iloc[:,10:12], epochs=100, batch_size=10)

test_loss = model.evaluate(df_test.iloc[:,0:10], df_test.iloc[:,10:12], verbose=1)
print("Test Loss:", test_loss)

NN_results=model.predict(df_test.iloc[:,0:10])

new_row = pd.DataFrame(NN_results, columns=['X','Y'])
df_NN = pd.concat([df_NN, new_row], ignore_index=True)
df_NN.to_csv("NN_results.csv")
"""




















