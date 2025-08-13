# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 18:12:43 2025

@author: asus
"""
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
#env.tanimi
env = gym.make("CliffWalking-v0")
#Qtable olustur
state_size = env.observation_space.n
action_size = env.action_space.n
Q = np.zeros((state_size, action_size))

#degiskenler
alpha = 0.1 #ogrenme oranı
gamma = 0.99#
episodes = 500#egitim dongusu

outcomes=[]

#training
for _ in tqdm(range(episodes)):
    state,_=env.reset()
    done=False#ajanın basari durumu
    
    outcomes.append("Failure")
    while not done:#ajan basasrili olana kadar state icinde hareket et
        if np.max(Q[state]>0):
            action=np.argmax(Q[state])
        else:
            action=env.action_space.sample()
            
        new_state,reward,done,info,_=env.step(action)
        #update qtable
        Q[state,action]=Q[state,action]+alpha*(reward+gamma*np.max(Q[new_state])-Q[state,action])  
        state=new_state
        
        if reward == 0:
            outcomes[-1] = "Success"

            
print("qtable after training")
print(Q)

numeric_outcomes = [1 if o == "Success" else 0 for o in outcomes]
plt.bar(range(episodes), numeric_outcomes)
plt.xlabel("Episode")
plt.ylabel("Success (1) / Failure (0)")
plt.title("Eğitim Başarı Grafiği")
plt.show()
#test
episodes=100
nb_success=0


for _ in tqdm(range(episodes)):
    state,_=env.reset()
    done=False#ajanın basari durumu
    
    
    while not done:#ajan basasrili olana kadar state icinde hareket et
        if np.max(Q[state]>0):
            action=np.argmax(Q[state])
        else:
            action=env.action_space.sample()
            
        new_state,reward,done,info,_=env.step(action)
        #update qtable
        Q[state,action]= Q[state,action]+alpha*(reward+gamma*np.max(Q[new_state])-Q[state,action])  
        state=new_state
        if reward == 0:
            nb_success += 1
print("success rate:",100*nb_success/episodes)
