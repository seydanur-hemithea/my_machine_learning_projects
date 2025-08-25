# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 17:45:37 2025
@author: asus
"""

import gym
import numpy as np

# Ortamı oluştur
env = gym.make("CartPole-v1")

# Her gözlem boyutu için 10 ayrık değer → toplam 10x10x10x10 durum
V = np.zeros((10, 10, 10, 10))

# Öğrenme parametreleri
alpha = 0.1       # Öğrenme oranı
gamma = 0.99      # Discount factor
epsilon = 0.1     # Keşfetme oranı
n_episodes = 500  # Eğitim bölümü sayısı
rewards = []
total_reward = 0

def discretize(obs):
    """Gözlemi 4 boyutlu ayrık duruma çevir"""
    bins = [(-2.4, 2.4), (-3.0, 3.0), (-0.5, 0.5), (-2.0, 2.0)]
    idx = []
    for i in range(len(obs)):
        val = obs[i]
        b = bins[i]
        scaled = int((val - b[0]) / (b[1] - b[0]) * 10)
        idx.append(max(0, min(9, scaled)))
    return tuple(idx)  # (i, j, k, l)

for episode in range(n_episodes):
    obs, _ = env.reset()
    done = False
   

    
    while not done:
        state = discretize(obs)

        # ε-greedy eylem seçimi
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            # Basit politika: çubuğun açısına göre yön seç
            action = 1 if obs[2] > 0 else 0

        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_state = discretize(next_obs)
        total_reward += reward


        # TD(0) güncellemesi
        V[state] += alpha * (reward + gamma * V[next_state] - V[state])

        obs = next_obs
        done = terminated or truncated
        rewards.append(total_reward)


env.close()
import matplotlib.pyplot as plt
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Öğrenme Eğrisi")
plt.show()
# Sonuçları yazdır
print("İlk 10 durumun tahmini değerleri (pole açısı sabit):")
for i in range(10):
    
    print("Ziyaret edilen bazı durumlar ve değerleri:")
    visited = np.argwhere(V > 0)
    for i in range(min(10, len(visited))):
        s = tuple(visited[i])
        print(f"V{str(s)} = {V[s]:.2f}")
