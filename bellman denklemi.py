import numpy as np
import random

# Ortam ve parametreler
grid_size = 3
actions = ['up', 'down', 'left', 'right']
Q = np.zeros((grid_size, grid_size, len(actions)))

gamma = 0.9
alpha = 0.1
epsilon = 0.2
reward_goal = 10
reward_step = -1

# Eğitim döngüsü
for episode in range(100):
    i, j = 0, 0  # Başlangıç durumu

    while (i, j) != (2, 2):  # Hedefe ulaşana kadar

        # Eylem seçimi (epsilon-greedy)
        if random.uniform(0, 1) < epsilon:
            a = random.randint(0, 3)
        else:
            a = np.argmax(Q[i, j])

        # Geçiş
        i_next, j_next = i, j
        if actions[a] == 'up': i_next = max(i-1, 0)
        elif actions[a] == 'down': i_next = min(i+1, grid_size-1)
        elif actions[a] == 'left': j_next = max(j-1, 0)
        elif actions[a] == 'right': j_next = min(j+1, grid_size-1)

        # Ödül
        if (i_next, j_next) == (2, 2):
            reward = reward_goal
        else:
            reward = reward_step

        # Bellman optimaliteye göre Q-table güncellemesi
        max_Q_next = np.max(Q[i_next, j_next])  # s' durumundaki en iyi eylem
        Q[i, j, a] += alpha * (reward + gamma * max_Q_next - Q[i, j, a])

        # Durum güncelle
        i, j = i_next, j_next
import matplotlib.pyplot as plt

# Eylem isimleri
actions = ['up', 'down', 'left', 'right']

# Her eylem için ayrı grafik çiz
for a_idx in range(len(actions)):
    plt.figure()
    plt.title(f"Q-table: Action = {actions[a_idx]}")
    plt.imshow(Q[:, :, a_idx], cmap='coolwarm', interpolation='none')
    plt.colorbar(label='Q-value')
    plt.xlabel('Grid Column')
    plt.ylabel('Grid Row')
    plt.xticks(range(Q.shape[1]))
    plt.yticks(range(Q.shape[0]))
    plt.grid(False)
    plt.show()
best_actions = np.argmax(Q, axis=2)
plt.figure()
plt.title("En iyi eylem yönü (0:up, 1:down, 2:left, 3:right)")
plt.imshow(best_actions, cmap='Accent', interpolation='none')
plt.colorbar(label='Action Index')
plt.xticks(range(Q.shape[1]))
plt.yticks(range(Q.shape[0]))
plt.grid(False)
plt.show()