import numpy as np
import matplotlib.pyplot as plt
import time
import random

grid_size = 10
max_episodes = 50
max_iterations = 25
epsilon = 0.9
epsilon_decay = 0.02
learning_rate = 0.8
discount = 0.9
blue_dot_start_pos = (0, 0)
red_dot_pos = (5, 5)

# Up, Down, Left, Right
actions = [(0, 1), (0, -1), (-1, 0), (1, 0)]
qtable = np.zeros((grid_size, grid_size, len(actions)))

# Distance between two points
def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def choose_action(state):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(range(len(actions)))
    else:
        return np.argmax(qtable[state[0], state[1]])

def take_action(state, action):
    new_state = (state[0] + action[0], state[1] + action[1])
    
    new_state = (max(0, min(grid_size - 1, new_state[0])), max(0, min(grid_size - 1, new_state[1])))
    if new_state == red_dot_pos:
        return new_state, 10  # Maximum reward if blue dot reaches red dot
    else:
        reward = -distance(new_state, red_dot_pos) + distance(state, red_dot_pos)
        return new_state, reward

# Bellmans Equation : Q(s,a) = Q(s,a) + α * (r + γ * max(Q(s',a')) - Q(s,a))
def update_qtable(state, action, reward, new_state):
    qtable[state[0], state[1], action] += learning_rate * (reward + discount * np.max(qtable[new_state[0], new_state[1]]) - qtable[state[0], state[1], action])

def run_episode():
    blue_dot_position = blue_dot_start_pos #(random.randint(0, 9), random.randint(0, 9))
    episode_reward = 0

    for _ in range(max_iterations):
        state = blue_dot_position
        action_index = choose_action(state)
        action = actions[action_index]
        new_state, reward = take_action(state, action)
        update_qtable(state, action_index, reward, new_state)
        blue_dot_position = new_state
        episode_reward += reward

        if blue_dot_position == red_dot_pos:
            break
        
        plt.clf()
        plt.scatter(*blue_dot_position, color='blue')
        plt.scatter(*red_dot_pos, color='red')
        plt.xlim(0, grid_size)
        plt.ylim(0, grid_size)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Episode Movement')
        plt.grid(True)
        plt.pause(0.01)

    return episode_reward

episode_rewards = []
for episode in range(max_episodes):
    episode_reward = run_episode()
    episode_rewards.append(episode_reward)
    
    epsilon = max(epsilon - epsilon_decay, 0)
    
    if episode % 10 == 0:
        print(f"Episode {episode}: Total Reward = {episode_reward}: Epsilon = {epsilon}")

plt.clf()
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Episode Rewards')
plt.grid(True)
plt.show()