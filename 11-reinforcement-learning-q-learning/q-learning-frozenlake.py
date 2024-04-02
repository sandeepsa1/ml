import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v1', render_mode='human') #4*4 map

def load_table(filename):
    data = np.load(filename)
    q = data['q']
    epsilon = data['epsilon']
    return q, epsilon

def learn(episodes, train=True):
    learning_rate = .1
    discount = .1
    epsilon_decay = .0001
    filename = 'q_table.npz'

    ob = env.observation_space.n
    act = env.action_space.n
    #print((ob, act)) #16, 4
    
    # Set this to True to start new training
    # False will load already trained q table and epsilon value from saved file to contine training
    start_new_learning = True # Loads already trained data if set to false
    if(start_new_learning == True):
        epsilon = 1
        q = np.zeros((ob, act)) # Inner bracket needed to get 2D 16*4 q space
    else:
        q, epsilon = load_table(filename)
    #print(q)

    rewards_table = []
    for i in range(episodes):
        truncated = False
        terminated = False
        state = env.reset()[0]
        totalreward = 0.0
        
        while (terminated == False and truncated == False):
            explore = np.random.rand()
            if(train == True and explore < epsilon): # Decide explore or exploit
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state,:]) # Select corresponding row in q table for a state and then max
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            if(train == True): # Update q table only in trining mode
                q[state, action] = (1 - learning_rate) * q[state, action] + \
                    learning_rate * (reward + discount * np.max(q[next_state,:]))

            state = next_state
            totalreward += reward
        
        epsilon = max(epsilon - epsilon_decay, 0)
        rewards_table.append(totalreward)
        print((i+1, totalreward))
    
    np.savez(filename, q=q, epsilon=epsilon)
    print("Final Q Table")
    print(q)
    print("Final Epsilon : " + str(epsilon))
    return rewards_table


episodes = 5
rewards_table = learn(episodes, True)
#print(rewards_table)
size = max(episodes // 10, 1)
num_windows = episodes // size
avg_rewards = [np.mean(rewards_table[i*size:(i+1)*size]) for i in range(num_windows)]

plt.plot(range(size, episodes+1, size), avg_rewards)
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.title(f'Average Reward per {size} Episodes')
plt.show()
