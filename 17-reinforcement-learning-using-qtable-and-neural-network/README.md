## Reinforcement Learning using Q-Table and Neural Network Q-Learning
This repository contains Python scripts implementing reinforcement learning algorithms for training a blue dot to navigate towards a fixed red dot in a 10 by 10 space. The blue dot aims to reach the position of the red dot to receive maximum reward. The distance between the blue and red dots is calculated after each step, and rewards are adjusted accordingly. The goal is to maximize the reward by reaching the red dot within a maximum number of iterations.

### Algorithms Implemented

#### Q-Table Learning
q_table_learning.py
Q-learning algorithm is implemented using a Q-table to store state-action values. Rewards are updated based on the distance between the blue and red dots. Maximum reward is achieved when the blue dot reaches the position of the red dot.

#### Neural Network Q-Learning
neural_network_q_learning.py
Q-learning algorithm is implemented using a neural network to approximate Q-values. The neural network takes the current state as input and outputs Q-values for each possible action. Rewards are updated based on the distance between the blue and red dots. The neural network is trained to maximize the reward by adjusting its weights.

### Results
After training, the performance of the reinforcement learning algorithm is evaluated. A line chart is plotted to show the number of successful target achievements per episode over multiple iterations.

### Dependencies
Python 3.x
NumPy
Tensorflow
Matplotlib

### License
This project is licensed under the MIT License.