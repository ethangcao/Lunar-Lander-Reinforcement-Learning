# Experiments for Lunar Lander Environment

## Abstract
Gym library in Python provides different environments for developing reinforcement learning algorithms. Lunar Lander environment’s goal is to land the lander on the landing pad by taken action to control the lander. The experiments in this paper are conducted based on discretized states with Q-Learning and Deep Q Network (DQN).

## Result
After fine-tuning:
Discretized states with Q-Learning: 66% success landing
Deep Q Network (DQN): 99% success landing

## Running the Code
### Required Python Packages
gym, numpy, pandas, random, torch, matplotlib, time
### Functions
Please change the flag in lunarlander.py for each plot to train the agent, run the experiment and render the plot.

Flag for Q-Learning\
plot1: Q-learning VS Random Action, 4 min runtime\
plot2: Q-learning with diffrent alpha, 11 min runtime\
plot3: Q-learning with diffrent epsilon, 11 min runtime\
plot4: Q-learning with diffrent gamma, 11 min runtime\
plot5: Q-learning Trained Agent VS Random Action & Q-learning Trained Agent Performance, 166 min runtime\
plot7: False # Dyna, 12 min runtime

Flag for DQN\
plot8: DQN VS Random Action, 9 min runtime\
plot9: DQN with sample size, 109 min runtime\
plot10: DQN with different node size, 58 min runtime\
plot11: DQN with different frequency, 48 min runtime\
plot12: DQN Trained Agent VS Random Action & DQN Trained Agent Performance, 30 min runtime
