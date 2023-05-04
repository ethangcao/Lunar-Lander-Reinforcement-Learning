import gym
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

# Flag for Q-Learning
plot1 = False # Q-learning VS Random Action, 4 min runtime
plot2 = False # Q-learning with diffrent alpha, 11 min runtime
plot3 = False # Q-learning with diffrent epsilon, 11 min runtime
plot4 = False # Q-learning with diffrent gamma, 11 min runtime
plot5 = False # Q-learning Trained Agent VS Random Action, 166 min runtime
# plo6 is control by plot5 flag together
plot7 = False # Dyna, 12 min runtime

# Flag for DQN
plot8 = False # DQN VS Random Action, 9 min runtime
plot9 = False # DQN with sample size, 109 min runtime
plot10 = False # DQN with different node size, 58 min runtime
plot11 = False # DQN with different frequency, 48 min runtime
plot12 = False # DQN Trained Agent VS Random Action, 30 min runtime

# Method 1: Q-Learning with discretized states

env=gym.make('LunarLander-v2')

class QAgent():
    def __init__(self, gamma=0.99,
                       alpha = 0.1,
                       num_of_actions = env.action_space.n,
                       dyna = 0,
                       seed = 0):
        np.random.seed(seed)
        env.seed(seed)
        self.gamma = gamma
        self.alpha = alpha
        self.buckets=(5, 5, 5, 5, 5, 5, 2, 2)
        self.upper_bounds = [ 1,  1,  1,  1,  1,  1, 1, 1]
        self.lower_bounds = [-1, -1, -1, -1, -1, -1, 0, 0]
        self.num_of_actions = num_of_actions
        self.Q = np.zeros(self.buckets + (self.num_of_actions,))
        self.dyna = dyna
        self.replay = []
        # Discretize initial state
        state = env.reset()
        discretized = []
        for i in range(len(state)):
            discrete_state = int(round((self.buckets[i] - 1) * (state[i]-self.lower_bounds[i]) / (self.upper_bounds[i] - self.lower_bounds[i])))
            discrete_state = max(0, discrete_state)
            discrete_state = min(self.buckets[i] - 1, discrete_state)
            discretized.append(discrete_state)
        self.state = tuple(discretized)
       
    def act(self, epsilon = 0): # Determine action based on epsilon greedy
        if np.random.random() <= epsilon:  action = np.random.randint(self.num_of_actions)
        else: action = np.argmax(self.Q[self.state])
        return action
    
    def learn(self, action, next_state, reward, done):
        # Discretize observed next state
        discretized = []
        for i in range(len(next_state)):
            discrete_state = int(round((self.buckets[i] - 1) * (next_state[i]-self.lower_bounds[i]) / (self.upper_bounds[i] - self.lower_bounds[i])))
            discrete_state = max(0, discrete_state)
            discrete_state = min(self.buckets[i] - 1, discrete_state)
            discretized.append(discrete_state)
        discretized_next_state = tuple(discretized)

        # Update Q table
        if done: Q_next = 0
        else: Q_next = np.max(self.Q[discretized_next_state])
        self.Q[self.state][action]+=self.alpha*(reward+self.gamma*Q_next-self.Q[self.state][action])
        
        if 0<self.dyna:
            self.replay.append((self.state, action, discretized_next_state, reward))
            sample = random.choices(self.replay, k=self.dyna)
            for record in sample:
                self.Q[record[0]][record[1]]+=self.alpha*(record[3]+self.gamma*np.max(self.Q[record[2]])-self.Q[record[0]][record[1]])
        
        self.state = discretized_next_state
        
def Q_Learning(alpha = 0.2, gamma = 0.99 ,epsilon = 0.5, num_of_episodes=2000, dyna = 0):
    rewards = []
    Q_learner = QAgent(alpha = alpha, gamma = gamma, dyna = dyna)
    env.seed(0)
    for n in range(num_of_episodes):
        env.reset()
        reward_sum = 0
        done = False
        while not done:
            action = Q_learner.act(epsilon = epsilon)
            next_state, reward, done, info = env.step(action)
            reward_sum += reward
            Q_learner.learn(action, next_state, reward, done)
        rewards.append(reward_sum)
        print('\rTrained {} episodes'.format(n+1), end="")
        epsilon = max(0, min(1, epsilon*0.995))
    return rewards, Q_learner

def Q_Learning_infinity(alpha = 0.2, gamma = 0.99 ,epsilon = 0.5, dyna = 0):
    rewards = []
    Q_learner = QAgent(alpha = alpha, gamma = gamma, dyna = dyna)
    env.seed(0)
    n = 0
    mean = 0
    count = 0
    max_count = 0
    while count < 20:
        env.reset()
        reward_sum = 0
        done = False
        while not done:
            action = Q_learner.act(epsilon = epsilon)
            next_state, reward, done, info = env.step(action)
            reward_sum += reward
            Q_learner.learn(action, next_state, reward, done)
        rewards.append(reward_sum)
        if reward_sum>200: count+=1
        else: count=0
        max_count = max(max_count, count)
        mean = np.mean(rewards[-100:])
        print('\rmax:{}, Continuous Success: {}, Trained {} episodes: {}'.format(max_count,count, n+1, mean), end="")
        epsilon = max(0, min(1, epsilon*0.995))
        n+=1
    return rewards, Q_learner

def random_action(num_of_episodes=2000):
    rewards = []
    env.seed(0)
    for n in range(num_of_episodes):
        env.reset()
        reward_sum = 0
        done = False
        while not done:
            next_state, reward, done, info = env.step(env.action_space.sample())
            reward_sum += reward
        rewards.append(reward_sum)
        print('\rRandom acted {} episodes'.format(n+1), end="")
    return rewards

def Q_action(Q_learner, num_of_episodes=100):
    rewards = []
    Q_learner.alpha = 0
    env.seed(0)
    for n in range(num_of_episodes):
        env.reset()
        reward_sum = 0
        done = False
        while not done:
            action = Q_learner.act()
            next_state, reward, done, info = env.step(action)
            reward_sum += reward
            Q_learner.learn(action, next_state, reward, done)
        rewards.append(reward_sum)
        print('\rActed {} episodes'.format(n+1), end="")
    return rewards

runtime = []
runtime.append(time.time())
if plot1: #plot 1: random action VS Q-Learning
    print("Running plot 1: random action VS Q-Learning")
    reward_result1 = {}
    reward_result1['Random Action'] = random_action()
    print("\nRandom Action Completed")
    rewards, Q_learner = Q_Learning()
    reward_result1['Q Learning'] = rewards
    reward_df1 = pd.DataFrame(reward_result1) 
    reward_df1 = reward_df1.rolling(window = 100).mean()
    plt.clf()
    reward_df1.plot()
    plt.title("Q-Learning with discretized states VS Random Action")
    plt.legend()
    plt.savefig("Q-Learning1.png")
    print("\nPlot 1 finished")
    runtime.append(time.time())
    
if plot2: #plot 2: different alpha
    print("Running plot 2: different alpha")
    reward_result2 = {}
    for alpha in [0.05, 0.1, 0.2]:
        print("Training alpha="+str(alpha))
        rewards, Q_learner = Q_Learning(alpha = alpha)
        reward_result2["alpha="+str(alpha)] = rewards
        print("\nCompleted")
    reward_df2 = pd.DataFrame(reward_result2) 
    reward_df2 = reward_df2.rolling(window = 100).mean()
    plt.clf()
    reward_df2.plot()
    plt.title("Q-Learning with Different Learning Rate")
    plt.legend()
    plt.savefig("Q-Learning2.png")
    print("\nPlot 2 finished")
    runtime.append(time.time())
    
if plot3:     # plot 3: different epsilon
    print("Running plot 3: different epsilon")
    reward_result3 = {}
    for epsilon in [0.3, 0.5, 0.7]:
        print("Training epsilon="+str(epsilon))
        rewards, Q_learner = Q_Learning(epsilon = epsilon)
        reward_result3["epsilon="+str(epsilon)] = rewards
        print("\nCompleted")
    reward_df3 = pd.DataFrame(reward_result3) 
    reward_df3 = reward_df3.rolling(window = 100).mean()
    plt.clf()
    reward_df3.plot()
    plt.title("Q-Learning with Different Epsilon")
    plt.legend()
    plt.savefig("Q-Learning3.png")
    print("\nPlot 3 finished")
    runtime.append(time.time())
    
if plot4:     # plot 4: different gamma
    print("Running plot 4: different gamma")
    reward_result4 = {}
    for gamma in [0.9, 0.99, 1]:
        print("Training gamma="+str(gamma))
        rewards, Q_learner = Q_Learning(gamma = gamma)
        reward_result4["gamma="+str(gamma)] = rewards
        print("\nCompleted")
    reward_df4 = pd.DataFrame(reward_result4) 
    reward_df4 = reward_df4.rolling(window = 100).mean()
    plt.clf()
    reward_df4.plot()
    plt.title("Q-Learning with Different Discount Rate")
    plt.legend()
    plt.savefig("Q-Learning4.png")
    print("\nPlot 4 finished")
    runtime.append(time.time())
    
if plot5:     # plot 5: random action VS trained agent
    print("Running plot 5: random action VS trained agent")
    reward_result5 = {}
    reward_result5['Random Action'] = random_action(num_of_episodes=100)
    print("\nRandom Action Completed")
    print("Training Agent")
    rewards, Q_learner = Q_Learning_infinity()
    print("\nTraining Completed")
    rewards = Q_action(Q_learner, num_of_episodes=100)
    reward_result5['Trained Agent'] = rewards
    reward_df5 = pd.DataFrame(reward_result5) 
    plt.clf()
    reward_df5.plot()
    plt.title("Q-Learning Trained Agent VS Random Action")
    plt.legend()
    plt.savefig("Q-Learning5.png")
    
    plt.clf()
    out = pd.cut(reward_result5['Trained Agent'], bins=[-500, 0, 100, 200, 500])    
    count = out.value_counts().tolist()
    x = np.arange(4)
    fig, ax = plt.subplots()
    rects = ax.bar(x = x, height = count, width = 0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(['less than 0', '0 to 100', '100 to 200', 'greater than 200'])
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 1),
                        textcoords="offset points",
                        ha='center', va='bottom')
    autolabel(rects)
    plt.title("Q-Learning Trained Agent Performace")
    plt.savefig("Q-Learning6.png")
    print("\nPlot 5&6 finished")
    runtime.append(time.time())
    
if plot7: #plot 7: Dyna
    print("Running plot 7: Dyna")
    reward_result7 = {}
    reward_result7['Random Action'] = random_action()
    print("\nRandom Action Completed")
    rewards, Q_learner = Q_Learning(dyna = 200, epsilon = 1)
    reward_result7['Q Learning'] = rewards
    reward_df7 = pd.DataFrame(reward_result7) 
    reward_df7 = reward_df7.rolling(window = 100).mean()
    plt.clf()
    reward_df7.plot()
    plt.title("Q-Learning with discretized states & Dyna VS Random Action")
    plt.legend()
    plt.savefig("Q-Learning7.png")
    print("\nPlot 7 finished")
    runtime.append(time.time())

# Method 2: DQN

class LLModel(nn.Module):
    def __init__(self, num_of_states, hidden_nodes, num_of_actions, seed = 0):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.linear1 = nn.Linear(num_of_states, hidden_nodes)
        self.linear2 = nn.Linear(hidden_nodes, num_of_actions)
        
    def forward(self, x):
        out = self.linear1(x)
        out = F.relu(out)
        out = self.linear2(out)
        return out
    
    
class replay():
    def __init__(self, replay_size, sample_size, seed = 0):
        self.seed = random.seed(seed)
        self.replay_size = replay_size
        self.sample_size = sample_size
        self.replay = []
        
    def append(self, state, action, reward, next_state, done):
        if len(self.replay) == self.replay_size:  self.replay.pop(0)
        pair = [state[i] for i in range(len(state))]
        pair+=([next_state[j] for j in range(len(next_state))])
        pair+=[action, reward, done]
        self.replay.append(pair)
    
    def sample(self):
        sample = random.sample(self.replay, self.sample_size)
        sample = np.array(sample)
        states = torch.from_numpy(sample[:,0:8]).float()
        actions = torch.from_numpy(np.reshape(sample[:,16], (-1,1))).long()
        rewards = torch.from_numpy(np.reshape(sample[:,17], (-1,1))).float()
        next_states = torch.from_numpy(sample[:,8:16]).float()
        dones = torch.from_numpy(np.reshape(sample[:,18], (-1,1)).astype(np.uint8)).float()
        return (states, actions, rewards, next_states, dones)

class DQN_Agent():
    def __init__(self, 
                 alpha = 0.001, 
                 num_of_states = 8, 
                 num_of_actions = 4,
                 replay_size = 10000,
                 gamma = 0.99,
                 sample_size = 200,
                 hidden_nodes = 64,
                 frequency = 4,
                 seed = 0):
       
        self.num_of_states = num_of_states
        self.num_of_actions = num_of_actions
        self.seed = random.seed(seed)
        
        self.eval = LLModel(num_of_states, hidden_nodes, num_of_actions, seed)
        self.target = LLModel(num_of_states, hidden_nodes, num_of_actions, seed)
        self.optimizer = optim.Adam(self.eval.parameters(), lr=alpha)
        
        self.memory = replay(replay_size, sample_size, seed)
        self.sample_size = sample_size
        
        self.frequency = frequency
        self.t = 0
        
        self.gamma = gamma
            
    def act(self, state, epsilon=0):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_values = self.eval(state)
        if random.random() <= epsilon:
            return random.randint(0,self.num_of_actions-1)
        else:
            return np.argmax(action_values.numpy())
        
    def learn(self, state, action, reward, next_state, done):
        self.memory.append(state, action, reward, next_state, done)
        self.t = (self.t + 1) % self.frequency
        if self.t == 0 and len(self.memory.replay) > self.sample_size:
            states, actions, rewards, next_states, dones = self.memory.sample()
            Q_next = self.target(next_states).detach().max(1)[0]
            Q_next = torch.unsqueeze(Q_next, 1)* (1 - dones)
            Q_targets = rewards + (self.gamma * Q_next)
            Q_expected = self.eval(states).gather(1, actions)
            loss = F.mse_loss(Q_expected, Q_targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            for theta_t, theta_e in zip(self.target.parameters(), self.eval.parameters()):
                theta_t.data.copy_(theta_e.data)

def dqn(sample_size = 300, 
        hidden_nodes = 64,
        frequency = 4, 
        num_of_episodes=2000,
        terminate = True):
    rewards = []
    epsilon = 1
    DQN_Learner = DQN_Agent(sample_size = sample_size, 
                            hidden_nodes = hidden_nodes,
                            frequency = frequency,
                            seed =0)
    if not terminate: num_of_episodes = 50000
    count = 0
    for n in range(num_of_episodes):
        state = env.reset()
        reward_sum = 0
        done = False
        while not done:
            action = DQN_Learner.act(state, epsilon)
            next_state, reward, done, info = env.step(action)
            reward_sum += reward
            DQN_Learner.learn(state, action, reward, next_state, done)
            state = next_state
        rewards.append(reward_sum)
        if terminate:
            mean = np.mean(rewards[-100:])
            print('\rTrained {} episodes: {}'.format(n+1, mean), end="")
            if (mean>200):break
        else:
            if reward_sum>200: count+=1
            else:count = 0
            mean = np.mean(rewards[-100:])
            print('\rContinuous success: {},Tuned Agent Trained {} episodes: {}'.format(count,n+1, mean), end="")
            if (count>50):break
        epsilon = max(0.01, min(1, epsilon*0.995))
    return rewards, DQN_Learner
                
env = gym.make('LunarLander-v2')

def DQN_action(DQN_Learner, num_of_episodes=100):
    rewards = []
    for n in range(num_of_episodes):
        state = env.reset()
        reward_sum = 0
        done = False
        while not done:
            action = DQN_Learner.act(state = state, epsilon=0)
            next_state, reward, done, info = env.step(action)
            reward_sum += reward
            state = next_state
        rewards.append(reward_sum)
        print('\rActed {} episodes'.format(n+1), end="")
    return rewards

        
# plot 8: random action VS DQN
if plot8:
    print("Running plot 8: random action VS DQN")
    env.seed(0)
    reward_result8 = {}
    rewards, DQN_Learner = dqn()
    reward_result8['DQN'] = rewards
    print("\nTraining Completed")
    reward_result8['Random Action'] = random_action(num_of_episodes = len(rewards))
    print("\nRandom Action Completed")
    reward_df8 = pd.DataFrame(reward_result8) 
    reward_df8 = reward_df8.rolling(window = 100).mean()
    
    plt.clf()
    reward_df8.plot()
    plt.title("DQN VS Random Action")
    plt.legend()
    plt.savefig("DQN1.png")
    print("\nPlot8 finished")
    runtime.append(time.time())
    print("runtime: "+str((runtime[-1]-runtime[-2])/60))
    
# plot 9: DQN with different sample size
if plot9:    
    print("Running plot 9: DQN with sample size")
    reward_result9 = {}
    for num in [200, 300, 400, 500]:
        env = gym.make('LunarLander-v2')
        env.seed(0)
        print("Training sample size="+str(num))
        rewards, DQN_Learner = dqn(sample_size = num)
        env.close()
        reward_result9["sample size="+str(num)] = rewards
        print("\nCompleted")
    reward_df9 = pd.DataFrame.from_dict(reward_result9, orient='index').transpose()
    reward_df9 = reward_df9.rolling(window = 100).mean()
   
    plt.clf()
    reward_df9.plot()
    plt.title("DQN with Different Sample Size")
    plt.legend()
    plt.savefig("DQN2.png")
    print("\nPlot 9 finished")
    runtime.append(time.time())
    print("runtime: "+str((runtime[-1]-runtime[-2])/60))

# plot 10: DQN with different node size
if plot10:    
    print("Running plot 10: DQN with different hidden node size")
    reward_result10 = {}
    for node in [32,64,128]:
        env = gym.make('LunarLander-v2')
        env.seed(0)
        print("Training hidden node size="+str(node))
        rewards, DQN_Learner = dqn(hidden_nodes = node)
        env.close()
        reward_result10["hidden node size="+str(node)] = rewards
        print("\nCompleted")
    reward_df10 = pd.DataFrame.from_dict(reward_result10, orient='index').transpose()
    reward_df10 = reward_df10.rolling(window = 100).mean()
   
    plt.clf()
    reward_df10.plot()
    plt.title("DQN with Different Hidden Node Size")
    plt.legend()
    plt.savefig("DQN3.png")
    print("\nPlot 10 finished")
    runtime.append(time.time())
    print("runtime: "+str((runtime[-1]-runtime[-2])/60))
    
# plot 11: DQN with different frequency
if plot11:    
    print("Running plot 11: DQN with different frequency")
    reward_result11 = {}
    for freq in [3,4,5]:
        env = gym.make('LunarLander-v2')
        env.seed(0)
        print("Training frequency="+str(freq))
        rewards, DQN_Learner = dqn(frequency = freq)
        env.close()
        reward_result11["frequency="+str(freq)] = rewards
        print("\nCompleted")
    reward_df11 = pd.DataFrame.from_dict(reward_result11, orient='index').transpose()
    reward_df11 = reward_df11.rolling(window = 100).mean()
   
    plt.clf()
    reward_df11.plot()
    plt.title("DQN with Different Frequency")
    plt.legend()
    plt.savefig("DQN4.png")
    print("\nPlot 11 finished")
    runtime.append(time.time())
    print("runtime: "+str((runtime[-1]-runtime[-2])/60))
    
if plot12:     # plot 12: random action VS DQN trained agent
    print("Running plot 12: random action VS DQN trained agent")
    reward_result12 = {}
    reward_result12['Random Action'] = random_action(num_of_episodes=100)
    print("\nRandom Action Completed")
    print("Training Agent")
    env = gym.make('LunarLander-v2')
    env.seed(0)
    rewards, DQN_Learner = dqn(sample_size = 300, 
                            hidden_nodes = 64,
                            frequency = 4, 
                            terminate = False)
    env.close()
    print("\nTraining Completed")
    env = gym.make('LunarLander-v2')
    env.seed(0)
    rewards = DQN_action(DQN_Learner = DQN_Learner, num_of_episodes=100)
    reward_result12['Trained Agent'] = rewards
    reward_df12 = pd.DataFrame(reward_result12) 
    plt.clf()
    reward_df12.plot()
    plt.title("DQN Trained Agent VS Random Action")
    plt.legend()
    plt.savefig("DQN5.png")
    
    plt.clf()
    out = pd.cut(reward_result12['Trained Agent'], bins=[-500, 0, 100, 200, 500])    
    count = out.value_counts().tolist()
    x = np.arange(4)
    fig, ax = plt.subplots()
    rects = ax.bar(x = x, height = count, width = 0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(['less than 0', '0 to 100', '100 to 200', 'greater than 200'])
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 1),
                        textcoords="offset points",
                        ha='center', va='bottom')
    autolabel(rects)
    plt.title("DQN Trained Agent Performace")
    plt.savefig("DQN6.png")
    print("\nPlot 12&13 finished")
    runtime.append(time.time())
    print("runtime: "+str((runtime[-1]-runtime[-2])/60))
