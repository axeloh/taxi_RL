#!/usr/bin/env python
# coding: utf-8

# In[2]:


import gym


# #### Representations
# 
# | --> WALL (Can't pass through, will remain in the same position if tries to move through wall)
# 
# Yellow --> Taxi Current Location
# 
# Blue --> Pick up Location
# 
# Purple --> Drop-off Location
# 
# Green --> Taxi turn green once passenger board
# 
# Letters --> Locations

# In[3]:


env = gym.make('Taxi-v2').env # Env is the unified environment interface


# Following are the **env** methods that could be quite helpful to us: <br />
# - env.reset(): Resets the environment and returns a random initial state
# - env.step(action): Step the environment by one timestep. Returns
#  - observation: Observation of the environment
#  - reward: If your action was beneficial or not
#  - done: Indicates if we have successfully picked up and dropped off a passenger, also called one *episode*
#  - info: Addition info such as performance and latency for debugging purposes
# - env.render(): Renders one frame of the environment 

# In[4]:


env.reset()
env.render()


# In[5]:


env.observation_space.n # Total number of states


# ##### Actions (6 in total)
# 0: move south <br />
# 1: move north <br />
# 2: move east <br />
# 3: move west <br />
# 4: pickup passenger <br />
# 5: dropoff passenger <br />

# In[6]:


env.action_space.n # Total number of actions


# In[7]:


state = env.encode(3, 1, 2, 0) # Taxi row, taxi column, passenger index, destination index
print('State: ', state)
env.s = state
env.render()
print(env.step(3))


# In[8]:


env.P[328] # Structure of dictionary: {action: [(probability, nextstate, reward, done)]}


# In[9]:


from IPython.display import clear_output
from time import sleep

def print_frames(frames, delay):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'])
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(delay)


# #### Brute Force algorithm 

# In[10]:


# Let's see what happens if we try to brute force,
# meaning choosing random actions until passenger is picked up and dropped of at right destination

env.s = 328 # set environment to the illustration state above

epochs = 0
penalties, reward = 0, 0
frames = [] # for animation
done = False 

while not done:
    action = env.action_space.sample() # Chosing random action
    state, reward, done, info = env.step(action) # Extracting info

    if reward == -10:
        penalties += 1
        
    # Put each rendered frame into dict for animation
    frames.append({
        'frame': env.render(mode='ansi'),
        'state': state,
        'action': action,
        'reward': reward
        }
    )

    epochs += 1

print_frames(frames, 0.05)
print("Timesteps taken: {}".format(epochs))
print("Penalties incurred: {}".format(penalties))


# ### Using Reinforcement Learning
# We are going to use a simple RL algorithm called Q-learning which will give our agent some memory.
# We use a Q-table with Q-values (Q: quality) that states the quality of an state-action-combination.
# Q-values are initialized to arbitrary values, and as the agent exposes itself to the environment and receives different rewards by executing different actions, the Q-values are updated using the equation:
# 
# **Q(state, action) = (1−α)Q(state, action) + α( reward + γ*max_a {Q(next state,all actions)} )**

# The Q-table is a matrix where we have a row for every state (500) and a column for every action (6). It's first initialized to 0, and then values are updated after training. Note that the Q-table has the same dimensions as the reward table, but it has a completely different purpose.

# After enough random exploration of actions, the Q-values tend to converge serving our agent as an action-value function which it can exploit to pick the most optimal action from a given state.
# 
# There's a tradeoff between exploration (choosing a random action) and exploitation (choosing actions based on already learned Q-values). We want to prevent the action from always taking the same route, and possibly overfitting, so we'll be introducing another parameter called ϵ "epsilon" to cater to this during training.
# 
# Instead of just selecting the best learned Q-value action, we'll sometimes favor exploring the action space further. Higher epsilon value results in episodes with more penalties (on average) which is obvious because we are exploring and making random decisions.

# In[11]:


# ** TRAINING THE AGENT **
import numpy as np
q_table = np.zeros([env.observation_space.n, env.action_space.n]) # Initialize the Q-table to a 500 x 6 matrix of zeros
q_table


# In[12]:


import random
from IPython.display import clear_output

# Hyper parameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

# For plotting metrics
all_epochs = []
all_penalties = []
randoms = 0

for i in range(1, 50000):
    state = env.reset()
    
    epochs, penalties, reward = 0, 0, 0
    done = False
    
    while not done:
        if random.uniform(0, 1) < epsilon:
            randoms += 1
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[state]) #Exploit learned values
        
        next_state, reward, done, info = env.step(action)
        
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        new_value = (1-alpha)*old_value + alpha*(reward + gamma*next_max)
        q_table[state, action] = new_value
        
        if reward == -10:
            penalties += 1
        
        state = next_state
        epochs += 1
        
        if i % 100 == 0:
            clear_output(wait=True)
            print('Episode: {}'.format(i))
    
print("Training finished. \n")
            
    


# In[13]:


"""Evaluate agent's performance after Q-learning"""

total_epochs, total_penalties = 0, 0
episodes = 100

for _ in range(episodes):
    state = env.reset()
 
    epochs, penalties, reward = 0, 0, 0
    done = False
    
    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)
        
        if reward == -10:
            penalties += 1

        epochs += 1
        
    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")
print(q_table[328])


# In[14]:


env.s = random.randint(0, 500) # set environment to illustration state
state = env.reset()
epochs = 0
penalties, reward = 0, 0
frames = [] # for animation
done = False 
i = 0
while not done:
    action = np.argmax(q_table[state])
    state, reward, done, info = env.step(action)

    if reward == -10:
        penalties += 1
        
    # Put each rendered frame into dict for animation
    frames.append({
        'frame': env.render(mode='ansi'),
        'state': state,
        'action': action,
        'reward': reward
        }
    )

    epochs += 1
    
print_frames(frames, 0.5)
print("Timesteps taken: {}".format(epochs))
print("Penalties incurred: {}".format(penalties))


# In[ ]:





# In[ ]:




