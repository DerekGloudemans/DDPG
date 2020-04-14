from envs.toy_environment import Car_Follow_1D
from ddpg_torch_copy import Agent
from ddpg_torch_copy_safe import Agent as Agent_Safe
#from DDPG_pytorch_phil_tabor.ddpg_torch import Agent

import _pickle as pickle
import random
import numpy as np
import os

#activation_fn = "sigmoid"
activation_fn = "tanh"

#define agent
agent = Agent_Safe(alpha=0.001, beta=0.01, input_dims=[3], tau=0.002, env=None,
              batch_size=64,  layer1_size=10, layer2_size=10, max_size = 5000, n_actions=1, activation = activation_fn)
agent.load_models()
best_score = -10000
score_history = []
episode_history = []
crash_penalty = -10000

print ("Starting Episodes")
# for each loop, reset environment with new random seed
for i in range(5000):
    # to create unique episodes
    np.random.seed(i)
    random.seed(i)
    
    # define environment
    env = Car_Follow_1D(sigma = 0.1,crash_penalty = crash_penalty) 
    obs = env.vis_state
    done = False
    score = 0
    while not done:
        act = agent.choose_action(obs)
        
        if activation_fn == "tanh":
            act = act*0.1
        else:
            act = (act-0.5)*0.2 # accelerations in range (-0.1,0.1)
        # shift action into reasonable range
        
        new_state,reward,step_counter = env(act)
       
        # set done
        done = 0
        if reward <= crash_penalty or step_counter > 500: # terminal state
            done = 1

        obs = obs.reshape(-1).astype(float)
        new_state = new_state.reshape(-1).astype(float)

        # store memory
        agent.remember(obs, act, reward, new_state, int(done))
        
        # update actor and critic networks
        agent.learn()
        
        # append reward to total score for episode
        score += reward
        obs = new_state
        
    # at end of episode, store score_history
    score_history.append(score)
    avg_score = score/step_counter 
    
    # periodically save checkpoints of model states
    if score/step_counter > best_score:
        best_score = score/step_counter
        agent.save_models()
        env.show_episode()

        # store episode history in file
        episode_history.append(env)
        with open(os.path.join("model_current","episode_history"),'wb') as f:
            pickle.dump(episode_history,f)

    if i % 100 == 0:
        env.show_episode()
    
    # decay model
    agent.noise.reset()
    
    print('Episode {} average score: {}'.format(i,score/step_counter))

episode_history[-1].show_episode()