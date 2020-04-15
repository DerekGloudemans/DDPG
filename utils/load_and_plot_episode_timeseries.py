from envs.environment_multi_with_IDM import Multi_Car_Follow
from ddpg_torch_copy_safe import Agent
import _pickle as pickle
import numpy as np

with open("models//Model 15//episode_history.cpkl",'rb') as f:
    output = pickle.load(f)
    
for i in range(0,len(output)):
    print(i,np.mean(output[i].all_rewards))
    
output[-1].show_episode(SAVE = True)