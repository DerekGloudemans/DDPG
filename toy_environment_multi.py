import numpy as np
import matplotlib.pyplot as plt

class Multi_Car_Follow_1D():
    """
    A simplistic environment that models n vehicles (leader and followers) along
    a 1D track with first-order dynamics (position and velocity of each vehicle
    is tracked at each frame.) The environment returns at each step the state
    visible by the follower vehicle and expects the control acceleration as input
    """
    
    def __init__(self,agent_list = ["rand","step_accel"],sigma = 0.01,crash_penalty = -10000):
        self.n_agents = len(agent_list)

        
        self.all_acc =     [np.zeros(self.n_agents)]
        self.all_pos =     [np.arange(10*(self.n_agents-1),-1,-10)]
        self.all_spacing = [np.ones(self.n_agents)*10.0]
        self.all_vel =     [np.zeros(self.n_agents)]
        self.all_dv =      [np.zeros(self.n_agents)]
        self.all_rewards = [0]
        self.all_vel[0][0] = 1
        self.all_dv[0][1] = -1
        
        # store input params
        self.agent_types = agent_list
        self.sigma = sigma
        self.crash_penalty = crash_penalty
        self.step = 0
        
        # initialize state 
        
    def get_actions(self,model = None):
        """
        Generates visible state for each vehicle and queries appropriate action 
        function to get action. Returns a list of actions
        """
        actions = []
        
        for ag in range(self.n_agents):
            # get visible state
            state = np.array([self.all_spacing[-1][ag],self.all_vel[-1][ag],self.all_dv[-1][ag]])
            
            # query agent function for action
            if self.agent_types[ag] == "rand":
                        actions.append(np.random.normal(0,self.sigma))
                        
            elif self.agent_types[ag] == "step_accel":
                if state[0] > 10: #spacing > goal spacing
                            acc = 0.1
                            if state[2] > 0: # dv > 0
                                acc = acc - state[2]
                else:
                    acc = -0.05
                actions.append(acc)
        
            elif self.agent_types[ag] == "RL":
                actions.append(model.choose_actions(state))
                
        return actions
                
                
    def __call__(self,actions):
        """
        Expects control input for each car for timestep t. Calculates state t+1, 
        returns reward, and sets visible state for time t+1
        """
        # accelerations
        self.all_acc.append(actions)
        
        # positions
        positions = np.zeros(self.n_agents)
        for i in range(self.n_agents):
            positions[i] = self.all_pos[-1][i] + max(0,self.all_vel[-1][i]+0.5*actions[i])
        self.all_pos.append(positions)
        
        # velocities
        velocities = np.zeros(self.n_agents)
        for i in range(self.n_agents):
            velocities[i] = max(self.all_vel[-1][i]+actions[i], 0)
        self.all_vel.append(velocities)
        
        # spacings
        spacing = np.zeros(self.n_agents)
        for i in range(self.n_agents):
            if i == 0:
                spacing[i] = 10
            else:
                spacing[i] = self.all_pos[-1][i-1] - self.all_pos[-1][i] 
        self.all_spacing.append(spacing)
        
        # dv
        dv = np.zeros(self.n_agents)
        for i in range(self.n_agents):
            if i == 0: 
                dv[i] = 0
            else:
                dv[i] = self.all_vel[-1][i] - self.all_vel[-1][i-1]
        self.all_dv.append(dv) 
        
        # reward
        REW_WEIGHT = 100
        rew_vel = np.std(self.all_vel[-1]) * REW_WEIGHT
        rew_spacing = np.sum(np.abs(self.all_spacing[-1]-10.0)**2) 
        reward = -rew_vel -rew_spacing
        
        # end of episode penalties
        for i in range(1,self.n_agents):
            if self.all_spacing[-1][i] < 0 or self.all_spacing[-1][i] > 40:
                reward = self.crash_penalty
                break
        self.all_rewards.append(reward)
        
        self.step += 1
        
        # flatten reward for some reason
        try:
            reward = reward[0]
        except:
            pass
        

        return reward,self.step
 
    
    def show_episode(self,close = True):
        plt.style.use("seaborn")
        
        plt.figure(figsize = (30,10))
        plt.title("Single Episode")

        colors = np.random.rand(self.n_agents,3)
        
        for i in range(len(self.all_pos)):
            plt.subplot(311)
            
            for j in range(len(self.all_pos[0])):
                plt.scatter(self.all_pos[i][j],1,color = colors[j])
                
            reward = round(self.all_rewards[i] *1000)/1000.0
            plt.annotate("Reward: {}".format(reward),(self.all_pos[i][1]-5,1))

            center = self.all_pos[i][0]
            plt.xlim([center -40*self.n_agents, center + 10])
            plt.xlabel("Position")

            
        
            plt.subplot(312)
            plt.plot(self.all_rewards[:i])
            plt.ylabel("Reward")
            plt.xlabel("Timestep")
            plt.xlim([0,len(self.all_rewards)])
            plt.legend(["Reward"])
            
            
            
            plt.draw()
            plt.pause(0.01)
            plt.clf()
        if close:
            plt.close()

if True and __name__ == "__main__":        
    # test code
    agent_list = ["rand","step_accel","step_accel","step_accel","step_accel","step_accel"]
    env = Multi_Car_Follow_1D(agent_list = agent_list)
    for i in range(0,200):
        actions = env.get_actions()
        reward,step = env(actions)

    env.show_episode()