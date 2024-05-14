# python related
import numpy as np
import random
import pickle
from collections import deque

# training related
import torch

from osim.env import L2M2019Env

# others
# import wandb


# Environment and model hyperparameters
STATE_SIZE = 2 * 11 * 11 + 97  # Dimensions of the game observation
ACTION_SIZE = 22  # Number of valid actions in the game
HIDDEN_SIZE = 256
GAMMA = 0.99  # Discount factor
TAU = 0.01
LR_ACTOR = 0.0005
LR_CRITIC = 0.0005
LR_ALPHA = 0.0005
BATCH_SIZE = 256  # Batch size for training
MEMORY_SIZE = 4000000  # Size of the replay memory buffer

# check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNetwork(torch.nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(STATE_SIZE + ACTION_SIZE, HIDDEN_SIZE) # instead of concate action after fc1
        self.fc2 = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc4 = torch.nn.Linear(HIDDEN_SIZE, 1) # instead of concate action after fc1
        # self.fc3 = torch.nn.Linear(64, 1)

        self.init_weight()

    def init_weight(self):
        # Initialize weights using Xavier initialization
        torch.nn.init.xavier_uniform_(self.fc1.weight, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.fc2.weight, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.fc3.weight, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.fc4.weight)

        # Initialize biases to zero
        self.fc1.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)
        self.fc3.bias.data.fill_(0)
        self.fc4.bias.data.fill_(0)

    def forward(self, state, action):
        x = torch.cat([state, action], 1) # concatenate with action
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
class CriticNetwork(torch.nn.Module):
    def __init__(self):
        super(CriticNetwork, self).__init__()
        self.q_net_1 = QNetwork()
        self.q_net_2 = QNetwork()

    def forward(self, state, action):
        action = action.clone()
        q_value_1 = self.q_net_1.forward(state, action)
        q_value_2 = self.q_net_2.forward(state, action)
        return q_value_1, q_value_2
    
class PolicyNetwork(torch.nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(STATE_SIZE, HIDDEN_SIZE) 
        self.fc2 = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc_mean = torch.nn.Linear(HIDDEN_SIZE, ACTION_SIZE)
        self.fc_log_std = torch.nn.Linear(HIDDEN_SIZE, ACTION_SIZE)
        self.init_weight()

    def init_weight(self):
        # Initialize weights using Xavier initialization
        torch.nn.init.xavier_uniform_(self.fc1.weight, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.fc2.weight, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.fc3.weight, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.fc_mean.weight)
        torch.nn.init.xavier_uniform_(self.fc_log_std.weight)

        # Initialize biases to zero
        self.fc1.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)
        self.fc3.bias.data.fill_(0)
        self.fc_mean.bias.data.fill_(0)
        self.fc_log_std.bias.data.fill_(0)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)

        # round_mean = [round(x, 2) for x in mean[0].tolist()]
        # round_std = [round(x, 2) for x in log_std[0].exp().tolist()]
        # if np.random.rand() < 0.01:
        #     print("mean:", round_mean, "log std:", round_std) # => always 0 and 1
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample() # the reparameterization trick, (x_t is tensor)
        action = torch.tanh(x_t)

        # Enforcing Action Bound (pranz24 ver)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True) 

        # change action = sampled or tanh(mean) according to mode
        if not self.training:
            action = torch.tanh(mean)

        action = torch.clamp(action, min=0, max=1) # dk if it works

        return action, log_prob 

class Agent:
    def __init__(self):

        self.alpha = 0.2
        self.tau = TAU
        self.update_interval = 1
        # critic network
        self.critic = CriticNetwork()#.to(device)
        self.critic_target = None
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        # actor network
        self.policy = PolicyNetwork()#.to(device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=LR_ACTOR)
        ## entropy
        self.target_entropy = -torch.prod(torch.Tensor(ACTION_SIZE)).item()
        ## alpha
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=LR_ALPHA)

        # replay buffer
        self.memory = deque(maxlen=MEMORY_SIZE)

        # others
        self.processed_obs = None
        self.load_test("110060062_hw4_data")
        
    def init_target_model(self): # used only before training
        self.critic_target = CriticNetwork()#.to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for param in self.critic_target.parameters():
            param.requires_grad = False
    
    def extract_scalar(self, data):
        scalars = []

        def extract(data):
            if isinstance(data, dict):
                for value in data.values():
                    extract(value)
            elif isinstance(data, list):
                for item in data:
                    extract(item)
            else:
                scalars.append(data)

        extract(data)
        return scalars

    def process_obs(self, obs):
        obs = list(obs.values())
        obs[0] = torch.flatten(torch.tensor(obs[0]))
        for i in range(1, 4):
            obs[i] = self.extract_scalar(obs[i].copy())
            obs[i] = torch.tensor(obs[i])
        flat_obs = torch.cat((obs[0], obs[1], obs[2], obs[3]), dim=-1).to(torch.float32)
        return flat_obs#.to(device)

    def act(self, observation):
        # process observation
        flat_obs = self.process_obs(observation)
        action, _ = self.policy.sample(flat_obs)
        
        self.processed_obs = flat_obs

        return action.detach().cpu().numpy()
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        minibatch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*minibatch)


        states = torch.stack(states).float().to(device)
        actions = torch.stack(actions).float().to(device)
        rewards = torch.FloatTensor(rewards).float().unsqueeze(1).to(device)
        next_states = torch.stack(next_states).float().to(device)
        dones = torch.FloatTensor(dones).float().unsqueeze(1).to(device)

        # update critic
        with torch.no_grad(): # according to paper equation 6
            next_states_action, next_states_log_pi = self.policy.sample(next_states) # a_t+1, log_pi(a_t+1)
            q1_next_target, q2_next_target = self.critic_target(next_states, next_states_action) # Q(s_t+1, a_t+1)
            min_q_next_target = torch.min(q1_next_target, q2_next_target) - self.alpha * next_states_log_pi # bootstrap = Q(s_t+1, a_t+1) - alpha * log_pi(a_t+1)
            next_q_value = rewards + GAMMA * min_q_next_target * (1 - dones) # y = r_t + gamma * bootstrap * (1-dones)
        q1, q2 = self.critic(states, actions) # Q(s_t, a_t)
        q1_loss = torch.nn.MSELoss()(q1, next_q_value)
        q2_loss = torch.nn.MSELoss()(q2, next_q_value)
        q_loss = q1_loss + q2_loss
        
        # wandb.log({"q loss": q_loss.item()})
        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()

        for p in self.critic.parameters(): # freeze q
            p.requires_grad = False

        # update policy
        pis, log_pis,  = self.policy.sample(states) # pi(s_t), log_pi(a_t)
        q1_pi, q2_pi = self.critic(states, pis)
        min_q_pi = torch.min(q1_pi, q2_pi) # Q(s_t, f)

        policy_loss = ((self.alpha * log_pis) - min_q_pi).mean() # J_pi = E[(alpha * log_pi) - Q]
        # wandb.log({"pi loss": policy_loss.item()})
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        for p in self.critic.parameters(): # unfreeze q
            p.requires_grad = True

        # update alpha
        alpha_loss = -(self.log_alpha * (log_pis + self.target_entropy).detach()).mean() # not on paper
        # wandb.log({"alpha loss": alpha_loss.item()})
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        self.soft_update(self.critic_target, self.critic, self.tau)
        
    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def load_test(self, pi_name):
        self.policy.eval()
        # self.policy.load_state_dict(torch.load(pi_name))
        self.policy.load_state_dict(torch.load(pi_name, map_location=torch.device('cpu')))

    def load_cont_train(self, pi_name, q_name):
        # load policy weights
        self.policy.load_state_dict(torch.load(pi_name))
        # load critic weights
        self.critic.load_state_dict(torch.load(q_name))
        # load replay buffer
        with open('memory.pkl', 'rb') as f:
            self.memory = pickle.load(f)

    def save(self, pi_name, q_name):
        # save policy weights
        policy_weights = self.policy.state_dict()
        torch.save(policy_weights, pi_name)
        # save critic weights
        critic_weights = self.critic.state_dict()
        torch.save(critic_weights, q_name)
        # save replay buffer for continue the training
        with open('memory.pkl', 'wb') as f:
            pickle.dump(self.memory, f)

if __name__ == '__main__':
    
    env = L2M2019Env(visualize=False)
    obs = env.reset()

    agent = Agent()

    done = False
    total_reward = 0
    
    while not done:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)

        total_reward += reward
        env.render()

    print("score:", total_reward)
    env.close()