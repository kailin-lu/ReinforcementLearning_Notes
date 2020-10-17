import torch
from torch.functional import F
from torch.optim import Adam


class MLP(torch.nn.Module):
    def __init__(self, input_size, l1_size, l2_size, output_size, lr):
        """
        Constructs a MLP outputting softmax probabilities for a
        Categorical distribution
        """
        self.l1 = torch.nn.Linear(input_size, l1_size)
        self.l2 = torch.nn.Linear(l1_size, l2_size)
        self.l3 = torch.nn.Linear(l2_size, output_size)
        self.optimizer = Adam(self.parameters(), lr=lr)

    def forward(self, x):
        l1 = F.tanh(self.l1(x))
        l2 = F.tanh(self.l2(l1))
        pred = F.softmax(self.l3(l2))
        return pred


class PPOAgent(object):
    def __init__(self, env, l1_size, l2_size,
                 actor_lr, critic_lr, gamma, epsilon,
                 n_trajectories, iterations, max_steps_per_episode):
        self.env = env
        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_trajectories = n_trajectories
        self.iterations = iterations
        self.max_steps_per_episode = max_steps_per_episode
        self.actor = MLP(input_size=self.observation_dim, output_size=self.action_dim,
                         l1_size=l1_size, l2_size=l2_size, lr=actor_lr)
        self.critic = MLP(input_size=self.observation_dim, output_size=1,
                          l1_size=l1_size, l2_size=l2_size, lr=critic_lr)

    def sample_action(self, obs):
        """
        Sample and action and calculate the log likelihood
        """
        probs = self.actor(obs)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample(1)
        logp = dist.log_prob(action)
        return action.item(), logp

    def sample_trajectory(self):
        trajectories = []
        # TODO: parallelize
        for i in range(self.n_trajectories):
            trajectory = []
            obs = self.env.reset()
            for j in range(self.max_steps_per_episode):
                action, logp = self.sample_action(obs)

    def advantage(self):
        """
        A_t = log_likelihood(action_t) - V_t
        """
        pass

    def train(self):
        for i in range(self.iterations):
            pass

