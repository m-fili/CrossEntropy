import torch


class Agent:

    def __init__(self, env, model):

        device = torch.device("cpu")
        self.env = env
        self.n_states = env.observation_space.shape[0]
        self.n_action = env.action_space.shape[0]
        self.model = model.to(device)

    def select_action(self, state):
        return self.model.forward(state).cpu().detach().numpy()

    def calculate_return(self, params, gamma=1.0, t_max=1000):
        """
        Calculates discounted return for a completed episode using the current policy in agent.
        """
        s1, _ = self.env.reset()
        rewards = []

        # first set the weights
        self.model.set_params_array(params)

        for t in range(t_max):
            action = self.select_action(s1)
            s2, reward, done, info, _ = self.env.step(action)
            rewards.append(reward)
            s1 = s2
            if done:
                break
        # calculate the total discounted return for the episode
        discounts = [gamma ** i for i in range(len(rewards))]
        G = sum([a * b for a, b in zip(rewards, discounts)])

        return G

    def load_best_model(self, file_path):
        self.model.load_state_dict(torch.load(file_path))
