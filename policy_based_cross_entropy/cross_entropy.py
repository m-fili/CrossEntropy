import numpy as np


class CrossEntropy:
    """
    Maximization Problem.
    """

    def __init__(self, n_params, pop_size=50, elite_frac=0.2, noise_scale=0.5):
        self.noise_scale = noise_scale
        self.n_params = n_params
        self.pop_size = pop_size
        self.elite_frac = elite_frac
        self.n_elite = int(pop_size * elite_frac)

    def find_new_candidate(self, x_pop, f_pop):
        elite_indx = np.argsort(f_pop)[-self.n_elite:]
        params_elite = np.array([x_pop[indx] for indx in elite_indx])
        return params_elite.mean(axis=0)

    def generate_next_population(self, x):
        x_pop = [x + self.noise_scale * np.random.randn(self.n_params) for _ in range(self.pop_size)]
        return x_pop
