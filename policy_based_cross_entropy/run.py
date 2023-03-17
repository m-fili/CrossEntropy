import numpy as np


def policy_based_run(agent, optimizer, n_iterations=1000, t_max=1000, gamma=1.0):
    assert hasattr(optimizer, '__class__'), 'optimizer should be of type class.'
    assert hasattr(agent, '__class__'), 'agent should of type class.'

    PRINT_EVERY = 10
    TARGET_SCORE = 90.0
    scores = []
    best_p = agent.model.get_params_array()
    best_p = 0.5 * np.random.randn(best_p.shape[0])
    details = {}

    for n in range(1, n_iterations + 1):

        x_pop = optimizer.generate_next_population(best_p)
        f_pop = np.array([agent.calculate_return(p, gamma, t_max) for p in x_pop])
        best_p = optimizer.find_new_candidate(x_pop, f_pop)
        G = agent.calculate_return(best_p, gamma, t_max)
        agent.model.set_params_array(best_p)
        scores.append(G)
        details[n] = {'score': G,
                      'params': agent.model.get_params_array(),
                      'f_pop': f_pop,
                      'x_pop': x_pop,
                      'best_p': best_p}

        print(f'Episode {n:03}/{n_iterations}: Score={scores[-1]:.2f}', end='\r')
        if n % PRINT_EVERY == 0:
            S = np.mean(scores[-PRINT_EVERY:])
            print(f'Episode {n:03}/{n_iterations}: Average Score ={S:.3f}')

        if np.mean(scores[-PRINT_EVERY:]) >= TARGET_SCORE:
            print(f'Agent reached the target score of {TARGET_SCORE} in {n} episodes!')
            break

    return scores
