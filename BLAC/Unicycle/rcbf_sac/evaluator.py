import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from rcbf_sac.utils import *


class Evaluator(object):            # Used in other algorithm, not used in this BLAC framework

    def __init__(self, num_episodes, interval, save_path=''):
        self.num_episodes = num_episodes
        self.interval = interval
        self.save_path = save_path
        self.results = np.array([]).reshape(num_episodes, 0)
        self.constraints = np.array([]).reshape(num_episodes, 0)

    def __call__(self, env, policy, cbf_wrapper=None, dynamics_model=None, debug=False, visualize=False, save=True,save_constraint=True):

        self.is_training = False
        result = []
        cons = []

        for episode in range(self.num_episodes):

            # reset at the start of episode
            observation = env.reset()
            self.center_pos = np.array([-2.47,-2.5])


            # Make sure to start from a safe state
            if cbf_wrapper and dynamics_model:
                out = None
                while out is None or cbf_wrapper.get_min_h_val(out) < 1e-4:
                    observation = env.reset()
                    state = dynamics_model.get_state(observation)
                    out = dynamics_model.get_output(state)

            episode_steps = 0
            episode_reward = 0.

            assert observation is not None

            # start episode
            done = False
            single_episode_constraints = []
            while not done:
                # basic operation, action ,reward, blablabla ...
                action = policy(observation,self.center_pos)

                observation, reward,constraint,center_pos,next_center_pos, done, info = env.step(action)

                self.center_pos = next_center_pos

                if visualize:
                    env.render(mode='human')

                # update
                episode_reward += reward
                episode_steps += 1
                single_episode_constraints.append(constraint)


            episode_final_average_constraint = np.mean( single_episode_constraints[-5:])
            if debug: prYellow('[Evaluate] #Episode{}: episode_reward:{},episode_constraint:{}'.format(episode, episode_reward,episode_final_average_constraint))
            result.append(episode_reward)
            cons.append(episode_final_average_constraint)

        result = np.array(result).reshape(-1, 1)
        cons = np.array(cons).reshape(-1, 1)
        self.results = np.hstack([self.results, result])
        self.constraints = np.hstack([self.constraints, cons])

        if save:
            self.save_results('{}/validate_reward'.format(self.save_path))

        if save_constraint:
            self.save_constraint('{}/validate_constraint'.format(self.save_path))

        return np.mean(result), np.mean(cons)

    def save_results(self, fn):

        y = np.mean(self.results, axis=0)
        error = np.std(self.results, axis=0)

        x = range(0, self.results.shape[1] * self.interval, self.interval)
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        plt.xlabel('Timestep')
        plt.ylabel('Average Reward')
        ax.errorbar(x, y, yerr=error, fmt='-o')
        plt.savefig(fn + '.png')
        savemat(fn + '.mat', {'reward': self.results})
        plt.close()

    def save_constraint(self, fn):

        y = np.mean(self.constraints, axis=0)
        error = np.std(self.constraints, axis=0)

        x = range(0, self.constraints.shape[1] * self.interval, self.interval)
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        plt.xlabel('Timestep')
        plt.ylabel('Last Average Constraint')
        ax.errorbar(x, y, yerr=error, fmt='-o')
        plt.savefig(fn + '_constraint.png')
        savemat(fn + '_constraint.mat', {'last_average_constraint': self.constraints})
        plt.close()