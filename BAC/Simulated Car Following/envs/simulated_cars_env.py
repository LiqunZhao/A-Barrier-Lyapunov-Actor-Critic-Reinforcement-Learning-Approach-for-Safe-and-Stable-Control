import numpy as np
import gym
from gym import spaces

from rcbf_sac.utils import prYellow
from rcbf_sac.utils import prRed

class SimulatedCarsEnv(gym.Env):
    """Simulated Car Following Env, almost identical to https://github.com/rcheng805/RL-CBF/blob/master/car/DDPG/car_simulator.py
    Front <- Car 1 <- Car 2 <- Car 3 <- Car 4 (controlled) <- Car 5
    """

    metadata = {'render.modes': ['human']}

    def __init__(self,seed):

        super(SimulatedCarsEnv, self).__init__()

        self.dynamics_mode = 'SimulatedCars'
        self.action_space = spaces.Box(low=-6.5, high=6.5, shape=(1,))
        self.safe_action_space = spaces.Box(low=-6.5, high=6.5, shape=(1,))  # be the same with the action space.
        self.observation_space = spaces.Box(low=-1e10, high=1e10, shape=(10,))
        self.max_episode_steps = 300
        self.dt = 0.02
        self.act_acc_coef = 1.0
        np.random.seed(seed)

        # Gains
        self.kp = 4.0
        self.k_brake = 20.0

        self.state = None  # State [x_1 v_1 ... x_5 v_5]
        self.t = 0  # Time
        self.episode_step = 0  # Episode Step
        self.should_keep = 9.5
        self.should_keep_thre = 0.5
        self.reward_goal = 1.5

        # Gaussian Noise Parameters on the accelerations of the other vehicles
        self.disturb_mean = np.zeros((1,))
        self.disturb_covar = np.diag([0.2 ** 2])

        self.reset()  # initialize the env
        self.safety_cost_coef = 1.0

    def step(self, action):
        """

        """

        # Current State
        pos = self.state[::2]
        vels = self.state[1::2]

        self.record_action = action

        # Accelerations
        vels_des = 3.0 * np.ones(5)  # Desired velocities
        vels_des[0] -= 4 * np.sin( self.t)  # modify the desired velocity of the fist car
        accels = self.kp * (
                    vels_des - vels)  # calculate the acc of the 5 cars (including 4th one and the acc of the 4th car is not solely dicided by the input action)
        accels[1] += -self.k_brake * (pos[0] - pos[1]) * (
                    (pos[0] - pos[1]) < 6.5)  # change the velocity of the 2,3,5 cars.
        accels[2] += -self.k_brake * (pos[1] - pos[2]) * ((pos[1] - pos[2]) < 6.5)
        accels[3] = 0.0
        accels[4] += -self.k_brake * (pos[2] - pos[4]) * (
                    (pos[2] - pos[4]) < 13.0)  # can be changed according to the simulation result.

        # Unknown part to the dynamics
        accels *= 1.1

        self.previous_positions = np.array([self.state[4], self.state[5],self.state[6],self.state[7]])

        # Determine action of each car
        f_x = np.zeros(10)
        g_x = np.zeros(10)

        f_x[::2] = vels  # Derivatives of positions are velocities
        f_x[1::2] = accels  # Derivatives of velocities are acceleration
        f_x[6] = 0.0
        f_x[7] = 0.0
        g_x[6] = 1.0
        g_x[7] = self.act_acc_coef / self.dt  # Car 4's velocity is the control input

        new_state = np.expand_dims(self.state,1)

        tran_mat = np.diag(np.ones(10))

        tran_mat[7,7] = 0.0           # for the dynamics of the 4th car
        res = np.dot(tran_mat,new_state).squeeze(-1)

        self.state = res + self.dt * (f_x + g_x * action)

        self.t = self.t + self.dt  # time

        self.episode_step += 1  # steps in episode

        info = dict()


        self.third_pos = self.state[4]
        self.fourth_pos = self.state[6]
        self.next_positions = np.array([self.state[4], self.state[5], self.state[6], self.state[7]])
        self.fifth_pos = self.state[8]

        self.distance_third_fourth = self.third_pos - self.fourth_pos

        reward = self._get_reward(action[0])  # need to pay attention to the format of the action.


        satisfied_num = 0
        if (abs(self.distance_third_fourth - self.should_keep) < self.should_keep_thre):
            satisfied_num = 1
            prRed('The desired region is reached!!!!!!!!!!')
            reward = reward + self.reward_goal

        info['reached'] = satisfied_num


        done = self.episode_step >= self.max_episode_steps  # done?

        info['goal_met'] = False

        # Include the cost in the info
        num_safety_violation = 0
        safety_cost_val = 0
        if ((self.third_pos - self.fourth_pos) < 3.5):
            num_safety_violation = num_safety_violation + 1
            prYellow('In front or close to the third one.')
            dist_third_fourth = (self.third_pos - self.fourth_pos)
            safety_cost_val = safety_cost_val + np.abs(dist_third_fourth - 3.5) * self.safety_cost_coef

        if ((self.fourth_pos - self.fifth_pos) < 3.5):
            num_safety_violation = num_safety_violation + 1
            prYellow('Behind or close to the fifth one.')
            dist_fourth_fourth = (self.fourth_pos - self.fifth_pos)
            safety_cost_val = safety_cost_val + np.abs(dist_fourth_fourth - 3.5) * self.safety_cost_coef


        info['num_safety_violation'] = num_safety_violation
        info['safety_cost'] = safety_cost_val                     # Used by other algorithms like CPO, PPO-Lag and TRPO-Lag


        constraint = abs(self.distance_third_fourth - self.should_keep)

        return self._get_obs(), reward, constraint, self.previous_positions, self.next_positions, done, info

    def _get_reward(self, action):

        self.reward_action = -0.5 * np.square(action - 3.0) / self.max_episode_steps

        return self.reward_action

    def reset(self):
        """ Reset the state of the environment to an initial state.

        Returns
        -------
        observation : ndarray
            Next observation.
        """

        self.t = 0
        self.state = np.zeros(10)  # first col is pos, 2nd is vel
        self.state[::2] = [42.0, 34.0, 26.0, 18.0, 10.0]  # initial positions
        self.state[1::2] = 3.0 + np.random.normal(0, 0.5)  # initial velocities
        self.state[7] = 3.0  # initial velocity of car 4


        self.episode_step = 0
        self.num_conti_follow = 0

        return self._get_obs()

    def render(self, mode='human', close=False):
        """Render the environment to the screen

        Parameters
        ----------
        mode : str
        close : bool

        Returns
        -------

        """

        print('Ep_step = {}, action={}, first = {:.4f}, third = {:.4f},{:.4f}, fourth = {:.4f},{:.4f}, fifth = {:.4f},{:.4f}, diff1 = {:.4f}, diff2 = {:.4f}, reward_act = {:.4f}, reward_dis = {:.4f}'.format(self.episode_step, self.record_action,self.state[1], self.state[4],self.state[5],self.state[6],self.state[7],self.state[8],self.state[9],
                                                                                                       self.state[4]-self.state[6],self.state[6]-self.state[8],self.reward_action,self.reward_distance))


    def _get_obs(self):
        """Given the state, this function returns it to an observation akin to the one obtained by calling env.step

        Parameters
        ----------

        Returns
        -------
        observation : ndarray
          Observation: [car_1_x, car_1_v, car_1_a, ...]
        """

        obs = np.copy(np.ravel(self.state))
        obs[::2] /= 100.0  # scale positions
        obs[1::2] /= 30.0  # scale velocities
        return obs
