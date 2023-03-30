import argparse
from utils.logx import EpochLogger
import torch
import numpy as np
import random

from rcbf_sac.sac_cbf import RCBF_SAC
from rcbf_sac.replay_memory import ReplayMemory
from rcbf_sac.dynamics import DynamicsModel
from build_env import *
import os

from rcbf_sac.utils import prGreen, get_output_folder, prYellow
from rcbf_sac.evaluator import Evaluator
from rcbf_sac.generate_rollouts import generate_model_rollouts
import wandb

def train(agent, env, dynamics_model, args,logger_kwargs=dict()):
    # To log the data
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Memory
    memory = ReplayMemory(args.replay_size, args.seed)
    memory_model = ReplayMemory(args.replay_size, args.seed)  # Used in other algorithm. In this algorithm this "memory_model" will not be used and can be left.

    # Training Loop
    total_numsteps = 0
    updates = 0


    for i_episode in range(args.max_episodes):

        positions_record = []

        episode_reward = 0
        episode_cost = 0
        episode_safety_cost = 0.0              # There is an additional term called safety cost and will be used to record the safety cost of each epsiode. Used by CPO, PPO-Lag and TRPO-Lag to form constraints
        episode_steps = 0
        done = False
        obs = env.reset()
        cur_cen_pos = np.array([-2.47,-2.5])          # Current position. Here we use this state as the input of the Lyapunov network.
        next_center_pos = np.array([-2.47, -2.5])           # Next position. Same with the current information at the beginning of the one episode. Later will be different.



        while not done:
            if episode_steps % 300 == 0:
                distance = np.linalg.norm((next_center_pos - [2.5,2.5]))
                prYellow('Episode {} - step {} - eps_rew {} - eps_cost {} - eps_safety_cost {} - current position {} - current distance {} '.format(i_episode, episode_steps, episode_reward, episode_cost,episode_safety_cost, next_center_pos, distance))
            state = dynamics_model.get_state(obs)


            # If using model-based RL then we only need to have enough data for the real portion of the replay buffer
            if len(memory) + len(memory_model) * args.model_based > args.batch_size:

                # Number of updates per step in environment
                for i in range(args.updates_per_step):

                    # Update parameters of all the networks
                    if args.model_based:
                        # Pick the ratio of data to be sampled from the real vs model buffers
                        real_ratio = max(min(args.real_ratio, len(memory) / args.batch_size), 1 - len(memory_model) / args.batch_size)
                        # Update parameters of all the networks
                        critic_1_loss, critic_2_loss,lyapunov_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory,
                                                                                                             args.batch_size,
                                                                                                             updates,
                                                                                                             dynamics_model,
                                                                                                             memory_model,
                                                                                                             real_ratio)
                    else:
                        critic_1_loss, critic_2_loss,lyapunov_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory,
                                                                                                           args.batch_size,
                                                                                                           updates,
                                                                                                           dynamics_model)



                    logger.store(critic_1_loss=critic_1_loss)
                    logger.store(critic_2_loss=critic_2_loss)
                    logger.store(lyapunov_loss=lyapunov_loss)
                    logger.store(policy_loss=policy_loss)
                    logger.store(ent_loss=ent_loss)
                    logger.store(alpha=alpha)


                    updates += 1


            action = agent.select_action(obs, dynamics_model, cur_cen_pos,
                                         warmup=args.start_steps > total_numsteps)  # Use the RL-based controller to have the control signal. Since here we only have CBF constraints, there is no need to use backup controller when no feasible solution exists when CBF and CLF constraints cannot be satisfied simultaneously.


            next_obs, reward,constraint,center_pos,next_center_pos, done, info = env.step(action)  # Step

            if 'cost_exception' in info:
                prYellow('Cost exception occured.')
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward
            episode_cost += info.get('num_safety_violation', 0)
            episode_safety_cost += info.get('safety_cost', 0)

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == env.max_episode_steps else float(not done)

            memory.push(obs, action, reward, constraint, center_pos, next_center_pos, next_obs, mask,
                        t=episode_steps * env.dt,
                        next_t=(episode_steps + 1) * env.dt)


            # Update state and store transition for GP model learning
            next_state = dynamics_model.get_state(next_obs)
            if episode_steps % 2 == 0 and i_episode < args.gp_max_episodes:  # Stop learning the dynamics after a while to stabilize learning
                # TODO: Clean up line below, specifically (t_batch)
                dynamics_model.append_transition(state, action, next_state, t_batch=np.array([episode_steps*env.dt]))

            positions_record.append(next_center_pos)

            if episode_steps >= 50:
                focus_position_record = positions_record[-40:]
                position_difference = focus_position_record[39] - focus_position_record[0]
                x_difference = position_difference[0]
                y_difference = position_difference[1]
                x_difference_square = x_difference * x_difference
                y_difference_square = y_difference * y_difference
                total_difference_square = x_difference_square + y_difference_square

                numpy_focus_position_record = np.array(focus_position_record)
                x_positions_list = numpy_focus_position_record[:, 0]
                y_positions_list = numpy_focus_position_record[:, 1]
                x_positions_list_var = np.var(x_positions_list)
                y_positions_list_var = np.var(y_positions_list)
                position_total_variance = x_positions_list_var + y_positions_list_var



            obs = next_obs
            cur_cen_pos = next_center_pos

            # env.render(mode='human')    # Choose whether to render.

            if done == True:
                distance = np.linalg.norm((next_center_pos - [2.5, 2.5]))
                prYellow(
                    'Episode {} - step {} - eps_rew {} - eps_cost {} - eps_safety_cost {} - current position {} - current distance {} '.format(
                        i_episode, episode_steps, episode_reward, episode_cost,episode_safety_cost, next_center_pos, distance))

        final_center_pos = np.array([obs[0], obs[1]])
        final_center_pos_x = obs[0]
        final_center_pos_y = obs[1]
        final_distance = np.linalg.norm((final_center_pos - [2.5, 2.5]))

        final_x_difference_square = x_difference_square
        final_y_difference_square = y_difference_square
        final_total_difference_square = total_difference_square

        final_x_positions_list_var = x_positions_list_var
        final_y_positions_list_var = y_positions_list_var
        final_position_total_variance = position_total_variance


        # [optional] save intermediate model
        if (i_episode % int(args.max_episodes / 6) == 0) or (i_episode == args.max_episodes - 1):
            agent.save_model(args.output)
            dynamics_model.save_disturbance_models(args.output)

        writer.log({
            'Episode Reward': episode_reward,
            'Episode Length': episode_steps,
            'Episode Safety Cost': episode_safety_cost,
            'Episode Number of Safety Violations': episode_cost,
            'Cumulated Number of steps': total_numsteps,

        }
        )


        logger.store(Episode=i_episode)
        logger.store(episode_steps=episode_steps)
        logger.store(reward_train=episode_reward)
        logger.store(cost_train=episode_cost)
        logger.store(safety_cost_train=episode_safety_cost)
        logger.store(final_x_difference_square=final_x_difference_square)
        logger.store(final_y_difference_square=final_y_difference_square)
        logger.store(final_total_difference_square=final_total_difference_square)
        logger.store(final_x_positions_list_var=final_x_positions_list_var)
        logger.store(final_y_positions_list_var=final_y_positions_list_var)
        logger.store(final_position_total_variance=final_position_total_variance)
        logger.store(final_center_pos_x=final_center_pos_x)
        logger.store(final_center_pos_y=final_center_pos_y)
        logger.store(final_distance=final_distance)

        prGreen("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}, cost: {}, safety_cost: {}".format(i_episode, total_numsteps,
                                                                                      episode_steps,
                                                                                             round(episode_reward, 2), round(episode_cost, 2),round(episode_safety_cost, 2)))





        logger.log_tabular('Episode',average_only=True)
        logger.log_tabular('episode_steps',average_only=True)
        logger.log_tabular('reward_train',average_only=True)
        logger.log_tabular('cost_train',average_only=True)
        logger.log_tabular('safety_cost_train',average_only=True)

        logger.log_tabular('final_x_difference_square', average_only=True)
        logger.log_tabular('final_y_difference_square', average_only=True)
        logger.log_tabular('final_total_difference_square', average_only=True)
        logger.log_tabular('final_x_positions_list_var', average_only=True)
        logger.log_tabular('final_y_positions_list_var', average_only=True)
        logger.log_tabular('final_position_total_variance', average_only=True)

        logger.log_tabular('final_center_pos_x', average_only=True)
        logger.log_tabular('final_center_pos_y', average_only=True)
        logger.log_tabular('final_distance', average_only=True)
        logger.log_tabular('critic_1_loss',with_min_and_max=True,average_only=False)
        logger.log_tabular('critic_2_loss',with_min_and_max=True,average_only=False)
        logger.log_tabular('lyapunov_loss',with_min_and_max=True,average_only=False)
        logger.log_tabular('policy_loss',with_min_and_max=True,average_only=False)
        logger.log_tabular('ent_loss',with_min_and_max=True,average_only=False)
        logger.log_tabular('alpha',with_min_and_max=True,average_only=False)
        logger.dump_tabular()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    # Environment Args
    parser.add_argument('--env-name', default="SimulatedCars", help='Options are Unicycle or SimulatedCars.')
    # Comet ML
    parser.add_argument('--log_comet', action='store_true', dest='log_comet', help="Whether to log data")
    parser.add_argument('--comet_key', default='', help='Comet API key')
    parser.add_argument('--comet_workspace', default='', help='Comet workspace')
    # SAC Args
    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--visualize', action='store_true', dest='visualize', help='visualize env -only in available test mode')
    parser.add_argument('--output', default='output', type=str, help='')
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 5 episode (default: True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                        help='Automatically adjust α (default: False)')
    parser.add_argument('--seed', type=int, default=12345, metavar='N',
                        help='random seed (default: 12345)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--max_episodes', type=int, default=400, metavar='N',
                        help='maximum number of episodes (default: 400)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=3000, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=10000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--cuda', action="store_true",
                        help='run on CUDA (default: False)')
    parser.add_argument('--device_num', type=int, default=0, help='Select GPU number for CUDA (default: 0)')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
    parser.add_argument('--validate_episodes', default=5, type=int, help='how many episode to perform during validate experiment')
    parser.add_argument('--validate_steps', default=1000, type=int, help='how many steps to perform a validate experiment')
    # CBF, Dynamics, Env Args
    parser.add_argument('--no_diff_qp', action='store_false', dest='diff_qp', help='Should the agent diff through the CBF?')
    parser.add_argument('--gp_model_size', default=3000, type=int, help='gp')
    parser.add_argument('--gp_max_episodes', default=100, type=int, help='gp max train episodes.')
    parser.add_argument('--k_d', default=3.0, type=float)
    parser.add_argument('--gamma_b', default=20, type=float)
    parser.add_argument('--l_p', default=0.03, type=float,
                        help="Look-ahead distance for unicycle dynamics output.")
    # Model Based Learning
    parser.add_argument('--model_based', action='store_true', dest='model_based', help='If selected, will use data from the model to train the RL agent.')
    parser.add_argument('--real_ratio', default=0.3, type=float, help='Portion of data obtained from real replay buffer for training.')
    parser.add_argument('--k_horizon', default=1, type=int, help='horizon of model-based rollouts')
    parser.add_argument('--rollout_batch_size', default=5, type=int, help='Size of initial states batch to rollout from.')
    # Compensator
    parser.add_argument('--comp_rate', default=0.005, type=float, help='Compensator learning rate')
    parser.add_argument('--comp_train_episodes', default=200, type=int, help='Number of initial episodes to train compensator for.')
    parser.add_argument('--comp_update_episode', default=50, type=int, help='Modulo for compensator updates')
    parser.add_argument('--use_comp', type=bool, default=False, help='Should the compensator be used.')
    args = parser.parse_args()



    if args.mode == 'train':
        args.output = get_output_folder(args.output, args.env_name)
    if args.resume == 'default':
        args.resume = os.getcwd() + '/output/{}-run0'.format(args.env_name)
    elif args.resume.isnumeric():
        args.resume = os.getcwd() + '/output/{}-run{}'.format(args.env_name, args.resume)

    if args.cuda:
        torch.cuda.set_device(args.device_num)

    if args.mode == 'train' and args.log_comet:
        project_name = 'sac-rcbf-unicycle-environment' if args.env_name == 'Unicycle' else 'sac-rcbf-sim-cars-env'
        prYellow('Logging experiment on comet.ml!')
        # Create an experiment with your api key
    else:
        experiment = None

    if args.use_comp and (args.model_based or args.diff_qp):
        raise Exception('Compensator can only be used with model free RL and regular CBF.')

    # Environment
    env = build_env(args)
    dynamics_model = DynamicsModel(env, args)

    # Random Seed
    if args.seed >= 0:
        env.seed(args.seed)
        random.seed(args.seed)
        env.action_space.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        dynamics_model.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Agent
    agent = RCBF_SAC(env.observation_space.shape[0], env.action_space, env, args)


    # If model based, we warm up in the model too
    if args.model_based:
        args.start_steps /= (1 + args.rollout_batch_size)

    from utils.run_utils import setup_logger_kwargs

    exp_name = 'BAC_Unicycle'

    logger_kwargs = setup_logger_kwargs(exp_name,args.seed,data_dir='./')

    writer = wandb.init(
        project='OnlyCBF+Augmented-Unicycle',  #   Name of the group where data are saved in wandb
        config=args,
        dir='wandb_logs',
        group='Unicycle',
    )

    if args.mode == 'train':
        train(agent, env, dynamics_model, args,logger_kwargs=logger_kwargs)


    env.close()

