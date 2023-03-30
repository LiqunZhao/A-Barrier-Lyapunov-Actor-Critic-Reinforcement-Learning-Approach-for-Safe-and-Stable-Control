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

from rcbf_sac.utils import prGreen, get_output_folder, prYellow, prCyan
from rcbf_sac.evaluator import Evaluator
import wandb


def train(agent, env, dynamics_model, args,logger_kwargs=dict()):


    # To log the data
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Memory
    memory = ReplayMemory(args.replay_size, args.seed)
    memory_model = ReplayMemory(args.replay_size, args.seed)   # Used in other algorithm. In this algorithm this "memory_model" will not be used and can be left.

    # Training Loop
    total_numsteps = 0
    updates = 0
    start_using_backup = False
    use_backup = False           # Whether to use the backup controller
    para_lam = 2.0 * 0.02        # The parameter of the Lyapunov part in the objective function of the backup controller


    for i_episode in range(args.max_episodes):
        use_backup = False
        if i_episode >= 0:               #Can use the backup controller from the first episode
            start_using_backup = True
        backup_time = 0                  # To record the time to use the backup controller. When the time exceeds the predefined threshold, RL-based controller will be reused.
        episode_reward = 0
        episode_cost = 0
        episode_safety_cost = 0.0        # There is an additional term called safety cost and will be used to record the safety cost of each epsiode. Used by CPO, PPO-Lag and TRPO-Lag to form constraints
        episode_steps = 0
        episode_reached = 0              # Record how many times the car is within the required range [9.0,10.0]
        episode_clf_cbf_collision = 0    # Record how many times the CLF and CBF cannot be met at the same time and therefore the backup controller is used.

        done = False
        obs = env.reset()
        cur_pos_vel_info = np.array([obs[4],obs[5],obs[6],obs[7]])           # Current positions and velocities of 3rd and 4th cars. Here we use this state as the input of the Lyapunov network to decrease the dimension of the input of the Lyapunov network and therefore make the training easier.
        next_pos_vel_info = np.array([obs[4],obs[5],obs[6],obs[7]])          # Next positions and velocities of 3rd and 4th cars. Same with the current information at the beginning of the one episode. Later will be different.



        while not done:

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



                    if i_episode < 2:     # There should be some data to record when i_episode < 2, otherwise an error will be encountered.
                        critic_1_loss = 0
                        critic_2_loss = 0
                        lyapunov_loss = 0
                        policy_loss = 0
                        ent_loss = 0
                        alpha = 0
                    logger.store(critic_1_loss=critic_1_loss)
                    logger.store(critic_2_loss=critic_2_loss)
                    logger.store(lyapunov_loss=lyapunov_loss)
                    logger.store(policy_loss=policy_loss)
                    logger.store(ent_loss=ent_loss)
                    logger.store(alpha=alpha)

                    updates += 1


            if (use_backup and start_using_backup):
                action = agent.select_action_backup(para_lam,obs, dynamics_model,cur_pos_vel_info,(episode_steps) * env.dt,episode_steps)   # Use the backup controller to have the control signal
                action = np.array([action])
                backup_time = backup_time + 1
                prCyan('Backup is being used!')
            else:
                action = agent.select_action(obs, dynamics_model, cur_pos_vel_info,warmup=args.start_steps > total_numsteps)  # Use the RL-based controller to have the control signal


            next_obs, reward,constraint,cur_pos_vel_info,next_pos_vel_info, done, info = env.step(action)                     # Step



            if 'cost_exception' in info:
                prYellow('Cost exception occured.')
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward
            episode_cost += info.get('num_safety_violation', 0)
            episode_safety_cost += info.get('safety_cost', 0)



            episode_reached += info.get('reached', 0)
            if ((info.get('num_safety_violation', 0) != 0) and (info.get('reached', 0) != 0)):
                episode_clf_cbf_collision = episode_clf_cbf_collision + 1
            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == env.max_episode_steps else float(not done)

            if not (start_using_backup and use_backup):                     # Only store the transition pair when RL-based controller is used.
                memory.push(obs, action, reward, constraint, cur_pos_vel_info, next_pos_vel_info, next_obs, mask,
                            t=(episode_steps - 1) * env.dt,
                            next_t=(episode_steps) * env.dt)  # Append transition to memory


            # Update state and store transition for GP model learning
            next_state = dynamics_model.get_state(next_obs)
            if episode_steps % 2 == 0 and i_episode < args.gp_max_episodes:  # Stop learning the dynamics after a while to stabilize learning
                # TODO: Clean up line below, specifically (t_batch)
                dynamics_model.append_transition(state, action, next_state, t_batch=np.array([(episode_steps-1)*env.dt]))

            if (start_using_backup and (not use_backup)):
                if ((info.get('num_safety_violation', 0) != 0) and (info.get('reached', 0) != 0)):   # When safety constraint is violated while CLF constraint is met successfully, start to use backup controller to achieve and maintain safety.
                    prYellow('Backup needed!')
                    use_backup = True

            if (use_backup and start_using_backup):     # When the time to use the backup controller is longer than the threshold, or the car is far enough from the position where backup controller is started (namely safe enough now), the backup controller will be stopped.
                if backup_time >= 15:
                    use_backup = False
                    backup_time = 0
                if backup_time >= 5 and ((next_obs[4] * 100.0 - next_obs[6] * 100.0) > 4.5) and ((next_obs[6] * 100.0 - next_obs[8] * 100.0) > 4.5):
                    use_backup = False
                    backup_time = 0


            obs = next_obs
            cur_pos_vel_info = next_pos_vel_info

            # env.render(mode='human')       # Choose whether to render.

            if done == True:
                prYellow(
                    'Episode {} - step {} - eps_rew {} - eps_cost {} - eps_safety_cost {}'.format(
                        i_episode, episode_steps, episode_reward, episode_cost,episode_safety_cost))

        final_third_pos = obs[4]       # Calculated and then logged
        final_fourth_pos = obs[6]
        final_distance = np.linalg.norm((final_third_pos - final_fourth_pos))



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
            'Episode Number of reaching destination': episode_reached,

        }
        )

        logger.store(Episode=i_episode)
        logger.store(episode_steps=episode_steps)
        logger.store(reward_train=episode_reward)
        logger.store(cost_train=episode_cost)
        logger.store(safety_cost_train=episode_safety_cost)
        logger.store(reached_train=episode_reached)
        logger.store(CLF_CBF_violation_train=episode_clf_cbf_collision)
        logger.store(final_third_pos=final_third_pos)
        logger.store(final_fourth_pos=final_fourth_pos)
        logger.store(final_distance=final_distance)

        prGreen("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}, cost: {}, safety_cost: {}".format(i_episode, total_numsteps,
                                                                                      episode_steps,
                                                                                             round(episode_reward, 2), round(episode_cost, 2),round(episode_safety_cost, 2)))




        # Log the data
        logger.log_tabular('Episode',average_only=True)
        logger.log_tabular('episode_steps',average_only=True)
        logger.log_tabular('reward_train',average_only=True)
        logger.log_tabular('cost_train',average_only=True)
        logger.log_tabular('safety_cost_train', average_only=True)
        logger.log_tabular('reached_train', average_only=True)
        logger.log_tabular('CLF_CBF_violation_train', average_only=True)

        logger.log_tabular('final_third_pos', average_only=True)
        logger.log_tabular('final_fourth_pos', average_only=True)
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

    exp_name = 'BLAC_CarsFollowing'

    logger_kwargs = setup_logger_kwargs(exp_name,args.seed,data_dir='./')

    # Need to have a wandb account to save the data
    writer = wandb.init(
        project='CBF+CLF+Augmented-SimulatedCars',  #   Name of the group where data are saved in wandb
        config=args,
        dir='wandb_logs',
        group='SimulatedCars',
    )

    if args.mode == 'train':
        train(agent, env, dynamics_model, args,logger_kwargs=logger_kwargs)


    env.close()

