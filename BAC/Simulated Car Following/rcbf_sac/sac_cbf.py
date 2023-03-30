import random
import torch
import torch.nn.functional as F
from torch.optim import Adam
from rcbf_sac.utils import soft_update, hard_update
from rcbf_sac.model import GaussianPolicy, QNetwork, DeterministicPolicy, LyaNetwork
import numpy as np
from rcbf_sac.diff_cbf_qp import CBFQPLayer
from rcbf_sac.utils import to_tensor
DYNAMICS_MODE = {'Unicycle': {'n_s': 3, 'n_u': 2},   # state = [x y θ]
                 'SimulatedCars': {'n_s': 10, 'n_u': 1}}  # state = [x y θ v ω]
MAX_STD = {'Unicycle': [2e-1, 2e-1, 2e-1], 'SimulatedCars': [0, 0.2, 0, 0.2, 0, 0.2, 0, 0.2, 0, 0.2]}
k_d=1.5
l_p=0.03
# torch.autograd.set_detect_anomaly(True)
class RCBF_SAC(object):

    def __init__(self, num_inputs, action_space, env, args):
        self.gamma = args.gamma
        self.gamma_b = args.gamma_b
        self.tau = args.tau
        self.alpha = args.alpha
        self.previous_positions_num = 4   #the number of states as the input of the Lyapunov network as CLF

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        self.action_space = action_space
        self.action_space.seed(args.seed)

        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.critic_lyapunov_lr = args.lr   # learning rate for training value network and Lyapunov network

        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.critic_lyapunov_lr)

        self.lyapunovNet = LyaNetwork(self.previous_positions_num,args.hidden_size).to(device=self.device)
        self.lyaNet_optim = Adam(self.lyapunovNet.parameters(),lr = self.critic_lyapunov_lr)


        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)
        self.lyapunovNet_target = LyaNetwork(self.previous_positions_num,args.hidden_size).to(device=self.device)
        hard_update(self.lyapunovNet_target, self.lyapunovNet)


        self.lambdas_lr = 0.001      # learning rate for Lagrangian multipliers
        self.cost_limit = 0.0
        self.augmented_term = 1.0    # coefficient for the quadratic terms
        self.augmented_ratio = 1.0005   # Coefficient to enlarge the augmented_term
        self.policy_multiplier_update_ratio = 1

        if args.seed >= 0:
            env.seed(args.seed)
            random.seed(args.seed)
            env.action_space.seed(args.seed)
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        if self.policy_type == "Gaussian":
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)





        # CBF layer
        self.env = env
        self.cbf_layer = CBFQPLayer(env, args, args.gamma_b, args.k_d, args.l_p)
        self.diff_qp = args.diff_qp

        if self.env.dynamics_mode not in DYNAMICS_MODE:
            raise Exception('Dynamics mode not supported.')


        elif self.env.dynamics_mode == 'SimulatedCars':
            self.num_cbfs = 2

        self.action_dim = env.action_space.shape[0]

        self.u_min, self.u_max = self.get_control_bounds()

        num_cbfs = self.num_cbfs



        self.num_constraints = num_cbfs

        self.log_lambdas = []
        for i in range(self.num_constraints):
            log_lambda = torch.zeros(1, requires_grad=True, device=self.device)
            self.log_lambdas.append(log_lambda)
        self.log_lambdas_optims = []
        for i in range(self.num_constraints):
            log_lambda_optim = Adam([self.log_lambdas[i]], lr=self.lambdas_lr)
            self.log_lambdas_optims.append(log_lambda_optim)

    def select_action(self, state, dynamics_model,cur_pos_vel_info, evaluate=False, warmup=False):

        state = to_tensor(state, torch.FloatTensor, self.device)
        expand_dim = len(state.shape) == 1
        if expand_dim:
            state = state.unsqueeze(0)

        if warmup:

            batch_size = state.shape[0]
            action = torch.zeros((batch_size, self.action_space.shape[0])).to(self.device)
            for i in range(batch_size):
                action[i] = torch.from_numpy(self.action_space.sample()).to(self.device)
        else:
            if evaluate is False:
                action, _, _ = self.policy.sample(state)
            else:
                _, _, action = self.policy.sample(state)


        action = action.detach().cpu().numpy()[0] if expand_dim else action.detach().cpu().numpy()
        return action

    def select_action_backup(self, para_lam,state, dynamics_model,cur_pos_vel_info, time_batch,episode_steps):   # Will not be used in BAC
        state = to_tensor(state, torch.FloatTensor, self.device)
        time_batch = torch.tensor([time_batch]).type(torch.FloatTensor).to(self.device)
        cur_pos_vel_info = to_tensor(cur_pos_vel_info, torch.FloatTensor, self.device)
        expand_dim = len(state.shape) == 1
        if expand_dim:
            state = state.unsqueeze(0)
            time_batch = time_batch.unsqueeze(0)

        safe_action = self.get_safe_action_backup(para_lam,state, dynamics_model, cur_pos_vel_info,time_batch,episode_steps)
        safe_action = safe_action.detach().cpu().numpy()[0] if expand_dim else safe_action.detach().cpu().numpy()
        print('final safe action is:')
        print(safe_action)
        return safe_action

    def get_safe_action_backup(self, para_lam,obs_batch, dynamics_model,cur_pos_vel_info,time_batch,episode_steps):
        state_batch = dynamics_model.get_state(obs_batch)
        mean_pred_batch, sigma_pred_batch = dynamics_model.predict_disturbance(state_batch)
        cur_pos_vel_info = cur_pos_vel_info.requires_grad_()
        lyapunov_value = self.lyapunovNet(cur_pos_vel_info)   # get the Lyapunov network value for the current state
        grads = torch.autograd.grad(outputs=lyapunov_value, inputs=cur_pos_vel_info, grad_outputs=torch.ones_like(lyapunov_value))[0]  # Calculate the gradient
        cur_lya_grads_batch = grads
        lya_grads_batch = torch.clamp(cur_lya_grads_batch,-1.0,1.0)

        safe_action_batch = self.cbf_layer.get_safe_action_without_lya(para_lam,state_batch, mean_pred_batch,
                                                                       sigma_pred_batch, lya_grads_batch,
                                                                       lyapunov_value,time_batch,obs_batch,cur_pos_vel_info,episode_steps)
        return safe_action_batch


    def update_parameters(self, memory, batch_size, updates, dynamics_model, memory_model=None, real_ratio=None):


        for i in range(self.policy_multiplier_update_ratio):
            final_round = self.policy_multiplier_update_ratio - 1
            if i == final_round:
                update_multipliers = True
            else:
                update_multipliers = False
            # Model-based vs regular RL
            if memory_model and real_ratio:
                state_batch, action_batch, reward_batch,constraint_batch,previous_positions_batch,next_positions_batch, next_state_batch, mask_batch, t_batch, next_t_batch = memory.sample(batch_size=int(real_ratio*batch_size))
                state_batch_m, action_batch_m, reward_batch_m,constraint_batch_m,previous_positions_batch_m, next_positions_batch_m,next_state_batch_m, mask_batch_m, t_batch_m, next_t_batch_m = memory_model.sample(
                    batch_size=int((1-real_ratio) * batch_size))
                state_batch = np.vstack((state_batch, state_batch_m))
                action_batch = np.vstack((action_batch, action_batch_m))
                reward_batch = np.hstack((reward_batch, reward_batch_m))
                constraint_batch = np.hstack((constraint_batch, constraint_batch_m))
                previous_positions_batch = np.hstack((previous_positions_batch, previous_positions_batch_m))
                next_positions_batch = np.hstack((next_positions_batch, next_positions_batch_m))
                next_state_batch = np.vstack((next_state_batch, next_state_batch_m))
                mask_batch = np.hstack((mask_batch, mask_batch_m))
            else:
                state_batch, action_batch,reward_batch,constraint_batch,previous_positions_batch,next_positions_batch, next_state_batch, mask_batch, t_batch, next_t_batch = memory.sample(batch_size=batch_size)   # Sample the data


            state_batch = torch.FloatTensor(state_batch).to(self.device)
            next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
            action_batch = torch.FloatTensor(action_batch).to(self.device)
            reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
            constraint_batch = torch.FloatTensor(constraint_batch).to(self.device).unsqueeze(1)
            previous_positions_batch = torch.FloatTensor(previous_positions_batch).to(self.device)
            next_positions_batch = torch.FloatTensor(next_positions_batch).to(self.device)
            mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
            time_batch = torch.FloatTensor(t_batch).to(self.device).unsqueeze(1)

            with torch.no_grad():
                next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
                qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
                next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

                lf_next_target = self.lyapunovNet_target(next_positions_batch)
                next_l_value = constraint_batch + mask_batch * self.gamma * (lf_next_target)

            qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
            qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf_loss = qf1_loss + qf2_loss

            lf = self.lyapunovNet(previous_positions_batch)  # The Lyapunov network
            lf_loss = F.mse_loss(lf, next_l_value)  # Loss function for the Lyapunov network


            self.critic_optim.zero_grad()
            qf_loss.backward()
            self.critic_optim.step()

            self.lyaNet_optim.zero_grad()
            lf_loss.backward()
            self.lyaNet_optim.step()

            # Compute Actions and log probabilities
            pi, log_pi, _ = self.policy.sample(state_batch)

            qf1_pi, qf2_pi = self.critic(state_batch, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)

            policy_loss_1 = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]



            safety_matrix_original,policy_loss_2,sum_penalities = self.get_safety_matrix(state_batch, pi, dynamics_model, previous_positions_batch,update_multipliers,time_batch)  # Get other terms in the augmented Lagrangian function

            policy_loss = (policy_loss_1 + policy_loss_2)

            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()


            if self.automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()

                self.alpha = self.log_alpha.exp()
                alpha_tlogs = self.alpha.clone()  # For Comet.ml logs
            else:
                alpha_loss = torch.tensor(0.).to(self.device)
                alpha_tlogs = torch.tensor(self.alpha)  # For Comet.ml logs

            if updates % self.target_update_interval == 0:
                soft_update(self.critic_target, self.critic, self.tau)
                soft_update(self.lyapunovNet_target, self.lyapunovNet, self.tau)



        return qf1_loss.item(), qf2_loss.item(),lf_loss.item(), policy_loss_1.item(), alpha_loss.item(), alpha_tlogs.item()



    # Save model parameters
    def save_model(self, output):
        print('Saving models in {}'.format(output))
        torch.save(
            self.policy.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output)
        )
        torch.save(
            self.lyapunovNet.state_dict(),
            '{}/lyapunov.pkl'.format(output)
        )

    # Load model parameters
    def load_weights(self, output):
        if output is None: return
        print('Loading models from {}'.format(output))

        self.policy.load_state_dict(
            torch.load('{}/actor.pkl'.format(output), map_location=torch.device(self.device))
        )
        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output), map_location=torch.device(self.device))
        )
        self.lyapunovNet.load_state_dict(
            torch.load('{}/lyapunov.pkl'.format(output), map_location=torch.device(self.device))
        )

    def load_model(self, actor_path, critic_path,lyapunov_path):
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
        if lyapunov_path is not None:
            self.lyapunovNet.load_state_dict(torch.load(lyapunov_path))



    def get_safety_matrix(self, obs_batch, action_batch, dynamics_model,previous_positions_batch,update_multipliers,time_batch):


        state_batch = dynamics_model.get_state(obs_batch)


        mean_pred_batch, sigma_pred_batch = dynamics_model.predict_disturbance(state_batch)
        previous_positions_batch = previous_positions_batch.requires_grad_()
        lyapunov_value = self.lyapunovNet(previous_positions_batch)  #cur_cen_pos is tensor
        grads = torch.autograd.grad(outputs=lyapunov_value, inputs=previous_positions_batch, grad_outputs=torch.ones_like(lyapunov_value))[0]
        lya_grads_batch = grads


        required_matrix,policy_loss_2,sum_penalities = self.get_cbf_qp_constraints_matrix(dynamics_model,state_batch, action_batch, mean_pred_batch, sigma_pred_batch,lya_grads_batch,lyapunov_value,update_multipliers,previous_positions_batch,time_batch)

        return required_matrix,policy_loss_2,sum_penalities



    def get_cbf_qp_constraints_matrix(self, dynamics_model,state_batch, action_batch, mean_pred_batch, sigma_pred_batch,lya_grads_batch,lyapunov_value,update_multipliers,previous_positions_batch,t_batch):


        if update_multipliers == False:            # According to the setting update_multipliers will be true, therefore only the later part is used.
            assert len(state_batch.shape) == 2 and len(action_batch.shape) == 2 and len(mean_pred_batch.shape) == 2 and len(
                sigma_pred_batch.shape) == 2 and len(lya_grads_batch.shape) == 2, print(state_batch.shape, action_batch.shape, mean_pred_batch.shape,
                                                    sigma_pred_batch.shape,lya_grads_batch.shape)

            batch_size = state_batch.shape[0]
            gamma_b = self.gamma_b
            gamma_l = 1.0                          #define the coefficient for the CLF

            # Expand dims
            state_batch = torch.unsqueeze(state_batch, -1)
            action_batch = torch.unsqueeze(action_batch, -1)
            mean_pred_batch = torch.unsqueeze(mean_pred_batch, -1)
            sigma_pred_batch = torch.unsqueeze(sigma_pred_batch, -1)
            lya_grads_batch = torch.unsqueeze(lya_grads_batch, -2)     #[batch_num,1(number of lyapunov constraints),num_states_used_in_lyapunov]

            if self.env.dynamics_mode == 'SimulatedCars':

                n_u = action_batch.shape[1]  # dimension of control inputs
                num_cbfs = self.num_cbfs
                num_clfs = 1
                num_constraints = num_cbfs + num_clfs
                collision_radius = 4.5
                act_acc_coef = self.env.act_acc_coef

                # Inequality constraints (G[u, eps] <= h)
                G = torch.zeros((batch_size, num_constraints, n_u)).to(
                    self.device)  # the extra variable is for epsilon (to make sure qp is always feasible)
                h = torch.zeros((batch_size, num_constraints)).to(self.device)
                ineq_constraint_counter = 0

                # Current State
                pos = state_batch[:, ::2, 0]
                vels = state_batch[:, 1::2, 0]

                # Acceleration
                vels_des = 3.0 * torch.ones((state_batch.shape[0], 5)).to(self.device)  # Desired velocities
                vels_des[:, 0] -= 4 * torch.sin( t_batch.squeeze())
                accels = self.env.kp * (vels_des - vels)
                accels[:, 1] -= self.env.k_brake * (pos[:, 0] - pos[:, 1]) * ((pos[:, 0] - pos[:, 1]) < 6.5)
                accels[:, 2] -= self.env.k_brake * (pos[:, 1] - pos[:, 2]) * ((pos[:, 1] - pos[:, 2]) < 6.5)
                accels[:, 3] = 0.0
                accels[:, 4] -= self.env.k_brake * (pos[:, 2] - pos[:, 4]) * ((pos[:, 2] - pos[:, 4]) < 13.0)

                # f(x)
                f_x = torch.zeros((state_batch.shape[0], state_batch.shape[1])).to(self.device)
                f_x[:, ::2] = vels
                f_x[:, 1::2] = accels

                #f(x) for Lyapunov derivative
                lya_f_x = torch.zeros((state_batch.shape[0], 4)).to(self.device)
                lya_f_x[:,0] = vels[:, 2]
                lya_f_x[:,1] = accels[:, 2]
                lya_f_x[:,2] = vels[:, 3]
                lya_f_x[:,3] = accels[:, 3]

                # f_D(x) - disturbance in the drift dynamics
                fD_x = torch.zeros((state_batch.shape[0], state_batch.shape[1])).to(self.device)
                fD_x[:, 1::2] = sigma_pred_batch[:, 1::2, 0].squeeze(-1)

                lya_fD_x = torch.zeros((state_batch.shape[0], 4)).to(self.device)
                lya_fD_x[:, 0] = sigma_pred_batch[:, 4, 0].squeeze(-1)
                lya_fD_x[:, 1] = sigma_pred_batch[:, 5, 0].squeeze(-1)
                lya_fD_x[:, 2] = sigma_pred_batch[:, 6, 0].squeeze(-1)
                lya_fD_x[:, 3] = sigma_pred_batch[:, 7, 0].squeeze(-1)

                lya_fM_x = torch.zeros((state_batch.shape[0], 4)).to(self.device)
                lya_fM_x[:, 0] = mean_pred_batch[:, 4, 0].squeeze(-1)
                lya_fM_x[:, 1] = mean_pred_batch[:, 5, 0].squeeze(-1)
                lya_fM_x[:, 2] = mean_pred_batch[:, 6, 0].squeeze(-1)
                lya_fM_x[:, 3] = mean_pred_batch[:, 7, 0].squeeze(-1)


                # g(x)
                g_x = torch.zeros((state_batch.shape[0], state_batch.shape[1], 1)).to(self.device)
                g_x[:, 7, 0] = act_acc_coef  # Car 4's velocity

                # g(x) for Lyapunov derivative
                lya_g_x = torch.zeros((state_batch.shape[0], 4, 1)).to(self.device)
                lya_g_x[:, 3, 0] = act_acc_coef

                # h1
                h13 = 0.5 * (((pos[:, 2] - pos[:, 3]) ** 2) - collision_radius ** 2)
                h15 = 0.5 * (((pos[:, 4] - pos[:, 3]) ** 2) - collision_radius ** 2)

                # dh1/dt = Lfh1
                h13_dot = (pos[:, 3] - pos[:, 2]) * (vels[:, 3] - vels[:, 2])
                h15_dot = (pos[:, 3] - pos[:, 4]) * (vels[:, 3] - vels[:, 4])

                # Lffh1
                dLfh13dx = torch.zeros((batch_size, 10)).to(self.device)
                dLfh13dx[:, 4] = (vels[:, 2] - vels[:, 3])  # Car 3 pos
                dLfh13dx[:, 5] = (pos[:, 2] - pos[:, 3])  # Car 3 vel
                dLfh13dx[:, 6] = (vels[:, 3] - vels[:, 2])
                dLfh13dx[:, 7] = (pos[:, 3] - pos[:, 2])
                Lffh13 = torch.bmm(dLfh13dx.view(batch_size, 1, -1), f_x.view(batch_size, -1, 1)).squeeze()
                LfDfh13 = torch.bmm(torch.abs(dLfh13dx.view(batch_size, 1, -1)), fD_x.view(batch_size, -1, 1)).squeeze()

                dLfh15dx = torch.zeros((batch_size, 10)).to(self.device)
                dLfh15dx[:, 8] = (vels[:, 4] - vels[:, 3])  # Car 5 pos
                dLfh15dx[:, 9] = (pos[:, 4] - pos[:, 3])  # Car 5 vels
                dLfh15dx[:, 6] = (vels[:, 3] - vels[:, 4])
                dLfh15dx[:, 7] = (pos[:, 3] - pos[:, 4])
                Lffh15 = torch.bmm(dLfh15dx.view(batch_size, 1, -1), f_x.view(batch_size, -1, 1)).squeeze()
                LfDfh15 = torch.bmm(torch.abs(dLfh15dx.view(batch_size, 1, -1)), fD_x.view(batch_size, -1, 1)).squeeze()


                # Lfgh1
                Lgfh13 = torch.bmm(dLfh13dx.view(batch_size, 1, -1), g_x)
                Lgfh15 = torch.bmm(dLfh15dx.view(batch_size, 1, -1), g_x)


                #for Lyapunov, dL(x)/dt = LfL(x) = deltaL/deltax*f(x)
                LfLx = torch.bmm(lya_grads_batch.view(batch_size, 1, -1), lya_f_x.view(batch_size, -1, 1)).squeeze(-1)

                #delta(deltaL/deltax*f(x))/delta(x)
                second_order_grad = torch.autograd.grad(outputs=LfLx,inputs=previous_positions_batch,
                          grad_outputs=torch.ones_like(LfLx),
                          create_graph=False)[0]

                Lya_value_detach = lyapunov_value.detach()
                lya_dot = LfLx.detach()
                lya_Lg_Lf = torch.bmm(second_order_grad.view(batch_size, 1, -1), lya_g_x.view(batch_size, -1, 1))
                Lya_Lff = torch.bmm(second_order_grad.view(batch_size, 1, -1), lya_f_x.view(batch_size, -1, 1)).squeeze(-1)
                Lya_LfD = torch.bmm(torch.abs(second_order_grad.view(batch_size, 1, -1)), lya_fD_x.view(batch_size, -1, 1)).squeeze(-1)
                Lya_LfM = torch.bmm(second_order_grad.view(batch_size, 1, -1),lya_fM_x.view(batch_size, -1, 1)).squeeze(-1)


                # Inequality constraints (G[u, eps] <= h)
                h[:, 0] = Lffh13 - LfDfh13 + (gamma_b + gamma_b) * h13_dot + gamma_b * gamma_b * h13

                h[:, 1] = Lffh15 - LfDfh15 + (gamma_b + gamma_b) * h15_dot + gamma_b * gamma_b * h15

                h[:, 2] = (-Lya_Lff - Lya_LfD - (gamma_l + gamma_l) * lya_dot - gamma_l * gamma_l * Lya_value_detach).squeeze(-1)

                G[:, 0, 0] = -Lgfh13.squeeze()
                G[:, 1, 0] = -Lgfh15.squeeze()
                G[:, 2, 0] = lya_Lg_Lf.squeeze()

                ineq_constraint_counter += (num_cbfs+num_clfs)



            else:
                raise Exception('Dynamics mode unknown!')

            hh = h.unsqueeze(-1)
            matr = torch.bmm(G, action_batch) - hh

            filter = torch.zeros_like(matr)
            filtered_matr = torch.where(matr > 0, matr, filter)

            count_positive = torch.count_nonzero(filtered_matr,dim = 0)
            required_matrix = torch.sum(filtered_matr, dim=0, keepdim=False)
            for i in range(required_matrix.shape[0]):
                if count_positive[i] > 0.5:
                    required_matrix[i] = required_matrix[i] / count_positive[i]

            print('required_matrix')
            print(required_matrix)


            other_compoenent = required_matrix[:-1,:] - self.cost_limit
            other_compoenent_mean = torch.abs(torch.mean(other_compoenent))
            lya_component = torch.abs(required_matrix[-1,:] - self.cost_limit)
            ratio = float(other_compoenent_mean / lya_component)

            required_matrix_copy = required_matrix.detach()

            self.lambdas = []
            for i in range(len(self.log_lambdas)):
                real_lambda = torch.exp(self.log_lambdas[i].detach())
                real_lambda_clipped = torch.clamp(real_lambda,0.3,400.0)
                self.lambdas.append(real_lambda_clipped)
            policy_loss_2 = float(self.lambdas[0]) * (required_matrix[0] - self.cost_limit) + self.augmented_term / 2.0 * (required_matrix[0] - self.cost_limit) * (required_matrix[0] - self.cost_limit)
            for i in range(required_matrix.shape[0] - 2):
                policy_loss_2 += float(self.lambdas[i + 1]) * (required_matrix[i + 1] - self.cost_limit) + self.augmented_term / 2.0 * (required_matrix[i + 1] - self.cost_limit) * (required_matrix[i + 1] - self.cost_limit)
            policy_loss_2 += float(self.lambdas[-1]) * ratio * (required_matrix[-1] - self.cost_limit) + ratio * ratio * self.augmented_term / 2.0 * (required_matrix[-1] - self.cost_limit) * (required_matrix[-1] - self.cost_limit)


            sum_penalities = sum(self.lambdas)

            return required_matrix, policy_loss_2,sum_penalities

        if update_multipliers == True:     # The later part is used to construct part of the augmented Lagrangian function

            assert len(state_batch.shape) == 2 and len(action_batch.shape) == 2 and len(
                mean_pred_batch.shape) == 2 and len(
                sigma_pred_batch.shape) == 2 and len(lya_grads_batch.shape) == 2 and len(t_batch.shape) == 2, print(state_batch.shape,
                                                                                        action_batch.shape,
                                                                                        mean_pred_batch.shape,
                                                                                        sigma_pred_batch.shape,
                                                                                        lya_grads_batch.shape)

            batch_size = state_batch.shape[0]
            gamma_b = self.gamma_b

            # Expand dims
            state_batch = torch.unsqueeze(state_batch, -1)
            action_batch = torch.unsqueeze(action_batch, -1)
            mean_pred_batch = torch.unsqueeze(mean_pred_batch, -1)
            sigma_pred_batch = torch.unsqueeze(sigma_pred_batch, -1)
            lya_grads_batch = torch.unsqueeze(lya_grads_batch,
                                              -2)  # [batch_num,1(number of lyapunov constraints),num_states_used_in_lyapunov]

            if self.env.dynamics_mode == 'SimulatedCars':

                n_u = action_batch.shape[1]  # dimension of control inputs
                num_cbfs = self.num_cbfs
                num_clfs = 1
                collision_radius = 4.5
                act_acc_coef = self.env.act_acc_coef

                ineq_constraint_counter = 0

                # Current State
                pos = state_batch[:, ::2, 0]
                vels = state_batch[:, 1::2, 0]

                # Acceleration
                vels_des = 3.0 * torch.ones((state_batch.shape[0], 5)).to(self.device)  # Desired velocities
                vels_des[:, 0] -= 4 * torch.sin(t_batch.squeeze())
                accels = self.env.kp * (vels_des - vels)
                accels[:, 1] -= self.env.k_brake * (pos[:, 0] - pos[:, 1]) * ((pos[:, 0] - pos[:, 1]) < 6.5)
                accels[:, 2] -= self.env.k_brake * (pos[:, 1] - pos[:, 2]) * ((pos[:, 1] - pos[:, 2]) < 6.5)
                accels[:, 3] = 0.0
                accels[:, 4] -= self.env.k_brake * (pos[:, 2] - pos[:, 4]) * ((pos[:, 2] - pos[:, 4]) < 13.0)

                # f(x)
                f_x = torch.zeros((state_batch.shape[0], state_batch.shape[1])).to(self.device)
                f_x[:, ::2] = vels
                f_x[:, 1::2] = accels
                f_x[:, 6] = 0.0
                f_x[:, 7] = 0.0
                trans = torch.zeros((state_batch.shape[0],10,10)).to(self.device)
                ele = torch.diag(torch.ones(10)).to(self.device)
                ele[7, 7] = 0.0
                trans[:, ] = ele
                mid_resu = torch.bmm(trans,state_batch)
                resu = mid_resu.squeeze(-1)

                f_x = (self.env.dt * f_x + resu).unsqueeze(-1)



                # g(x)
                g_x = torch.zeros((state_batch.shape[0], state_batch.shape[1], 1)).to(self.device)
                g_x[:, 6, 0] = 1.0
                g_x[:, 7, 0] = act_acc_coef / self.env.dt # Car 4's velocity
                g_x = g_x * self.env.dt

                mu_star = mean_pred_batch
                sigma_star = sigma_pred_batch

                next_batch = f_x + torch.bmm(g_x,action_batch) + mu_star


                P23_star = torch.zeros(batch_size,1,state_batch.shape[1]).to(self.device)
                P23_star[:,0,4] = 1.0
                P23_star[:,0,6] = -1.0

                P34_star = torch.zeros(batch_size,1,state_batch.shape[1]).to(self.device)
                P34_star[:,0,6] = 1.0
                P34_star[:,0,8] = -1.0

                q = torch.zeros(batch_size,1,1).to(self.device)
                q[:,0,0] = -collision_radius

                h23_next = (torch.bmm(P23_star,next_batch) + q).squeeze(-1)
                h34_next = (torch.bmm(P34_star,next_batch) + q).squeeze(-1)

                h23_now = (torch.bmm(P23_star,state_batch) + q).squeeze(-1)
                h34_now = (torch.bmm(P34_star,state_batch) + q).squeeze(-1)

                cbf23_term = -((h23_next - h23_now) / self.env.dt) - gamma_b * h23_now

                cbf34_term = -((h34_next - h34_now) / self.env.dt) - gamma_b * h34_now

                ineq_constraint_counter += (num_cbfs + num_clfs)

            else:
                raise Exception('Dynamics mode unknown!')

            matr = torch.cat((cbf23_term,cbf34_term),1)    # Obtain the matrix containing all CBF information

            matr = matr.unsqueeze(-1)

            filter = torch.zeros_like(matr)
            filtered_matr = torch.where(matr > 0, matr, filter)  # Only use the information where CBF is violated. If the result is nonpositive it means CBF is meet here and therefore this information will be abandoned. Similar to Relu function.

            count_positive = torch.count_nonzero(filtered_matr,dim = 0)
            required_matrix = torch.sum(filtered_matr, dim=0, keepdim=False)
            for i in range(required_matrix.shape[0]):          # Calculate the average
                if count_positive[i] > 0.5:
                    required_matrix[i] = required_matrix[i] / count_positive[i]


            required_matrix_copy = required_matrix.detach()

            log_lambdas_losses = []
            for i in range(len(self.log_lambdas_optims)):
                item = required_matrix_copy[i]
                log_lambda_loss = - torch.mean(self.log_lambdas[i] * item)
                log_lambdas_losses.append(log_lambda_loss)
            for i in range(len(self.log_lambdas_optims)):
                self.log_lambdas_optims[i].zero_grad()
                log_lambdas_losses[i].backward()
                self.log_lambdas_optims[i].step()

            self.lambdas = []
            for i in range(len(self.log_lambdas)):
                real_lambda = torch.exp(self.log_lambdas[i].detach())
                real_lambda_clipped = torch.clamp(real_lambda, 0.3, 400.0)    # Confine the multiplier to be within a predefined range
                self.lambdas.append(real_lambda_clipped)
            self.augmented_term = self.augmented_term * self.augmented_ratio  # Enlarge the coefficient for quadratic term. Also confine it to be within a range.
            self.augmented_term = min(self.augmented_term,200)
            policy_loss_2 = float(self.lambdas[0]) * (required_matrix[0] - self.cost_limit) + self.augmented_term / 2.0 * (required_matrix[0] - self.cost_limit) * (required_matrix[0] - self.cost_limit)
            for i in range(required_matrix.shape[0] - 1):
                policy_loss_2 += float(self.lambdas[i + 1]) * (required_matrix[i + 1] - self.cost_limit) + self.augmented_term / 2.0 * (required_matrix[i + 1] - self.cost_limit) * (required_matrix[i + 1] - self.cost_limit)

            sum_penalities = sum(self.lambdas)

            return required_matrix, policy_loss_2, sum_penalities

    def get_control_bounds(self):
        """

        Returns
        -------
        u_min : torch.tensor
            min control input.
        u_max : torch.tensor
            max control input.
        """

        u_min = torch.tensor(self.env.safe_action_space.low).to(self.device)
        u_max = torch.tensor(self.env.safe_action_space.high).to(self.device)

        return u_min, u_max
