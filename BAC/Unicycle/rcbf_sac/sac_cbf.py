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
        self.center_pos_num = 2   #the number of states as the input of the Lyapunov network as CLF

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        self.action_space = action_space
        self.action_space.seed(args.seed)
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.critic_lyapunov_lr = args.lr   # learning rate for training value network and Lyapunov network

        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.critic_lyapunov_lr)

        self.lyapunovNet = LyaNetwork(self.center_pos_num,args.hidden_size).to(device=self.device)               #the real state used for clf is 2
        self.lyaNet_optim = Adam(self.lyapunovNet.parameters(),lr = self.critic_lyapunov_lr)


        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)
        self.lyapunovNet_target = LyaNetwork(self.center_pos_num,args.hidden_size).to(device=self.device)
        hard_update(self.lyapunovNet_target, self.lyapunovNet)


        self.lambdas_lr = 0.001      # Learning rate for Lagrangian multipliers
        self.cost_limit = 0.0
        self.augmented_term = 1.0    # Coefficient for the quadratic terms
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
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
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

        if self.env.dynamics_mode == 'Unicycle':
            self.num_cbfs = len(env.hazards_locations)
            self.k_d = k_d
            self.l_p = l_p

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

    def select_action(self, state, dynamics_model,cur_cen_pos, evaluate=False, warmup=False):

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

    def select_action_backup(self, para_lam,state, dynamics_model,cur_cen_pos,episode_steps):          # Will not be used in BAC
        state = to_tensor(state, torch.FloatTensor, self.device)
        cur_cen_pos = to_tensor(cur_cen_pos, torch.FloatTensor, self.device)
        expand_dim = len(state.shape) == 1
        if expand_dim:
            state = state.unsqueeze(0)

        safe_action = self.get_safe_action_backup(para_lam,state, dynamics_model, cur_cen_pos,episode_steps)
        safe_action = safe_action.detach().cpu().numpy()[0] if expand_dim else safe_action.detach().cpu().numpy()
        return safe_action

    def get_safe_action_backup(self, para_lam,obs_batch, dynamics_model,cur_cen_pos,episode_steps):
        state_batch = dynamics_model.get_state(obs_batch)
        mean_pred_batch, sigma_pred_batch = dynamics_model.predict_disturbance(state_batch)
        cur_cen_pos = cur_cen_pos.requires_grad_()
        lyapunov_value = self.lyapunovNet(cur_cen_pos)  # get the Lyapunov network value for the current state
        grads = torch.autograd.grad(outputs=lyapunov_value, inputs=cur_cen_pos, grad_outputs=torch.ones_like(lyapunov_value))[0]   # Calculate the gradient
        cur_lya_grads_batch = grads
        lya_grads_batch = torch.clamp(cur_lya_grads_batch,-1.0,1.0)


        safe_action_batch = self.cbf_layer.get_safe_action_without_lya(para_lam,state_batch, mean_pred_batch,
                                                                       sigma_pred_batch, lya_grads_batch,
                                                                       lyapunov_value,obs_batch,cur_cen_pos,episode_steps)
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
                state_batch, action_batch, reward_batch,constraint_batch,center_pos_batch,next_center_pos_batch, next_state_batch, mask_batch, t_batch, next_t_batch = memory.sample(batch_size=int(real_ratio*batch_size))
                state_batch_m, action_batch_m, reward_batch_m,constraint_batch_m,center_pos_batch_m, next_center_pos_batch_m,next_state_batch_m, mask_batch_m, t_batch_m, next_t_batch_m = memory_model.sample(
                    batch_size=int((1-real_ratio) * batch_size))
                state_batch = np.vstack((state_batch, state_batch_m))
                action_batch = np.vstack((action_batch, action_batch_m))
                reward_batch = np.hstack((reward_batch, reward_batch_m))
                constraint_batch = np.hstack((constraint_batch, constraint_batch_m))
                center_pos_batch = np.hstack((center_pos_batch, center_pos_batch_m))
                next_center_pos_batch = np.hstack((next_center_pos_batch, next_center_pos_batch_m))
                next_state_batch = np.vstack((next_state_batch, next_state_batch_m))
                mask_batch = np.hstack((mask_batch, mask_batch_m))
            else:
                state_batch, action_batch,reward_batch,constraint_batch,center_pos_batch,next_center_pos_batch, next_state_batch, mask_batch, t_batch, next_t_batch = memory.sample(batch_size=batch_size)   # Sample the data

            # print('The sampled state is:')
            # print(state_batch)
            # print('The sampled action is:')
            # print(action_batch)
            # assert 1==2

            state_batch = torch.FloatTensor(state_batch).to(self.device)
            next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
            action_batch = torch.FloatTensor(action_batch).to(self.device)
            reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
            constraint_batch = torch.FloatTensor(constraint_batch).to(self.device).unsqueeze(1)
            center_pos_batch = torch.FloatTensor(center_pos_batch).to(self.device)
            next_center_pos_batch = torch.FloatTensor(next_center_pos_batch).to(self.device)
            mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

            with torch.no_grad():
                next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
                qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
                next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

                lf_next_target = self.lyapunovNet_target(next_center_pos_batch)
                next_l_value = constraint_batch + mask_batch * self.gamma * (lf_next_target)

            qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
            qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf_loss = qf1_loss + qf2_loss

            lf = self.lyapunovNet(center_pos_batch)  # The Lyapunov network
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



            safety_matrix_original,policy_loss_2,sum_penalities = self.get_safety_matrix(state_batch, pi, dynamics_model, center_pos_batch,update_multipliers)  # Get other terms in the augmented Lagrangian function

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



    def get_safety_matrix(self, obs_batch, action_batch, dynamics_model,center_pos_batch,update_multipliers):

        state_batch = dynamics_model.get_state(obs_batch)


        mean_pred_batch, sigma_pred_batch = dynamics_model.predict_disturbance(state_batch)
        center_pos_batch = center_pos_batch.requires_grad_()
        lyapunov_value = self.lyapunovNet(center_pos_batch)  #cur_cen_pos is tensor
        lyapunov_value_detach = lyapunov_value.detach()
        grads = torch.autograd.grad(outputs=lyapunov_value, inputs=center_pos_batch, grad_outputs=torch.ones_like(lyapunov_value))[0]
        lya_grads_batch = grads.detach()

        required_matrix,policy_loss_2,sum_penalities = self.get_cbf_qp_constraints_matrix(state_batch, action_batch, mean_pred_batch, sigma_pred_batch,lya_grads_batch,lyapunov_value_detach,update_multipliers)

        return required_matrix,policy_loss_2,sum_penalities



    def get_cbf_qp_constraints_matrix(self, state_batch, action_batch, mean_pred_batch, sigma_pred_batch,lya_grads_batch,lyapunov_value,update_multipliers):


        if update_multipliers == False:  # According to the setting update_multipliers will be true, therefore only the later part is used.
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

            if self.env.dynamics_mode == 'Unicycle':

                num_cbfs = self.num_cbfs
                num_clfs = 0
                hazards_radius = self.env.hazards_radius
                hazards_locations = to_tensor(self.env.hazards_locations, torch.FloatTensor, self.device)
                collision_radius = 1.2 * hazards_radius  # add a little buffer
                l_p = self.l_p

                thetas = state_batch[:, 2, :].squeeze(-1)
                c_thetas = torch.cos(thetas)
                s_thetas = torch.sin(thetas)

                # p(x): lookahead output (batch_size, 2)
                ps = torch.zeros((batch_size, 2)).to(self.device)
                ps[:, 0] = state_batch[:, 0, :].squeeze(-1) + l_p * c_thetas
                ps[:, 1] = state_batch[:, 1, :].squeeze(-1) + l_p * s_thetas

                # f_p(x) = [0,...,0]^T
                f_ps = torch.zeros((batch_size, 2, 1)).to(self.device)

                # g_p(x) = RL where L = diag([1, l_p])
                Rs = torch.zeros((batch_size, 2, 2)).to(self.device)
                Rs[:, 0, 0] = c_thetas
                Rs[:, 0, 1] = -s_thetas
                Rs[:, 1, 0] = s_thetas
                Rs[:, 1, 1] = c_thetas
                Ls = torch.zeros((batch_size, 2, 2)).to(self.device)
                Ls[:, 0, 0] = 1
                Ls[:, 1, 1] = l_p
                g_ps = torch.bmm(Rs, Ls)  # (batch_size, 2, 2)

                # D_p(x) = g_p [0 D_θ]^T + [D_x1 D_x2]^T
                mu_theta_aug = torch.zeros([batch_size, 2, 1]).to(self.device)
                mu_theta_aug[:, 1, :] = mean_pred_batch[:, 2, :]
                mu_ps = torch.bmm(g_ps, mu_theta_aug) + mean_pred_batch[:, :2, :]
                sigma_theta_aug = torch.zeros([batch_size, 2, 1]).to(self.device)
                sigma_theta_aug[:, 1, :] = sigma_pred_batch[:, 2, :]
                sigma_ps = torch.bmm(torch.abs(g_ps), sigma_theta_aug) + sigma_pred_batch[:, :2, :]

                # hs (batch_size, hazards_locations)
                ps_hzds = ps.repeat((1, num_cbfs)).reshape((batch_size, num_cbfs, 2))

                hs = 0.5 * (torch.sum((ps_hzds - hazards_locations.view(1, num_cbfs, -1)) ** 2, axis=2) - collision_radius ** 2)  # 1/2 * (||x - x_obs||^2 - r^2)

                dhdps = (ps_hzds - hazards_locations.view(1, num_cbfs, -1))  # (batch_size, n_cbfs, 2)
                                                                              # (batch_size, 5, 1)
                n_u = action_batch.shape[1]  # dimension of control inputs
                self.num_u = n_u
                num_constraints = num_cbfs + num_clfs  # each cbf is a constraint, and we need to add actuator constraints (n_u of them)

                # Inequality constraints (G[u, eps] <= h)
                G = torch.zeros((batch_size, num_constraints, n_u)).to(self.device)  # the extra variable is for epsilon (to make sure qp is always feasible)
                h = torch.zeros((batch_size, num_constraints)).to(self.device)
                ineq_constraint_counter = 0

                self.num_x = n_u

                # Add inequality constraints
                G[:, :num_cbfs, :n_u] = -torch.bmm(dhdps, g_ps)  # h1^Tg(x)

                G[:, num_cbfs, :n_u] = torch.bmm(lya_grads_batch, g_ps).squeeze(-2)


                h[:, :num_cbfs] = gamma_b * (hs ** 1) + (torch.bmm(dhdps, f_ps + mu_ps) - torch.bmm(torch.abs(dhdps), sigma_ps)).squeeze(-1)

                h[:, num_cbfs] = (-gamma_l * (lyapunov_value ** 1) - (torch.bmm(lya_grads_batch, f_ps + mu_ps) - torch.bmm(torch.abs(lya_grads_batch), sigma_ps)).squeeze(-1)).squeeze(-1)

                ineq_constraint_counter += (num_cbfs+num_clfs)


            else:
                raise Exception('Dynamics mode unknown!')




            hh = h.unsqueeze(-1)
            matr = torch.bmm(G, action_batch) - hh



            filter = torch.zeros_like(matr)
            filtered_matr = torch.where(matr > 0, matr, filter)
            #
            count_positive = torch.count_nonzero(filtered_matr,dim = 0)
            required_matrix = torch.sum(filtered_matr, dim=0, keepdim=False)
            for i in range(required_matrix.shape[0]):
                if count_positive[i] > 0.5:
                    required_matrix[i] = required_matrix[i] / count_positive[i]

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

        if update_multipliers == True:  # The later part is used to construct part of the augmented Lagrangian function

            assert len(state_batch.shape) == 2 and len(action_batch.shape) == 2 and len(mean_pred_batch.shape) == 2 and len(
                sigma_pred_batch.shape) == 2 and len(lya_grads_batch.shape) == 2, print(state_batch.shape,
                                                                                        action_batch.shape,
                                                                                        mean_pred_batch.shape,
                                                                                        sigma_pred_batch.shape,
                                                                                        lya_grads_batch.shape)

            batch_size = state_batch.shape[0]
            gamma_b = self.gamma_b
            gamma_l = 1.0  # define the coefficient for the CLF

            # Expand dims
            state_batch = torch.unsqueeze(state_batch, -1)
            action_batch = torch.unsqueeze(action_batch, -1)
            mean_pred_batch = torch.unsqueeze(mean_pred_batch, -1)
            sigma_pred_batch = torch.unsqueeze(sigma_pred_batch, -1)
            lya_grads_batch = torch.unsqueeze(lya_grads_batch,
                                              -2)  # [batch_num,1(number of lyapunov constraints),num_states_used_in_lyapunov]

            if self.env.dynamics_mode == 'Unicycle':

                num_cbfs = self.num_cbfs
                num_clfs = 0
                hazards_radius = self.env.hazards_radius
                hazards_locations = to_tensor(self.env.hazards_locations, torch.FloatTensor, self.device)
                collision_radius = 1.05 * hazards_radius  # add a little buffer
                l_p = self.l_p

                thetas = state_batch[:, 2, :].squeeze(-1)
                c_thetas = torch.cos(thetas)
                s_thetas = torch.sin(thetas)

                # p(x): lookahead output (batch_size, 2)
                ps = torch.zeros((batch_size, 2)).to(self.device)
                ps[:, 0] = state_batch[:, 0, :].squeeze(-1) + l_p * c_thetas
                ps[:, 1] = state_batch[:, 1, :].squeeze(-1) + l_p * s_thetas

                # Next we need to predict the state x_(t+1) for constructing the constraints
                matrix_three_to_two = torch.zeros((batch_size, 2, 3)).to(self.device)
                matrix_three_to_two[:,0,0] = 1.0
                matrix_three_to_two[:,1,1] = 1.0

                f_star= torch.zeros((batch_size, 3, 1)).to(self.device)
                f_star = f_star + state_batch


                g_star = torch.zeros((batch_size, 3, 2)).to(self.device)
                g_star[:, 0, 0] = c_thetas
                g_star[:, 1, 0] = s_thetas
                g_star[:, 2, 1] = 1.0
                g_star = g_star * self.env.dt


                mu_star = mean_pred_batch


                matrix_three_to_one = torch.zeros((batch_size, 1, 3)).to(self.device)
                matrix_three_to_one[:,0,2] = 1.0

                tem_result = f_star + torch.bmm(g_star,action_batch) + mu_star

                tem_result1 = torch.bmm(matrix_three_to_two,tem_result)
                term_result2 = torch.zeros((batch_size, 2, 1)).to(self.device)
                new_theta = torch.bmm(matrix_three_to_one,tem_result).squeeze()
                cos_new_theta = torch.cos(new_theta)
                sin_new_theta = torch.sin(new_theta)
                term_result2[:,0,0] = cos_new_theta
                term_result2[:,1,0] = sin_new_theta
                term_result2 = self.l_p * term_result2

                ps_next = (tem_result1 + term_result2).squeeze(-1)


                # hs (batch_size, hazards_locations)
                ps_hzds = ps.repeat((1, num_cbfs)).reshape((batch_size, num_cbfs, 2))

                # CBFs h(x_t)
                hs = 0.5 * (torch.sum((ps_hzds - hazards_locations.view(1, num_cbfs, -1)) ** 2,
                                      axis=2) - collision_radius ** 2)  # 1/2 * (||x - x_obs||^2 - r^2)
                # CBFs h(x_{t+1})
                ps_next_hzds = ps_next.repeat((1, num_cbfs)).reshape((batch_size, num_cbfs, 2))
                hs_next = 0.5 * (torch.sum((ps_next_hzds - hazards_locations.view(1, num_cbfs, -1)) ** 2,
                                      axis=2) - collision_radius ** 2)

                cbf_term = -((hs_next - hs) / self.env.dt) - gamma_b * hs

                # (batch_size, 5, 1)
                n_u = action_batch.shape[1]  # dimension of control inputs
                self.num_u = n_u

                self.num_x = n_u

            else:
                raise Exception('Dynamics mode unknown!')



            matr = cbf_term          # Obtain the matrix containing all the CBF information

            matr = matr.unsqueeze(-1)

            filter = torch.zeros_like(matr)
            filtered_matr = torch.where(matr > 0, matr, filter)   # Only use the information where CBF is violated. If the result is nonpositive it means CBF is meet here and therefore this information will be abandoned. Similar to Relu function.

            count_positive = torch.count_nonzero(filtered_matr,dim = 0)
            required_matrix = torch.sum(filtered_matr, dim=0, keepdim=False)
            for i in range(required_matrix.shape[0]):
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
                real_lambda_clipped = torch.clamp(real_lambda, 0.3, 400.0)   # Confine the multiplier to be within a predefined range
                self.lambdas.append(real_lambda_clipped)
            self.augmented_term = self.augmented_term * self.augmented_ratio   # Enlarge the coefficient for quadratic term. Also confine it to be within a range.
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
