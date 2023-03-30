import argparse
import numpy as np
import torch
from rcbf_sac.dynamics import DYNAMICS_MODE
from rcbf_sac.utils import to_tensor, prRed
from time import time
from qpth.qp import QPFunction
import cvxpy as cp


class CBFQPLayer:

    def __init__(self, env, args, gamma_b=100, k_d=1.5, l_p=0.03):
        """Constructor of CBFLayer.

        Parameters
        ----------
        env : gym.env
            Gym environment.
        gamma_b : float, optional
            gamma of control barrier certificate.
        k_d : float, optional
            confidence parameter desired (2.0 corresponds to ~95% for example).
        """

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.env = env
        self.u_min, self.u_max = self.get_control_bounds()
        self.gamma_b = gamma_b

        if self.env.dynamics_mode not in DYNAMICS_MODE:
            raise Exception('Dynamics mode not supported.')

        if self.env.dynamics_mode == 'Unicycle':
            self.num_cbfs = len(env.hazards_locations)
            self.k_d = k_d
            self.l_p = l_p
        elif self.env.dynamics_mode == 'SimulatedCars':
            self.num_cbfs = 2

        self.action_dim = env.action_space.shape[0]
        self.num_ineq_constraints = self.num_cbfs + 2 * self.action_dim
        self.num_u = 0
        self.num_x = 0


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





    def get_safe_action_without_lya(self, para_lam,state_batch, mean_pred_batch, sigma_batch,lya_grads_batch,lyapunov_value,time_batch,obs_batch,cur_pos_vel_info,episode_steps):  # Will not be used in BAC


        expand_dims = len(state_batch.shape) == 1
        if expand_dims:
            state_batch = state_batch.unsqueeze(0)
            mean_pred_batch = mean_pred_batch.unsqueeze(0)
            sigma_batch = sigma_batch.unsqueeze(0)

        if len(lyapunov_value.shape) == 1:
            lya_grads_batch = lya_grads_batch.unsqueeze(0)  #lya_grads_batch [1,num_states], namely[1,2], 1 here is the batch_size
            lyapunov_value = lyapunov_value.unsqueeze(0)


        start_time = time()
        Ps, qs, Gs, hs = self.get_cbf_qp_constraints_without_lya(state_batch, mean_pred_batch, sigma_batch, lya_grads_batch,lyapunov_value,time_batch)
        build_qp_time = time()
        safe_action_batch = self.solve_qp_without_lya(para_lam,Ps, qs, Gs, hs,episode_steps)

        final_action = safe_action_batch
        return final_action

    def get_cbf_qp_constraints_without_lya(self, state_batch, mean_pred_batch, sigma_pred_batch,lya_grads_batch,lyapunov_value,time_batch):


        assert len(state_batch.shape) == 2 and len(time_batch.shape) == 2 and len(mean_pred_batch.shape) == 2 and len(
            sigma_pred_batch.shape) == 2 and len(lya_grads_batch.shape) == 2, print(state_batch.shape,time_batch, mean_pred_batch.shape,
                                                sigma_pred_batch.shape,lya_grads_batch.shape)

        batch_size = state_batch.shape[0]
        gamma_b = self.gamma_b
        gamma_l = 5.0                         #define the coefficient for the CLF

        # Expand dims
        state_batch = torch.unsqueeze(state_batch, -1)
        # action_batch = torch.unsqueeze(action_batch, -1)
        mean_pred_batch = torch.unsqueeze(mean_pred_batch, -1)
        sigma_pred_batch = torch.unsqueeze(sigma_pred_batch, -1)
        lya_grads_batch = torch.unsqueeze(lya_grads_batch, -2)     #[batch_num,1(number of lyapunov constraints),num_states_used_in_lyapunov]

        if self.env.dynamics_mode == 'SimulatedCars':

            n_u = 1
            num_cbfs = self.num_cbfs
            collision_radius = 4.5
            num_constraints = num_cbfs + 2 * n_u
            act_acc_coef = self.env.act_acc_coef

            # Inequality constraints (G[u, eps] <= h)
            G = torch.zeros((batch_size, num_constraints, n_u + num_cbfs)).to(
                self.device)  # the extra variable is for epsilon (to make sure qp is always feasible)
            h = torch.zeros((batch_size, num_constraints)).to(self.device)
            ineq_constraint_counter = 0

            pos = state_batch[:, ::2, 0]
            vels = state_batch[:, 1::2, 0]

            vels_des = 3.0 * torch.ones((state_batch.shape[0], 5)).to(self.device)  # Desired velocities
            vels_des[:, 0] -= 4 * torch.sin(time_batch.squeeze())
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
            trans = torch.zeros((state_batch.shape[0], 10, 10)).to(self.device)
            ele = torch.diag(torch.ones(10)).to(self.device)
            ele[7, 7] = 0.0
            trans[:, ] = ele
            mid_resu = torch.bmm(trans, state_batch)
            resu = mid_resu.squeeze(-1)

            f_x = (self.env.dt * f_x + resu).unsqueeze(-1)

            # g(x)
            g_x = torch.zeros((state_batch.shape[0], state_batch.shape[1], 1)).to(self.device)
            g_x[:, 6, 0] = 1.0
            g_x[:, 7, 0] = act_acc_coef / self.env.dt  # Car 4's acceleration
            g_x = g_x * self.env.dt

            mu_star = mean_pred_batch
            sigma_star = sigma_pred_batch

            P23_star = torch.zeros(batch_size, 1, state_batch.shape[1]).to(self.device)
            P23_star[:, 0, 4] = 1.0
            P23_star[:, 0, 6] = -1.0

            P34_star = torch.zeros(batch_size, 1, state_batch.shape[1]).to(self.device)
            P34_star[:, 0, 6] = 1.0
            P34_star[:, 0, 8] = -1.0

            q = torch.zeros(batch_size, 1, 1).to(self.device)
            q[:, 0, 0] = -collision_radius

            h23_now = (torch.bmm(P23_star, state_batch) + q)
            h34_now = (torch.bmm(P34_star, state_batch) + q)

            # g(x) for Lyapunov derivative
            lya_g_x = torch.zeros((state_batch.shape[0], 4, 1)).to(self.device)
            lya_g_x[:, 2, 0] = 1.0
            lya_g_x[:, 3, 0] = act_acc_coef / self.env.dt
            lya_g_x_star = lya_g_x * self.env.dt



            lya_obj = torch.zeros((batch_size,1,n_u + num_cbfs)).to(self.device)
            lya_obj[:,0,:n_u] = torch.bmm(lya_grads_batch.view(batch_size, 1, -1), lya_g_x_star.view(batch_size, -1, 1))   #(batchsize,num_lya,num_action)

            self.num_x_without_lya = n_u + num_cbfs

            term_1 = - torch.bmm(P23_star,g_x)
            term_2 = - torch.bmm(P34_star,g_x)


            G[:, 0, 0] = term_1.squeeze()
            G[:, 0, 1] = -1
            G[:, 1, 0] = term_2.squeeze()
            G[:, 1, 2] = -1

            total_x = f_x + mu_star
            term_23_total = torch.bmm(P23_star,total_x)
            term_34_total = torch.bmm(P34_star, total_x)

            h[:, 0] = (term_23_total - torch.bmm(torch.abs(P23_star), sigma_star) + q - h23_now + gamma_b * self.env.dt * h23_now).squeeze()
            h[:, 1] = (term_34_total - torch.bmm(torch.abs(P34_star), sigma_star) + q - h34_now + gamma_b * self.env.dt * h34_now).squeeze()


            ineq_constraint_counter += (num_cbfs)

            # Let's also build the cost matrices, vectors to minimize control effort and penalize slack
            P = torch.diag(torch.ones(n_u + num_cbfs)) * 5e6
            P[0][0] = 0.0
            P = P.repeat(batch_size, 1, 1).to(self.device)
            q = lya_obj    #Lyapunov part in the objective function
        else:
            raise Exception('Dynamics mode unknown!')


        for c in range(n_u):


            if self.u_max is not None:
                G[:, ineq_constraint_counter, c] = 1
                h[:, ineq_constraint_counter] = self.u_max[c]
                ineq_constraint_counter += 1


            if self.u_min is not None:
                G[:, ineq_constraint_counter, c] = -1
                h[:, ineq_constraint_counter] = - self.u_min[c]
                ineq_constraint_counter += 1



        return P, q, G, h

    def solve_qp_without_lya(self, para_lam, Ps: torch.Tensor, qs: torch.Tensor, Gs: torch.Tensor, hs: torch.Tensor,episode_steps):


        Ghs = torch.cat((Gs, hs.unsqueeze(2)), -1)
        Ghs_norm = torch.max(torch.abs(Ghs), dim=2, keepdim=True)[0]
        Gs /= Ghs_norm
        hs = hs / Ghs_norm.squeeze(-1)

        Ps = Ps.squeeze(0)
        qs = qs.squeeze(0)
        Gs = Gs.squeeze(0)
        hs = hs.squeeze(0)


        Ps = Ps.detach().cpu().numpy()
        qs = qs.detach().cpu().numpy()
        Gs = Gs.detach().cpu().numpy()
        hs = hs.detach().cpu().numpy()

        x = cp.Variable(self.num_x_without_lya)
        prob = cp.Problem(cp.Minimize((1 / 2)  * cp.quad_form(x, Ps) + para_lam * qs @ x),
                          [Gs @ x <= hs])
        prob.solve(solver=cp.OSQP, eps_abs=1.0e-01, eps_rel=1.0e-01, max_iter=100000, adaptive_rho=0)

        safe_action_batch = x.value[0]



        safe_action_batch = torch.tensor([safe_action_batch]).to(self.device)

        return safe_action_batch
