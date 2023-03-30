import argparse
import numpy as np
import torch
from rcbf_sac.dynamics import DYNAMICS_MODE
from rcbf_sac.utils import to_tensor, prRed, prCyan
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



    def get_safe_action_without_lya(self, para_lam,state_batch, mean_pred_batch, sigma_batch,lya_grads_batch,lyapunov_value,obs_batch,cur_cen_pos,episode_steps):

        action_batch = self.u_max                  # Use the maximum allowable action as the nominal control signal
        if len(action_batch.shape) == 1:
            action_batch = action_batch.unsqueeze(0)
        expand_dims = len(state_batch.shape) == 1
        if expand_dims:
            state_batch = state_batch.unsqueeze(0)
            mean_pred_batch = mean_pred_batch.unsqueeze(0)
            sigma_batch = sigma_batch.unsqueeze(0)

        if len(lyapunov_value.shape) == 1:
            lya_grads_batch = lya_grads_batch.unsqueeze(0)  #lya_grads_batch [1,num_states], namely[1,2], 1 here is the batch_size
            lyapunov_value = lyapunov_value.unsqueeze(0)
        start_time = time()

        Ps, qs, Gs, hs = self.get_cbf_qp_constraints_without_lya(state_batch, action_batch, mean_pred_batch, sigma_batch, lya_grads_batch,lyapunov_value)

        build_qp_time = time()
        modi_action_batch = self.solve_qp_without_lya(para_lam,Ps, qs, Gs, hs,episode_steps)
        final_action = action_batch - modi_action_batch
        return final_action if not expand_dims else final_action.squeeze(0)

    def get_cbf_qp_constraints_without_lya(self, state_batch, action_batch, mean_pred_batch, sigma_pred_batch,lya_grads_batch,lyapunov_value):

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
            hazards_radius = self.env.hazards_radius
            hazards_locations = to_tensor(self.env.hazards_locations, torch.FloatTensor, self.device)
            collision_radius = 1.05 * hazards_radius  # add a little buffer
            l_p = self.l_p

            thetas = state_batch[:, 2, :].squeeze(-1)
            c_thetas = torch.cos(thetas)
            s_thetas = torch.sin(thetas)

            # p(x): lookahead output (batch_size, 2)
            # To have the current x_t for constraint calculation
            ps = torch.zeros((batch_size, 2)).to(self.device)
            ps[:, 0] = state_batch[:, 0, :].squeeze(-1) + l_p * c_thetas
            ps[:, 1] = state_batch[:, 1, :].squeeze(-1) + l_p * s_thetas

            matrix_three_to_two = torch.zeros((batch_size, 2, 3)).to(self.device)
            matrix_three_to_two[:, 0, 0] = 1.0
            matrix_three_to_two[:, 1, 1] = 1.0

            g_star = torch.zeros((batch_size, 3, 2)).to(self.device)
            g_star[:, 0, 0] = c_thetas
            g_star[:, 1, 0] = s_thetas
            g_star[:, 2, 1] = 1.0
            g_star = g_star * self.env.dt

            mu_star = mean_pred_batch
            sigma_star = sigma_pred_batch

            matrix_three_to_one = torch.zeros((batch_size, 1, 3)).to(self.device)
            matrix_three_to_one[:, 0, 2] = 1.0

            theta_matrix = torch.zeros((batch_size, 2, 1)).to(self.device)
            theta_matrix[:,0,0] = -s_thetas
            theta_matrix[:,1,0] = c_thetas

            g_penta_med = torch.bmm(torch.bmm(theta_matrix,matrix_three_to_one),g_star)

            g_penta = torch.bmm(matrix_three_to_two,g_star) + self.l_p * g_penta_med

            mu_penta_med = torch.bmm(torch.bmm(theta_matrix,matrix_three_to_one),mu_star)

            mu_penta = torch.bmm(matrix_three_to_two, mu_star) + self.l_p * mu_penta_med

            sigma_penta_med = torch.bmm(torch.bmm(theta_matrix, matrix_three_to_one), sigma_star)

            sigma_penta = torch.bmm(matrix_three_to_two, sigma_star) + self.l_p * sigma_penta_med

            sigma_penta = torch.abs(sigma_penta)

            ps_hzds = ps.repeat((1, num_cbfs)).reshape((batch_size, num_cbfs, 2))

            # CBFs h(x_t)
            hs = 0.5 * (torch.sum((ps_hzds - hazards_locations.view(1, num_cbfs, -1)) ** 2, axis=2) - collision_radius ** 2)  # 1/2 * (||x - x_obs||^2 - r^2)

            dhdps = (ps_hzds - hazards_locations.view(1, num_cbfs, -1))  # (batch_size, n_cbfs, 2)
                                                                          # (batch_size, 5, 1)
            n_u = action_batch.shape[1]  # dimension of control inputs
            self.num_u = n_u
            num_constraints = num_cbfs + 2 * n_u  # each cbf is a constraint, and we need to add actuator constraints (n_u of them)

            # Inequality constraints (G[u, eps] <= h)
            G = torch.zeros((batch_size, num_constraints, n_u + num_cbfs)).to(self.device)  # the extra variable is for epsilon (to make sure qp is always feasible)
            h = torch.zeros((batch_size, num_constraints)).to(self.device)
            ineq_constraint_counter = 0

            lya_obj = torch.zeros((batch_size,1,2 + num_cbfs)).to(self.device)
            lya_grads_batch_dt = (lya_grads_batch) / self.env.dt
            lya_obj[:,0,:n_u] = torch.bmm(lya_grads_batch_dt,g_penta)

            self.num_x_without_lya = n_u + num_cbfs

            # Add inequality constraints
            dhdps_dt = (dhdps) / self.env.dt
            G[:, :num_cbfs, :n_u] = torch.bmm(dhdps_dt, g_penta)
            for ee in range(num_cbfs):
                G[:, ee, ee+n_u] = -1

            h[:, :num_cbfs] = gamma_b * (hs ** 1) + (torch.bmm(dhdps_dt,mu_penta) - torch.bmm(torch.abs(dhdps_dt), sigma_penta) + torch.bmm(torch.bmm(dhdps_dt, g_penta), action_batch)).squeeze(-1)

            ineq_constraint_counter += (num_cbfs)

            # Let's also build the cost matrices, vectors to minimize control effort and penalize slack
            P = torch.diag(torch.ones(n_u + num_cbfs)) * 5e6
            P[0][0] = 1.e0
            P[1][1] = 3.e-4
            P = P.repeat(batch_size, 1, 1).to(self.device)
            q = lya_obj         #Lyapunov part in the objective function
        else:
            raise Exception('Dynamics mode unknown!')

        # Second let's add actuator constraints
        n_u = action_batch.shape[1]  # dimension of control inputs

        for c in range(n_u):

            if self.u_max is not None:
                G[:, ineq_constraint_counter, c] = 1
                h[:, ineq_constraint_counter] = self.u_max[c] - self.u_min[c]
                ineq_constraint_counter += 1

            if self.u_min is not None:
                G[:, ineq_constraint_counter, c] = -1
                h[:, ineq_constraint_counter] = self.u_min[c] - self.u_min[c]
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

        prob = cp.Problem(cp.Minimize((1 / 2)  * cp.quad_form(x, Ps) - para_lam * qs @ x),
                          [Gs @ x <= hs])
        prob.solve(solver=cp.OSQP,eps_abs=1.0e-01,eps_rel = 1.0e-01,max_iter=100000,adaptive_rho=0)


        modi_action_batch = x.value[:self.num_u]

        modi_action_batch = torch.from_numpy(modi_action_batch).to(self.device)

        return modi_action_batch

