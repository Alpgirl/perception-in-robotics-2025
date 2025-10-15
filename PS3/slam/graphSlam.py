"""
Gonzalo Ferrer
g.ferrer@skoltech.ru
28-Feb-2021
"""

import numpy as np
import mrob
from scipy.linalg import inv
from slam.slamBase import SlamBase
from tools.task import get_motion_noise_covariance
from tools.jacobian import state_jacobian
from tools.objects import Gaussian

class GraphSLAM(SlamBase):
    def __init__(self, initial_state, alphas, **kwargs):  #state_dim=3, obs_dim=2, landmark_dim=2, action_dim=3, 
        super().__init__(**kwargs)
        # self.state_dim = state_dim
        # self.landmark_dim = landmark_dim
        # self.obs_dim = obs_dim
        # self.action_dim = action_dim
        self.alphas = alphas
        self.state = initial_state.mu

        self.obs_id = {} # correspondence between observation id and node id in graph
        
        self.graph = mrob.FGraph()
        self.cur_id = self.graph.add_node_pose_2d(initial_state.mu)
        self.next_id = self.cur_id
        self.graph.add_factor_1pose_2d(initial_state.mu, self.cur_id, inv(initial_state.Sigma))

    def predict(self, u):
        _, V = state_jacobian(self.state[:, 0], u)
        M = get_motion_noise_covariance(u, self.alphas)
        W_u = inv(V @ M @ V.T)

        new_id = self.graph.add_node_pose_2d(np.zeros(3))
        self.graph.add_factor_2poses_2d_odom(u, self.cur_id, new_id, W_u)

        self.state = self.graph.get_estimated_state()[-1]
        self.next_id = new_id


    def update(self, z):
        for i in range(z.shape[0]):
            z_id = int(z[i, 2])
            if z_id not in self.obs_id.keys():
                initializeLandmark = True
                node_id = self.graph.add_node_landmark_2d(np.zeros(2))
                self.obs_id[z_id] = node_id
            else:
                initializeLandmark = False
                node_id = self.obs_id[z_id]
            self.graph.add_factor_1pose_1landmark_2d(z[i, :2], self.cur_id, node_id, self.W_z, initializeLandmark=initializeLandmark)
        # print(f"obs ids to node id: {self.obs_id},\n states: {self.graph.get_estimated_state()}")
        self.cur_id = self.next_id

    def solve(self):
        self.graph.solve(mrob.GN)
    
    def manual_solution(self):
        A = self.graph.get_adjacency_matrix().toarray()
        W = self.graph.get_W_matrix().toarray()
        info_matrix_comp = A.T @ W @ A
        info_matrix_from_graph = self.graph.get_information_matrix().toarray()

        print(f"The l2-error between computed info matrix and the one in the graph: {np.linalg.norm(info_matrix_from_graph - info_matrix_comp)}")

        # solve
        b = self.graph.get_vector_b()
        r = inv(info_matrix_comp) @ b
        r_lib = inv(info_matrix_from_graph) @ b
        print(f"The l2-error between normal eq solution and the mrob library solution: {np.linalg.norm(r - r_lib)}")

    def chi2(self):
        return self.graph.chi2()

