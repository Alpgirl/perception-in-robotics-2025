"""
This file implements the Extended Kalman Filter.
"""

import numpy as np

from filters.localization_filter import LocalizationFilter
from tools.task import get_motion_noise_covariance
from tools.task import get_observation as get_expected_observation
from tools.task import get_prediction
from tools.task import wrap_angle


class EKF(LocalizationFilter):
    @staticmethod
    def G(theta, dtrans):
        return np.array([[1, 0, -dtrans * np.sin(theta)],
                         [0, 1, dtrans * np.cos(theta)],
                         [0, 0, 1]])
    
    @staticmethod
    def V(theta, dtrans):
        return np.array([[-dtrans * np.sin(theta), np.cos(theta), 0],
                         [dtrans * np.cos(theta), np.sin(theta), 0],
                         [1, 0, 1]])
    
    @staticmethod
    def H(z, mu, field_map):
        lm_id = int(z[1])
        mx, my = field_map.landmarks_poses_x[lm_id], field_map.landmarks_poses_y[lm_id] 
        mux, muy, _ = mu
        mmux = mx - mux
        mmuy = my - muy
        den = mmux**2 + mmuy**2
        return np.array([[mmuy / den, -mmux / den, -1]])
    
    def predict(self, u):
        # TODO Implement here the EKF, perdiction part. HINT: use the auxiliary functions imported above from tools.task
        drot1, dtrans, drot2 = u
        x, y, theta = self.mu
        self._state.mu = get_prediction(self.mu, u)[:,None]
        G = self.G(theta + drot1, dtrans)
        V = self.V(theta + drot1, dtrans)
        M = get_motion_noise_covariance(u, self._alphas)
        self._state.Sigma = G @ self.Sigma @ G.T + V @ M @ V.T
 
        self._state_bar.mu = self.mu[np.newaxis].T
        self._state_bar.Sigma = self.Sigma

    def update(self, z):
        # TODO implement correction step
        # field_map = FieldMap()
        z_bar = get_expected_observation(self._state_bar.mu[:,0], z[1])
        H = self.H(z, self._state_bar.mu[:,0], self._field_map)
        K = self._state_bar.Sigma @ H.T @ np.linalg.inv(H @ self._state_bar.Sigma @ H.T + self._Q)
        self._state_bar.mu = self._state_bar.mu + K @ np.array([z[0] - z_bar[0]])[:, None]
        self._state_bar.Sigma = self._state_bar.Sigma - K @ H @ self._state_bar.Sigma

        self._state.mu = self._state_bar.mu
        self._state.Sigma = self._state_bar.Sigma

