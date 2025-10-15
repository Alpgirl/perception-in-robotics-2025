"""
Sudhanva Sreesha
ssreesha@umich.edu
28-Mar-2018

This file implements the Particle Filter.
"""

import numpy as np
from numpy.random import uniform
from scipy.stats import norm as gaussian

from filters.localization_filter import LocalizationFilter
from tools.task import get_gaussian_statistics
from tools.task import get_observation
from tools.task import sample_from_odometry
from tools.task import wrap_angle

class PF(LocalizationFilter):
    def __init__(self, initial_state, alphas, bearing_std, num_particles, global_localization):
        super(PF, self).__init__(initial_state, alphas, bearing_std)
        # TODO add here specific class variables for the PF
        self.n_particles = num_particles
        self.global_loc = global_localization
        self.particles = np.random.multivariate_normal(self.mu, self.Sigma, self.n_particles)
        self.particles_pred = []

    def predict(self, u):
        # TODO Implement here the PF, perdiction part
        states_mu = []
        for i in range(self.n_particles):
            state_mu = sample_from_odometry(self.particles[i], u, self._alphas)
            states_mu.append(state_mu)

        self.particles_pred = np.array(states_mu)
        self._state = get_gaussian_statistics(self.particles_pred)
        self._state_bar.mu = self.mu[np.newaxis].T
        self._state_bar.Sigma = self.Sigma

    def update(self, z):
        # TODO implement correction step
        weights = []
        obs = gaussian(loc=0, scale=np.sqrt(self._Q))
        for i in range(self.n_particles):
            z_bar = get_observation(self.particles_pred[i], z[1])
            diff = z[0] - z_bar[0]
            w = obs.pdf(diff)
            weights.append(w)

        weights = np.array(weights)
        weights /= np.sum(weights)

        cum_func = np.cumsum(weights)

        # low-variance sampling
        i = 0
        new_particles = []
        r = uniform(0, 1. / self.n_particles)
        for m in range(self.n_particles):
            a = r + m / self.n_particles
            while a > cum_func[i]:
                i += 1
            new_particles.append(self.particles_pred[i])
        
        self.particles = np.array(new_particles)
        self._state_bar = get_gaussian_statistics(self.particles)

        self._state.mu = self._state_bar.mu
        self._state.Sigma = self._state_bar.Sigma
