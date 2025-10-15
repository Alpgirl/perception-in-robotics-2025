#!/usr/bin/python

"""
Sudhanva Sreesha
ssreesha@umich.edu
22-Apr-2018

Gonzalo Ferrer
g.ferrer@skoltech.ru
28-February-2021
"""

import contextlib
import os
from argparse import ArgumentParser

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from tqdm import tqdm

from tools.objects import Gaussian
from tools.plot import get_plots_figure
from tools.plot import plot_robot
from field_map import FieldMap
#from slam import SimulationSlamBase
from tools.data import generate_data as generate_input_data
from tools.data import load_data
from tools.plot import plot_field
from tools.plot import plot_observations, plot2dcov
from tools.task import get_dummy_context_mgr
from tools.task import get_movie_writer
from tools.helpers import get_cli_args, validate_cli_args

import mrob

from slam.slamBase import SlamBase
from slam.graphSlam import GraphSLAM


def main():
    args = get_cli_args()
    validate_cli_args(args)
    alphas = np.array(args.alphas) ** 2
    beta = np.array(args.beta)
    beta[1] = np.deg2rad(beta[1])


    mean_prior = np.array([180., 50., 0.])
    Sigma_prior = 1e-12 * np.eye(3, 3)
    initial_state = Gaussian(mean_prior, Sigma_prior)

    if args.input_data_file:
        data = load_data(args.input_data_file)
    elif args.num_steps:
        # Generate data, assuming `--num-steps` was present in the CL args.
        data = generate_input_data(initial_state.mu.T,
                                   args.num_steps,
                                   args.num_landmarks_per_side,
                                   args.max_obs_per_time_step,
                                   alphas,
                                   beta,
                                   args.dt)
    else:
        raise RuntimeError('')

    should_show_plots = True if args.animate else False
    should_write_movie = True if args.movie_file else False
    should_update_plots = True if should_show_plots or should_write_movie else False

    field_map = FieldMap(args.num_landmarks_per_side)

    fig = get_plots_figure(should_show_plots, should_write_movie)
    movie_writer = get_movie_writer(should_write_movie, 'Simulation SLAM', args.movie_fps, args.plot_pause_len)

    # initialize slam
    if args.filter_name == "graphSlam":
        filter = GraphSLAM(initial_state=initial_state, 
                           alphas=alphas, 
                           slam_type=args.filter_name, 
                           data_association=args.data_association,
                           update_type=args.update_type,
                           Q=beta
                           )
    else:
        raise NotImplementedError("The filter is not implemented")

    chi2_errs = []
    with movie_writer.saving(fig, args.movie_file, data.num_steps) if should_write_movie else get_dummy_context_mgr():
        for t in tqdm(range(data.num_steps)):
            # Used as means to include the t-th time-step while plotting.
            tp1 = t + 1

            # Control at the current step.
            u = data.filter.motion_commands[t]
            # Observation at the current step.
            z = data.filter.observations[t]

            print(f"Step: {t}")
            # TODO SLAM predict(u)
            filter.predict(u)
            # filter.graph.print(False)
            
            # TODO SLAM update
            filter.update(z)
            # filter.graph.print(False)

            if not should_update_plots:
                continue

            plt.cla()
            plot_field(field_map, z)
            plot_robot(data.debug.real_robot_path[t])
            plot_observations(data.debug.real_robot_path[t],
                              data.debug.noise_free_observations[t],
                              data.filter.observations[t])

            plt.plot(data.debug.real_robot_path[1:tp1, 0], data.debug.real_robot_path[1:tp1, 1], 'm')
            plt.plot(data.debug.noise_free_robot_path[1:tp1, 0], data.debug.noise_free_robot_path[1:tp1, 1], 'g')

            plt.plot([data.debug.real_robot_path[t, 0]], [data.debug.real_robot_path[t, 1]], '*r')
            plt.plot([data.debug.noise_free_robot_path[t, 0]], [data.debug.noise_free_robot_path[t, 1]], '*g')

            # TODO plot SLAM solution
            if args.solve_iter:
                filter.solve()

            # separate robot states and landmarks
            filter_obs, filter_state = [], []
            for state in filter.graph.get_estimated_state():
                if state.shape[0] == 2:
                    filter_obs.append(state)
                else:
                    filter_state.append(state)
            filter_obs = np.stack(filter_obs, axis=0)
            filter_state = np.stack(filter_state, axis=0)

            # Task 2B: plot the filtered trajectory
            plt.plot(filter_state[:,0,0], filter_state[:,1,0], color='b', label=f"{args.filter_name} trajectory")
            plt.scatter(filter_obs[:,0,0], filter_obs[:,1,0], color='orange', label=f"{args.filter_name} landmarks")

            # calculate chi2-error
            chi2_errs.append(filter.chi2())
            filter.graph.print(False)

            if should_show_plots:
                # Draw all the plots and pause to create an animation effect.
                plt.draw()
                plt.legend()
                plt.pause(args.plot_pause_len)

            if should_write_movie:
                movie_writer.grab_frame()

    # task 2E: solve until convergence
    if not args.solve_iter:
        err = 1e9
        tol = 1e-4
        prev_err = 1e10
        iter = 0
        while prev_err > err + tol:
            prev_err = err
            filter.graph.solve(mrob.LM)
            err = filter.chi2()
            print(f"chi^2 error = {err}")
            iter += 1
        print(f"Number of iterations: {iter}")

    # task 2A: calculate chi2 error
    if args.plot_chi2:
        plt.figure()
        plt.plot(np.arange(data.num_steps), chi2_errs)
        plt.xlabel("Time steps")
        plt.ylabel("chi2 error")

    # Task 2C: plot the adjacency matrix and information matrix
    if args.plot_matrices:
        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].spy(filter.graph.get_adjacency_matrix().toarray(), marker='*', markersize=3, color='b')
        ax[0].set_title("Adjacency matrix")
        ax[1].spy(filter.graph.get_information_matrix().toarray(), marker='*', markersize=3, color='b')
        ax[1].set_title("Information matrix")

    # Task 2D: plot the 3-sigma iso-contour of the covariance of the last pose
    # find the starting index of covariance of last robot pose in information matrix
    if args.plot_iso_contours:
        ind = 0
        for el in filter.graph.get_estimated_state():
            if np.array_equal(el, filter_state[-1,]):
                break
            else:
                ind += el.shape[0]
        covariance = np.linalg.inv(filter.graph.get_information_matrix().todense()[ind:ind+3, ind:ind+3])
        plot2dcov(filter_state[-1,:2,0], covariance[:2,:2], 'm', 3, legend="3-sigma iso-contour")

    plt.show(block=True)

if __name__ == '__main__':
    main()


