import os
import mediapy as media
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, NamedTuple, Optional, Union, List

# @title Import MuJoCo, MJX, and Brax
from datetime import datetime
from etils import epath
import functools
from IPython.display import HTML
from typing import Any, Dict, Sequence, Tuple, Union
import os
from ml_collections import config_dict


import jax
from jax import numpy as jp
import numpy as np
from flax.training import orbax_utils
from flax import struct
from matplotlib import pyplot as plt
import mediapy as media
from orbax import checkpoint as ocp
import time

import mujoco
from mujoco import mjx

from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, mjcf, model
from dataclasses import dataclass


class CommandSampler:
    def __init__(self, config):
        """
        Initializes the CommandSampler with the given configuration.

        Args:
            config: An object containing the following attributes:
                - min_lin_vel_x: Minimum linear velocity in x-direction.
                - max_lin_vel_x: Maximum linear velocity in x-direction.
                - min_lin_vel_y: Minimum linear velocity in y-direction.
                - max_lin_vel_y: Maximum linear velocity in y-direction.
                - min_ang_vel_yaw: Minimum angular velocity (yaw).
                - max_ang_vel_yaw: Maximum angular velocity (yaw).
                - prob_cmd_0: Probability of stopping (all zeros).
                - prob_cmd_1: Probability of max vx.
                - prob_cmd_2: Probability of min vx.
                - prob_cmd_3: Probability of max vy.
                - prob_cmd_4: Probability of max w.
                - prob_cmd_5: Probability of min w.
                - prob_cmd_6: Probability of random sampling.
        """
        self.config = config

    def sample_command(self, rng: jax.Array) -> jax.Array:
        """
        Samples a command vector based on predefined cases and a random sampling case.

        Args:
            rng: A JAX random number generator key.

        Returns:
            A JAX array containing [lin_vel_x, lin_vel_y, ang_vel_yaw].
        """
        # Split the RNG into multiple keys for different purposes
        keys = jax.random.split(rng, 8)
        key_case = keys[0]  # For selecting the case
        key_max_w = keys[1]  # For max angular velocity if needed
        key_min_w = keys[2]  # For min angular velocity if needed
        key_max_vx = keys[3]  # For max vx if needed
        key_min_vx = keys[4]  # For min vx if needed
        key_max_vy = keys[5]  # For max vy if needed
        key_random = keys[6]  # For random sampling case
        # keys[7] is unused but available if needed

        # Total number of cases: 7
        num_cases = 7

        # Sample a uniform random number in [0, 1) to select the case index
        u_case = jax.random.uniform(key_case, minval=0.0, maxval=1.0)
        case_index = jax.lax.floor(u_case * num_cases).astype(int)

        # Define each case as a separate function
        def case_stop(_):
            """Case 0: Stop (all velocities zero)"""
            return jp.array([0.0, 0.0, 0.0])

        def case_max_vx(_):
            """Case 1: Go only with maximum linear velocity in x"""
            return jp.array([self.config.max_lin_vel_x, 0.0, 0.0])

        def case_min_vx(_):
            """Case 2: Go only with minimum linear velocity in x"""
            return jp.array([self.config.min_lin_vel_x, 0.0, 0.0])

        def case_max_vy(_):
            """Case 3: Go only with maximum linear velocity in y"""
            return jp.array([0.0, self.config.max_lin_vel_y, 0.0])

        def case_max_w(_):
            """Case 4: Go only with maximum angular velocity"""
            return jp.array([0.0, 0.0, self.config.max_ang_vel_yaw])

        def case_min_w(_):
            """Case 5: Go only with minimum angular velocity"""
            return jp.array([0.0, 0.0, self.config.min_ang_vel_yaw])

        def case_random(_):
            """Case 6: Sample all velocities randomly within their respective bounds"""
            lin_vel_x = jax.random.uniform(
                key_max_vx,
                shape=(1,),
                minval=self.config.min_lin_vel_x,
                maxval=self.config.max_lin_vel_x,
            )
            lin_vel_y = jax.random.uniform(
                key_max_vy,
                shape=(1,),
                minval=self.config.min_lin_vel_y,
                maxval=self.config.max_lin_vel_y,
            )
            ang_vel_yaw = jax.random.uniform(
                key_max_w,
                shape=(1,),
                minval=self.config.min_ang_vel_yaw,
                maxval=self.config.max_ang_vel_yaw,
            )
            return jp.array([lin_vel_x[0], lin_vel_y[0], ang_vel_yaw[0]])

        # List of case functions
        cases: List[Callable] = [
            case_stop,  # Case 0
            case_max_vx,  # Case 1
            case_min_vx,  # Case 2
            case_max_vy,  # Case 3
            case_max_w,  # Case 4
            case_min_w,  # Case 5
            case_random,  # Case 6
        ]

        # Select and execute the appropriate case
        new_cmd = jax.lax.switch(case_index, cases, operand=None)

        return new_cmd


@dataclass
class Go2EnvConfig:
    # QUIM
    #
    rw_tracking_lin_vel: float = 1.5
    rw_tracking_ang_vel: float = 0.5
    rw_lin_vel_z: float = -2.0
    rw_ang_vel_xy: float = -0.05
    rw_orientation: float = -5.0 / 2.0
    rw_torques: float = -0.00001
    rw_action_rate: float = -0.01
    rw_feet_air_time: float = 0.1
    rw_stand_still: float = -0.5
    rw_termination: float = -1.0
    rw_foot_slip: float = -0.1
    rw_foot_clearance: float = 0.0
    rw_tracking_z: float = 0.0  # should be positive
    rw_diagonal_contact: float = 0.0  # Add this line
    rw_tracking_sigma: float = 0.25
    rw_tracking_z_sigma: float = 0.1 / 2.0
    rw_diagonal_contact_sigma: float = 0.25

    action_scale: float = 0.3
    kick_vel: float = 0.05
    scene_file: str = "../Go2Py/assets/mujoco/go2.xml"
    air_time_bias: float = 0.1
    timestep: float = 0.004
    target_base_z: float = 0.37
    p_mu_v: float = 0.3
    p_mu_v_min: float = 0.01
    p_mu_v_max: float = 0.2
    p_temperature: float = 0.1
    p_kp: float = 20.0
    p_kd: float = 0.5
    p_Fs: float = 0.1
    p_Fs_min: float = 0.0
    p_Fs_max: float = 0.12

    p_temperature_min: float = 0.08
    p_temperature_max: float = 0.12




    max_torque: float = 24
    min_torque: float = -24
    dt: float = 0.02

    foot_radius: float = 0.022
    len_obs_history: int = 5
    obs_size: int = 33
    prob_cmd_0: float = 0.05

    min_lin_vel_x: float = -0.6
    max_lin_vel_x: float = 1.5
    min_lin_vel_y: float = -0.8
    max_lin_vel_y: float = 0.8
    min_ang_vel_yaw: float = -0.7
    max_ang_vel_yaw: float = 0.7

    push_interval: int = 10
    clip_reward_step_min: float = -1.0
    clip_reward_step_max: float = 1.0
    resample_cmd_interval: int = 500

    obs_w_scaling: float = 0.25
    obs_cmd_scaling_x: float = 2.0
    obs_cmd_scaling_y: float = 2.0
    obs_cmd_scaling_w: float = 0.25

    randomize_custom_params: bool = True

    obs_noise: float = 0.05
    z_dead: float = 0.18

    geom_friction_min: float = 0.5
    geom_friction_max: float = 1.3

    prob_stop: float = 0.05
    prob_max_vx: float = 0.05
    prob_min_vx: float = 0.05
    prob_min_vy: float = 0.05
    prob_max_vy: float = 0.05
    prob_max_w: float = 0.05
    prob_min_w: float = 0.05
    prob_random: float = 1 - 0.35


def get_config():
    """Returns reward config for barkour quadruped environment."""

    def get_default_rewards_config():
        default_config = config_dict.ConfigDict(
            dict(
                # The coefficients for all reward terms used for training. All
                # physical quantities are in SI units, if no otherwise specified,
                # i.e. joint positions are in rad, positions are measured in meters,
                # torques in Nm, and time in seconds, and forces in Newtons.
                scales=config_dict.ConfigDict(
                    dict(
                        # Tracking rewards are computed using exp(-delta^2/sigma)
                        # sigma can be a hyperparameters to tune.
                        # Track the base x-y velocity (no z-velocity tracking.)
                        # tracking_lin_vel=1.5,
                        tracking_lin_vel=1.5,
                        # Track the angular velocity along z-axis, i.e. yaw rate.
                        tracking_ang_vel=0.5,
                        # Below are regularization terms, we roughly divide the
                        # terms to base state regularizations, joint
                        # regularizations, and other behavior regularizations.
                        # Penalize the base velocity in z direction, L2 penalty.
                        lin_vel_z=-2.0,
                        # Penalize the base roll and pitch rate. L2 penalty.
                        ang_vel_xy=-0.05,
                        # Penalize non-zero roll and pitch angles. L2 penalty.
                        orientation=-5.0 / 2.0,
                        # L2 regularization of joint torques, |tau|^2.
                        torques=-0.00001,
                        # Penalize the change in the action and encourage smooth
                        # actions. L2 regularization |action - last_action|^2
                        action_rate=-0.01,
                        # Encourage long swing steps.  However, it does not
                        # encourage high clearances.
                        feet_air_time=0.1,
                        # Encourage no motion at zero command, L2 regularization
                        # |q - q_default|^2.
                        stand_still=-0.5,
                        # Early termination penalty.
                        termination=-1.0,
                        # Penalizing foot slipping on the ground.
                        foot_slip=-0.1,
                        # reward the z position of the foot if it is not in contact
                        foot_clearance=0.0,
                        # reward having two-foot on air.
                        # two_feet_air=0.0,  # should be positive.
                        # height of the base
                        # tracking_z=2*2*10*0.1,  # should be positive
                        # diagonal_contact=5 * 0.1,  # Add this line
                        tracking_z=0.0,  # should be positive
                        diagonal_contact=0.0,  # Add this line
                    )
                ),
                # Tracking reward = exp(-error^2/sigma).
                tracking_sigma=0.25,
                tracking_z_sigma=0.1 / 2.0,
                diagonal_contact_sigma=0.25,
            )
        )
        return default_config

    default_config = config_dict.ConfigDict(
        dict(
            rewards=get_default_rewards_config(),
        )
    )

    return default_config


class Go2Env(PipelineEnv):
    """Environment for training the barkour quadruped joystick policy in MJX."""

    def __init__(self, config: Go2EnvConfig = Go2EnvConfig()):
        self.config = config

        sys = mjcf.load(self.config.scene_file)
        sys = sys.tree_replace({"opt.timestep": self.config.timestep})

        n_frames = int(self.config.dt / sys.opt.timestep)
        # print("n_frames is", n_frames)
        super().__init__(sys, backend="mjx", n_frames=n_frames)

        self.reward_config = get_config()


        self.reward_scales = {
            "tracking_lin_vel": self.config.rw_tracking_lin_vel,
            # Track the angular velocity along z-axis, i.e. yaw rate.
            "tracking_ang_vel": self.config.rw_tracking_ang_vel,
            # Below are regularization terms, we roughly divide the
            # terms to base state regularizations, joint
            # regularizations, and other behavior regularizations.
            # Penalize the base velocity in z direction, L2 penalty.
            "lin_vel_z": self.config.rw_lin_vel_z,
            # lin_vel_z=-2.0,
            # Penalize the base roll and pitch rate. L2 penalty.
            "ang_vel_xy": self.config.rw_ang_vel_xy,
            #  ang_vel_xy=-0.05,
            # Penalize non-zero roll and pitch angles. L2 penalty.
            "orientation": self.config.rw_orientation,
            # orientation=-5.0 / 2.0,
            # L2 regularization of joint torques, |tau|^2.
            "torques": self.config.rw_torques,
            # torques=-0.00001,
            # Penalize the change in the action and encourage smooth
            # actions. L2 regularization |action - last_action|^2
            "action_rate": self.config.rw_action_rate,
            # action_rate=-0.01,
            # Encourage long swing steps.  However, it does not
            # encourage high clearances.
            "feet_air_time": self.config.rw_feet_air_time,
            # feet_air_time=0.1,
            # Encourage no motion at zero command, L2 regularization
            # |q - q_default|^2.
            "stand_still": self.config.rw_stand_still,
            # stand_still=-0.5,
            # Early termination penalty.
            "termination": self.config.rw_termination,
            # termination=-1.0,
            # Penalizing foot slipping on the ground.
            "foot_slip": self.config.rw_foot_slip,
            # foot_slip=-0.1,
            # reward the z position of the foot if it is not in contact
            "foot_clearance": self.config.rw_foot_clearance,
            # foot_clearance=0.0,
            # reward having two-foot on air.
            # two_feet_air=0.0,  # should be positive.
            # height of the base
            # tracking_z=2*2*10*0.1,  # should be positive
            # diagonal_contact=5 * 0.1,  # Add this line
            # tracking_z=0.0,  # should be positive
            "tracking_z": self.config.rw_tracking_z,  # should be positive
            # diagonal_contact=0.0,  # Add this line
            "diagonal_contact": self.config.rw_diagonal_contact,  # Add this line
        }

        self.reward_extra_params = {
            "tracking_sigma": self.config.rw_tracking_sigma,
            "tracking_z_sigma": self.config.rw_tracking_z_sigma,
            "diagonal_contact_sigma": self.config.rw_diagonal_contact_sigma,
        }

        self._torso_idx = mujoco.mj_name2id(
            sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "base"
        )

        assert self._torso_idx != -1, "Torso not found."

        self._init_q = jp.array(sys.mj_model.keyframe("home").qpos)
        self._default_pose = sys.mj_model.keyframe("home").qpos[7:]
        # self.lowers = jp.array([-0.7, -1.0, 0.05] * 4)
        # self.uppers = jp.array([0.52, 2.1, 2.1] * 4)

        self.lowers = jp.array([-1, -1.5, -2.7] * 4)
        self.uppers = jp.array([1, 3.4, -0.83] * 4)

        feet_site = [
            "FR_foot",  #  'foot_front_right',
            "FL_foot",  # 'foot_front_left',
            "RR_foot",  # 'foot_hind_right',
            "RL_foot",  # 'foot_hind_left',
        ]
        feet_site_id = [
            mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, f)
            for f in feet_site
        ]
        assert not any(id_ == -1 for id_ in feet_site_id), "Site not found."

        # continue here!!
        self._feet_site_id = np.array(feet_site_id)
        lower_leg_body = [
            "FR_calf",  #'lower_leg_front_right',
            "FL_calf",  # 'lower_leg_front_left',
            "RR_calf",  #  'lower_leg_hind_right',
            "RL_calf",  #'lower_leg_hind_left',
        ]
        lower_leg_body_id = [
            mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, l)
            for l in lower_leg_body
        ]
        assert not any(id_ == -1 for id_ in lower_leg_body_id), "Body not found."
        self._lower_leg_body_id = np.array(lower_leg_body_id)
        self._nv = sys.nv
        # todo: add other modes when we only to x rotation, y rotation or yaw rotation.

    
    def sample_command(self, rng: jax.Array) -> jax.Array:
        prob_cmd_0 = self.config.prob_cmd_0

        _, key1, key2, key3, key4 = jax.random.split(rng, 5)
        u = jax.random.uniform(key4)

        def case_true(_):
            return jp.array([0.0, 0.0, 0.0])

        def case_false(_):
            lin_vel_x = jax.random.uniform(key1, (1,), minval=-0.6, maxval=1.5)
            lin_vel_y = jax.random.uniform(key2, (1,), minval=-0.8, maxval=0.8)
            ang_vel_yaw = jax.random.uniform(key3, (1,), minval=-0.7, maxval=0.7)
            return jp.array([lin_vel_x[0], lin_vel_y[0], ang_vel_yaw[0]])

        new_cmd = jax.lax.cond(u < prob_cmd_0, case_true, case_false, operand=None)
        return new_cmd
    
    
    # def sample_command(self, rng: jax.Array) -> jax.Array:

    #     keys = jax.random.split(rng, 8)
    #     key_case = keys[0]
    #     key_other = keys[1]
    #     # ... other keys as before

    #     # Define cumulative probabilities for each case
    #     cumulative_probs = jp.array([
    #         self.config.prob_stop,
    #         self.config.prob_max_vx,
    #         self.config.prob_min_vx,
    #         self.config.prob_max_vy,
    #         self.config.prob_min_vy,
    #         self.config.prob_max_w,
    #         self.config.prob_min_w,
    #         self.config.prob_random
    #     ]).cumsum()

    #     u_case = jax.random.uniform(key_case, minval=0.0, maxval=1.0)


    #     # Define each case as a separate function
    #     def case_stop(_):
    #         """Case 0: Stop (all velocities zero)"""
    #         return jp.array([0.0, 0.0, 0.0])

    #     def case_max_vx(_):
    #         """Case 1: Go only with maximum linear velocity in x"""
    #         return jp.array([self.config.max_lin_vel_x, 0.0, 0.0])

    #     def case_min_vx(_):
    #         """Case 2: Go only with minimum linear velocity in x"""
    #         return jp.array([self.config.min_lin_vel_x, 0.0, 0.0])

    #     def case_max_vy(_):
    #         """Case 3: Go only with maximum linear velocity in y"""
    #         return jp.array([0.0, self.config.max_lin_vel_y, 0.0])

    #     def case_max_w(_):
    #         """Case 4: Go only with maximum angular velocity"""
    #         return jp.array([0.0, 0.0, self.config.max_ang_vel_yaw])

    #     def case_min_w(_):
    #         """Case 5: Go only with minimum angular velocity"""
    #         return jp.array([0.0, 0.0, self.config.min_ang_vel_yaw])

    #     def case_random(_):
    #         """Case 6: Sample all velocities randomly within their respective bounds"""
    #         lin_vel_x = jax.random.uniform(
    #             key_other,
    #             shape=(1,),
    #             minval=self.config.min_lin_vel_x,
    #             maxval=self.config.max_lin_vel_x,
    #         )
    #         lin_vel_y = jax.random.uniform(
    #             key_other,
    #             shape=(1,),
    #             minval=self.config.min_lin_vel_y,
    #             maxval=self.config.max_lin_vel_y,
    #         )
    #         ang_vel_yaw = jax.random.uniform(
    #             key_other,
    #             shape=(1,),
    #             minval=self.config.min_ang_vel_yaw,
    #             maxval=self.config.max_ang_vel_yaw,
    #         )
    #         return jp.array([lin_vel_x[0], lin_vel_y[0], ang_vel_yaw[0]])
        
    #       # List of case functions
    #     cases: List[Callable] = [
    #         case_stop,  # Case 0
    #         case_max_vx,  # Case 1
    #         case_min_vx,  # Case 2
    #         case_max_vy,  # Case 3
    #         case_max_w,  # Case 4
    #         case_min_w,  # Case 5
    #         case_random,  # Case 6
    #     ]
        
    #     case_index = jp.sum(u_case >= cumulative_probs).astype(int)


    #     # Select and execute the appropriate case
    #     new_cmd = jax.lax.switch(case_index, cases, operand=None)

    #     return new_cmd


        # # Determine the case index based on where u_case falls in the cumulative distribution
        # # For example, if u_case < prob_stop => case 0,
        # # elif u_case < prob_stop + prob_max_vx => case 1, etc.
        # case_index = jp.sum(u_case >= cumulative_probs).astype(int)

        # # Define cases as before
        # # ...

        # # Execute the selected case
        # new_cmd = jax.lax.switch(case_index, cases, operand=None)

        # return new_cmd




        # _, key1, key2, key3, key4 = jax.random.split(rng, 5)
        # u = jax.random.uniform(key4)

        # def case_true(_):
        #     return jp.array([0.0, 0.0, 0.0])

        # def case_false(_):
        #     lin_vel_x = jax.random.uniform(
        #         key1,
        #         (1,),
        #         minval=self.config.min_lin_vel_x,
        #         maxval=self.config.max_lin_vel_x,
        #     )
        #     lin_vel_y = jax.random.uniform(
        #         key2,
        #         (1,),
        #         minval=self.config.min_lin_vel_y,
        #         maxval=self.config.max_lin_vel_y,
        #     )
        #     ang_vel_yaw = jax.random.uniform(
        #         key3,
        #         (1,),
        #         minval=self.config.min_ang_vel_yaw,
        #         maxval=self.config.max_ang_vel_yaw,
        #     )
        #     return jp.array([lin_vel_x[0], lin_vel_y[0], ang_vel_yaw[0]])

        # new_cmd = jax.lax.cond(
        #     u < self.config.prob_cmd_0, case_true, case_false, operand=None
        # )


        


        # return new_cmd

    def reset(self, rng: jax.Array) -> State:  # pytype: disable=signature-mismatch
        rng, key = jax.random.split(rng)

        pipeline_state = self.pipeline_init(self._init_q, jp.zeros(self._nv))

        state_info = {
            "rng": rng,
            "last_act": jp.zeros(12),
            "last_vel": jp.zeros(12),
            "command": self.sample_command(key),
            "last_contact": jp.zeros(4, dtype=bool),
            "feet_air_time": jp.zeros(4),
            "rewards": {k: 0.0 for k in self.reward_scales.keys()},
            "kick": jp.array([0.0, 0.0]),
            "step": 0,
            "p_mu_v": self.config.p_mu_v,
            "p_Fs": self.config.p_Fs,
            "p_temperature": self.config.p_temperature,
            "p_kp": self.config.p_kp,
            "p_kd": self.config.p_kd,
        }

        if self.config.randomize_custom_params:

            r1, r2, r3, r4 = jax.random.split(rng, 4)
            # state_info["p_mu_v"] = jax.random.uniform(rng, minval=0, maxval=0.5)

            # state_info["p_Fs"] = jax.random.uniform(rng, minval=0, maxval=3.0)
            # state_info["p_Fs"] = jax.random.uniform(rng, minval=0., maxval=.2)

            # state_info["p_Fs"] = jax.random.uniform(rng, minval=0.0, maxval=0.2)

            state_info["p_temperature"] = jax.random.uniform(
                r1,
                minval=self.config.p_temperature_min,
                maxval=self.config.p_temperature_max,
            )

            state_info["p_Fs"] = jax.random.uniform(
                r2, minval=self.config.p_Fs_min, maxval=self.config.p_Fs_max
            )
            state_info["p_mu_v"] = jax.random.uniform(
                r3, minval=self.config.p_mu_v_min, maxval=self.config.p_mu_v_max
            )

            state_info["p_kp"] = jax.random.uniform(rng, minval=20.0, maxval=22.0)
            state_info["p_kd"] = jax.random.uniform(rng, minval=0.5, maxval=0.7)

            #
            # state_info["p_mu_v"] = jp.clip(state_info["p_mu_v"], 0.01, 1e6)
            # state_info["p_Fs"] = jp.clip(state_info["p_Fs"], 0.01, 1e6)
            # state_info["p_temperature"] = jp.clip(
            #     state_info["p_temperature"], 0.01, 1e6
            # )
            # state_info["p_kd"] = jp.clip(state_info["p_kd"], 0.01, 1e6)
            # state_info["p_kp"] = jp.clip(state_info["p_kp"], 0.05, 1e6)

        obs = jp.zeros(
            self.config.len_obs_history * self.config.obs_size
        )  # store 15 steps of history

        # repeat the same observation for the history
        # current_obs = obs[:31]

        for _ in range(self.config.len_obs_history):
            obs = self._get_obs(pipeline_state, state_info, obs)
            # obs = jp.roll(obs_history, obs.size).at[: current_obs.size].set(current_obs)
        # jax.debug.print("obs = {}", obs)
        # jax.debug.print("obs shape = {}", obs.shape)

        reward, done = jp.zeros(2)
        metrics = {"total_dist": 0.0}
        for k in state_info["rewards"]:
            metrics[k] = state_info["rewards"][k]

        # jax.debug.print("obs = {}", obs)
        # jax.debug.print("obs shape = {}", obs.shape)

        state = State(
            pipeline_state, obs, reward, done, metrics, state_info
        )  # pytype: disable=wrong-arg-types
        return state

    def q_pipeline_step(self, state_all: Any, action: jax.Array) -> base.State:
        """Takes a physics step using the physics pipeline."""

        def f(state, _):

            Fs = state_all.info["p_Fs"]
            mu_v = state_all.info["p_mu_v"]
            temperature = state_all.info["p_temperature"]
            kp = state_all.info["p_kp"]
            kd = state_all.info["p_kd"]
            q = state.q[7:]
            dq = state.qd[6:]
            tau = (
                kp * (action - q) - kd * dq - mu_v * dq - Fs * jp.tanh(dq / temperature)
            )

            # tau_sticktion = .1 * jp.tanh(dq / .1)
            # tau_viscose = 1. * dq
            # tau = 20 * (action - q) - .5 * dq
            # tau -= tau_viscose
            # tau -= tau_sticktion
            tau = jp.clip(tau, self.config.min_torque, self.config.max_torque)
            return (
                self._pipeline.step(self.sys, state, tau, self._debug),
                None,
            )

        return jax.lax.scan(f, state_all.pipeline_state, (), self._n_frames)[0]

    def step(
        self, state: State, action: jax.Array
    ) -> State:  # pytype: disable=signature-mismatch
        rng, cmd_rng, kick_noise_2 = jax.random.split(state.info["rng"], 3)

        # jax.debug.print("action = {}", action)
        # jax.debug.print("action = {}", action.shape)

        # print("check step")
        # kick

        kick_theta = jax.random.uniform(kick_noise_2, maxval=2 * jp.pi)
        kick = jp.array([jp.cos(kick_theta), jp.sin(kick_theta)])
        kick *= jp.mod(state.info["step"], self.config.push_interval) == 0
        qvel = state.pipeline_state.qvel  # pytype: disable=attribute-error
        qvel = qvel.at[:2].set(kick * self.config.kick_vel + qvel[:2])
        state = state.tree_replace({"pipeline_state.qvel": qvel})

        # physics step
        motor_targets = self._default_pose + action * self.config.action_scale
        motor_targets = jp.clip(motor_targets, self.lowers, self.uppers)
        #
        pipeline_state = self.q_pipeline_step(state, motor_targets)
        x, xd = pipeline_state.x, pipeline_state.xd

        # observation data
        obs = self._get_obs(pipeline_state, state.info, state.obs)
        joint_angles = pipeline_state.q[7:]
        joint_vel = pipeline_state.qd[6:]

        # foot contact data based on z-position
        foot_pos = pipeline_state.site_xpos[
            self._feet_site_id
        ]  # pytype: disable=attribute-error
        foot_contact_z = foot_pos[:, 2] - self.config.foot_radius
        contact = foot_contact_z < 1e-3  # a mm or less off the floor
        contact_filt_mm = contact | state.info["last_contact"]
        contact_filt_cm = (foot_contact_z < 3e-2) | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0) * contact_filt_mm
        state.info["feet_air_time"] += self.dt

        # done if joint limits are reached or robot is falling
        up = jp.array([0.0, 0.0, 1.0])

        done = jp.dot(math.rotate(up, x.rot[self._torso_idx - 1]), up) < 0
        # jax.debug.print("x = {}", x)
        # jax.debug.print("p = {}",  math.rotate(up, x.rot[self._torso_idx - 1]) )

        # print("x is ", x)
        # print(math.rotate(up, x.rot[self._torso_idx - 1]))
        # print(jp.dot(math.rotate(up, x.rot[self._torso_idx - 1]), up))
        # print(joint_angles)
        # print(self.lowers)
        # print(self.uppers)
        # print(pipeline_state.x.pos[self._torso_idx - 1, 2])

        done |= jp.any(joint_angles < self.lowers)
        done |= jp.any(joint_angles > self.uppers)
        done |= pipeline_state.x.pos[self._torso_idx - 1, 2] < self.config.z_dead

        # reward
        rewards = {
            "tracking_lin_vel": (
                self._reward_tracking_lin_vel(state.info["command"], x, xd)
            ),
            "tracking_ang_vel": (
                self._reward_tracking_ang_vel(state.info["command"], x, xd)
            ),
            "lin_vel_z": self._reward_lin_vel_z(xd),
            "ang_vel_xy": self._reward_ang_vel_xy(xd),
            "orientation": self._reward_orientation(x),
            "torques": self._reward_torques(
                pipeline_state.qfrc_actuator
            ),  # pytype: disable=attribute-error
            "action_rate": self._reward_action_rate(action, state.info["last_act"]),
            "stand_still": self._reward_stand_still(
                state.info["command"],
                joint_angles,
            ),
            "feet_air_time": self._reward_feet_air_time(
                state.info["feet_air_time"],
                first_contact,
                state.info["command"],
            ),
            "foot_slip": self._reward_foot_slip(pipeline_state, contact_filt_cm),
            "termination": self._reward_termination(done, state.info["step"]),
            # TODO: check
            "foot_clearance": self._reward_foot_clearance(
                pipeline_state, contact_filt_cm
            ),
            "diagonal_contact": self._reward_diagonal_contact(
                contact_filt_cm, state.info["command"]
            ),
            # "two_feet_air": self._reward_two_feet_air(
            #     state.info["command"], contact_filt_cm
            # ), TODO: fix issue with
            "tracking_z": self._reward_tracking_z(x),
        }

        rewards = {
            k: v * self.reward_config.rewards.scales[k] for k, v in rewards.items()
        }

        reward = jp.clip(sum(rewards.values()) * self.dt, -1.0, 1.0)


        # rewards = {k: v * self.reward_scales[k] for k, v in rewards.items()}
        # # reward = jp.clip(
        # #     sum(rewards.values()) * self.dt, 0.0, 10000.0
        # # )  # why is reward clipped?
        # reward = jp.clip(
        #     sum(rewards.values()) * self.dt,
        #     self.config.clip_reward_step_min,
        #     self.config.clip_reward_step_max,
        # )

        # state management
        state.info["kick"] = kick
        state.info["last_act"] = action
        state.info["last_vel"] = joint_vel
        state.info["feet_air_time"] *= ~contact_filt_mm
        state.info["last_contact"] = contact
        state.info["rewards"] = rewards
        state.info["step"] += 1
        state.info["rng"] = rng

        # sample new command if more than 500 timesteps achieved
        state.info["command"] = jp.where(
            state.info["step"] > self.config.resample_cmd_interval,
            self.sample_command(cmd_rng),
            state.info["command"],
        )
        # reset the step counter when done
        state.info["step"] = jp.where(
            done | (state.info["step"] > self.config.resample_cmd_interval),
            0,
            state.info["step"],
        )

        # log total displacement as a proxy metric
        state.metrics["total_dist"] = math.normalize(x.pos[self._torso_idx - 1])[1]
        state.metrics.update(state.info["rewards"])

        done = jp.float32(done)
        state = state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )
        # print("step finished")
        return state

    def _get_obs(
        self,
        pipeline_state: base.State,
        state_info: dict[str, Any],
        obs_history: jax.Array,
    ) -> jax.Array:
        inv_torso_rot = math.quat_inv(pipeline_state.x.rot[0])
        # jax.debug.print("inv_torso_rot = {}", inv_torso_rot)
        # jax.debug.print("pos of torso = {}", pipeline_state.x.pos[0])

        local_rpyrate = math.rotate(pipeline_state.xd.ang[0], inv_torso_rot)
        obs_cmd_scaling = jp.array(
            [
                self.config.obs_cmd_scaling_x,
                self.config.obs_cmd_scaling_y,
                self.config.obs_cmd_scaling_w,
            ]
        )

        obs = jp.concatenate(
            [
                local_rpyrate * self.config.obs_w_scaling,  # yaw rate
                math.rotate(jp.array([0, 0, -1]), inv_torso_rot),  # projected gravity
                state_info["command"] * obs_cmd_scaling,  # command
                pipeline_state.q[7:] - self._default_pose,  # motor angles
                state_info["last_act"],  # last action
            ]
        )

        # clip, noise
        obs = jp.clip(obs, -100.0, 100.0) + self.config.obs_noise * jax.random.uniform(
            state_info["rng"], obs.shape, minval=-1, maxval=1
        )
        # stack observations through time
        obs = jp.roll(obs_history, obs.size).at[: obs.size].set(obs)
        # jax.debug.print("obs = {}", obs)
        # jax.debug.print("obs shape = {}", obs.shape)

        return obs

    def _reward_diagonal_contact(
        self, contact: jax.Array, commands: jax.Array
    ) -> jax.Array:
        """Reward for having diagonal legs in contact simultaneously without 2D tensors."""
        contact_f = contact.astype(jp.float32)
        # Desired patterns for diagonal contacts
        # notes: Front and Right are exchanged.
        # Pattern 1: FL and RR in contact
        error1 = (
            jp.abs(contact_f[0] - 1.0)  # FL
            + jp.abs(contact_f[1] - 0.0)  # RL
            + jp.abs(contact_f[2] - 0.0)  # FR
            + jp.abs(contact_f[3] - 1.0)  # RR
        )
        # Pattern 2: FR and RL in contact
        error2 = (
            jp.abs(contact_f[0] - 0.0)  # FL
            + jp.abs(contact_f[1] - 1.0)  # RL
            + jp.abs(contact_f[2] - 1.0)  # FR
            + jp.abs(contact_f[3] - 0.0)  # RR
        )
        # Compute rewards for each pattern
        reward1 = jp.exp(-error1 / self.reward_extra_params["diagonal_contact_sigma"])
        reward2 = jp.exp(-error2 / self.reward_extra_params["diagonal_contact_sigma"])
        # Take the maximum reward
        reward = jp.maximum(reward1, reward2)
        # Apply condition: remove reward if command norm < 0.05
        command_speed = math.normalize(commands[:2])[1]
        reward *= command_speed >= 0.05

        return reward

    # ------------ reward functions----------------
    def _reward_lin_vel_z(self, xd: Motion) -> jax.Array:
        # Penalize z axis base linear velocity
        return jp.square(xd.vel[0, 2])

    def _reward_ang_vel_xy(self, xd: Motion) -> jax.Array:
        # Penalize xy axes base angular velocity
        return jp.sum(jp.square(xd.ang[0, :2]))

    def _reward_orientation(self, x: Transform) -> jax.Array:
        # Penalize non flat base orientation
        up = jp.array([0.0, 0.0, 1.0])
        rot_up = math.rotate(up, x.rot[0])
        return jp.sum(jp.square(rot_up[:2]))

    def _reward_torques(self, torques: jax.Array) -> jax.Array:
        # Penalize torques
        return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))

    def _reward_action_rate(self, act: jax.Array, last_act: jax.Array) -> jax.Array:
        # Penalize changes in actions
        return jp.sum(jp.square(act - last_act))

    def _reward_tracking_lin_vel(
        self, commands: jax.Array, x: Transform, xd: Motion
    ) -> jax.Array:
        # Tracking of linear velocity commands (xy axes)
        local_vel = math.rotate(xd.vel[0], math.quat_inv(x.rot[0]))
        lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
        lin_vel_reward = jp.exp(
            -lin_vel_error / self.reward_extra_params["tracking_sigma"]
        )
        return lin_vel_reward

    def _reward_tracking_z(self, x: Transform) -> jax.Array:
        z = x.pos[self._torso_idx - 1, 2]
        z_error = jp.square(self.config.target_base_z - z)
        return jp.exp(-z_error / self.reward_extra_params["tracking_z_sigma"])

    def _reward_tracking_ang_vel(
        self, commands: jax.Array, x: Transform, xd: Motion
    ) -> jax.Array:
        # Tracking of angular velocity commands (yaw)
        base_ang_vel = math.rotate(xd.ang[0], math.quat_inv(x.rot[0]))
        ang_vel_error = jp.square(commands[2] - base_ang_vel[2])
        return jp.exp(-ang_vel_error / self.reward_extra_params["tracking_sigma"])

    def _reward_feet_air_time(
        self, air_time: jax.Array, first_contact: jax.Array, commands: jax.Array
    ) -> jax.Array:
        # Reward air time.
        rew_air_time = jp.sum((air_time - self.config.air_time_bias) * first_contact)
        rew_air_time *= (
            math.normalize(commands[:2])[1] > 0.05
        )  # no reward for zero command
        return rew_air_time

    def _reward_stand_still(
        self,
        commands: jax.Array,
        joint_angles: jax.Array,
    ) -> jax.Array:
        # Penalize motion at zero commands
        return jp.sum(jp.abs(joint_angles - self._default_pose)) * (
            math.normalize(commands[:2])[1] < 0.1
        )

    def _reward_foot_slip(
        self, pipeline_state: base.State, contact_filt: jax.Array
    ) -> jax.Array:
        # get velocities at feet which are offset from lower legs
        # pytype: disable=attribute-error
        pos = pipeline_state.site_xpos[self._feet_site_id]  # feet position
        feet_offset = pos - pipeline_state.xpos[self._lower_leg_body_id]
        # pytype: enable=attribute-error
        offset = base.Transform.create(pos=feet_offset)
        foot_indices = self._lower_leg_body_id - 1  # we got rid of the world body
        foot_vel = offset.vmap().do(pipeline_state.xd.take(foot_indices)).vel

        # Penalize large feet velocity for feet that are in contact with the ground.
        return jp.sum(jp.square(foot_vel[:, :2]) * contact_filt.reshape((-1, 1)))

    def _reward_foot_clearance(
        self, pipeline_state: base.State, contact_filt: jax.Array
    ):
        # if a foot is not in contact, penalize the z position.
        # TODO: check!!!
        pos = pipeline_state.site_xpos[self._feet_site_id]  # feet position
        z_pos = pos[:, 2]
        # no_contact = ~contact_filt
        return jp.sum(z_pos)

    def _reward_two_feet_air(self, commands: jax.Array, contact_filt_cm: jax.Array):
        feet = jp.ones(4)
        num_feet = jp.sum(feet * contact_filt_cm)
        dif = -1 * jp.abs(num_feet - 2)  # 0, 1 , 2
        dif *= math.safe_norm(commands[:2]) > 0.05  # no reward for zero command
        return dif

    def _reward_termination(self, done: jax.Array, step: jax.Array) -> jax.Array:
        return done & (step < 500)

    def render(
        self,
        trajectory: List[base.State],
        camera: str | None = None,
        width: int = 240,
        height: int = 320,
    ) -> Sequence[np.ndarray]:
        camera = camera or "track"
        return super().render(trajectory, camera=camera, width=width, height=height)

        # lets start the training loop

    def domain_randomize(self, sys, rng):
        """Randomizes the mjx.Model."""

        @jax.vmap
        def rand(rng):
            _, key = jax.random.split(rng, 2)
            # friction
            friction = jax.random.uniform(
                key,
                (1,),
                minval=self.config.geom_friction_min,
                maxval=self.config.geom_friction_max,
            )
            friction = sys.geom_friction.at[:, 0].set(friction)
            # todo: randomize something else!
            return friction

        friction = rand(rng)

        in_axes = jax.tree_util.tree_map(lambda x: None, sys)
        in_axes = in_axes.tree_replace({"geom_friction": 0})

        sys = sys.tree_replace(
            {
                "geom_friction": friction,
            }
        )

        return sys, in_axes
