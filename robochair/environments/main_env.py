from configs.environments.genesis_simple_cfg import GenesisEnvConfig
from .robot import Robot
from .corridor import Corridor
from .goal import Goal
from .vision import Grid
import math

import torch
import genesis as gs
from .utils import check_agent_collision, calculate_behind_camera_pos


class Rewards:
    def __init__(
        self,
        env_ctx,
    ):
        self.env_ctx = env_ctx
        self.num_envs = env_ctx.num_envs
        self.device = env_ctx.device

        self.reward_scales = {
            "action_rate": -0.0001,
            "goal_dist": -0.1,
            "relative_angle": -100,
            "speed_towards_goal": 100,
            "colision": -500,
        }

        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.env_ctx.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros(
                (self.num_envs,), device=self.device, dtype=gs.tc_float
            )

        self.rew_buf = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )

    def __call__(self):
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        return self.rew_buf

    def reset_idx(self, envs_idx):
        extras_episode = {}
        for key in self.episode_sums.keys():
            extras_episode["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item()
                / self.env_ctx.episode_length_s
            )
            self.episode_sums[key][envs_idx] = 0.0
        return extras_episode

    def _reward_speed_towards_goal(self):
        p_obj = self.env_ctx.robot.get_pos()  # Форма (N, 3)
        p_tgt = self.env_ctx.goal.get_pos()  # Форма (N, 3)
        v_obj = self.env_ctx.robot.get_global_vel()  # Форма (N, 3)

        vector = p_tgt - p_obj  # Форма (N, 3)
        magnitude = torch.linalg.norm(vector, dim=1)
        epsilon = 1e-9
        direction = vector / (magnitude + epsilon).unsqueeze(1)
        speed_towards = torch.sum(v_obj * direction, dim=1)
        return speed_towards

    def _reward_action_rate(self):
        return torch.sum(
            torch.square(self.env_ctx.last_actions - self.env_ctx.actions), dim=1
        )

    def _reward_colision(self):
        return check_agent_collision(self.env_ctx.robot.robot, 2).float()

    def _reward_goal_dist(self):
        dist = torch.norm(self.env_ctx.goal.get_pos() - self.env_ctx.robot.get_pos())
        return dist.abs()

    def _reward_relative_angle(self):
        return torch.abs(self.env_ctx.robot.buffer.relative_angle).squeeze(-1)


class Env:
    def __init__(
        self,
        cfg: GenesisEnvConfig,
        num_envs,
        show_viewer=False,
        arches_using=False,
        registered_keys=None,
        gta_cam=False,
    ):

        self.device = cfg.device
        self.show_viewer = show_viewer
        self.gta_cam = gta_cam

        self.num_actions = 4
        self.num_envs = num_envs

        self.resampling_time_s = 20.0
        self.episode_length_s = 20.0

        self.simulate_action_latency = True
        self.dt = 0.02  # control frequency on real robot is 50hz
        self.max_episode_length = math.ceil(self.episode_length_s / self.dt)

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
                registered_keys=registered_keys,
            ),
            # vis_options=gs.options.VisOptions(n_rendered_envs=1),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        self.robot = Robot(self.scene, num_envs)
        self.corridor = Corridor(self.scene, num_envs, arches_using)
        self.goal = Goal(self.scene, num_envs)

        self.scene.build(n_envs=num_envs)
        self.robot.build()

        grid_ignore_idx = self.robot.idx() + self.goal.idx()
        self.grid = Grid(self.scene, ignore_idx=grid_ignore_idx, num_envs=num_envs)

        self.actions = torch.zeros(
            (self.num_envs, self.num_actions),
            device=self.device,
            dtype=gs.tc_float,
        )
        self.last_actions = torch.zeros_like(self.actions)

        self.reset_buf = torch.ones(
            (self.num_envs,), device=self.device, dtype=gs.tc_int
        )
        self.episode_length_buf = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_int
        )
        self.extras = dict()

        self.rewards = Rewards(self)
        self.episode_sums = self.rewards.episode_sums

    def _check_reset(self):

        self.reset_buf = self.episode_length_buf > self.max_episode_length

        time_out_idx = (
            (self.episode_length_buf > self.max_episode_length)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self.extras["time_outs"] = torch.zeros_like(
            self.reset_buf, device=self.device, dtype=gs.tc_float
        )
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        return self.reset_buf

    def _resample(self):
        time_condition = (
            self.episode_length_buf % int(self.resampling_time_s / self.dt) == 0
        )
        distance_condition = (
            torch.norm(self.goal.get_pos() - self.robot.get_pos(), dim=1) < 1.3
        )
        combined_condition = time_condition | distance_condition
        envs_idx = combined_condition.nonzero(as_tuple=False).flatten()

        # self.robot.reset_idx(envs_idx)
        self.goal.reset_idx(envs_idx)

    def _cam_update(self):
        env_index_to_render = 0
        look_at_z_offset = 1.0

        robot_pos = self.robot.buffer.base_pos[env_index_to_render].cpu()
        robot_euler = self.robot.buffer.base_euler[env_index_to_render].cpu()

        cam_pos_tensor = calculate_behind_camera_pos(
            robot_pos,
            robot_euler,
        )
        look_at_tensor = robot_pos + torch.tensor([0.0, 0.0, look_at_z_offset])

        final_cam_pos_np = cam_pos_tensor.numpy()
        final_lookat_np = look_at_tensor.numpy()

        viewer = self.scene.viewer
        if viewer:
            viewer.set_camera_pose(
                pos=final_cam_pos_np,
                lookat=final_lookat_np,
            )

    def step(self, actions):

        self.last_actions[:] = self.actions[:]
        self.actions[:] = actions

        self.robot.step(self.last_actions)
        self.scene.step()
        self.robot.update_buffer(relative_pos=self.goal.get_pos())
        self.episode_length_buf += 1

        self._resample()

        self.grid.update(self.robot.buffer.base_pos, self.robot.buffer.base_quat)

        obs = self.get_observations()
        rew = self.rewards()

        is_reset = self._check_reset()

        if self.gta_cam:
            self._cam_update()

        return obs, None, rew, is_reset, self.extras

    def get_observations(self):
        return torch.cat(
            [
                self.robot.buffer.relative_angle,  # 1
                self.robot.buffer.base_ang_vel * 0.1,  # 3
                self.robot.buffer.base_lin_vel * 0.1,  # 3
                self.robot.buffer.projected_gravity * 0.8,  # 3
                # self.robot.get_turn_pos(),  # 2
                # self.robot.get_wheel_vel() * 0.01,  # 4
                self.robot.robot.get_dofs_position(self.robot.turn_idx),  # 2
                self.robot.robot.get_dofs_velocity(self.robot.wheel_idx) * 0.01,  # 4
                self.actions,  # 4
                self.grid.get_val().reshape(self.num_envs, -1),  # (39 * 39)
            ],
            axis=-1,
        )

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        self.robot.reset_idx(envs_idx=envs_idx)
        self.goal.reset_idx(envs_idx=envs_idx)
        self.corridor.reset_idx(envs_idx=envs_idx)

        extras_episode = self.rewards.reset_idx(envs_idx=envs_idx)
        self.extras["episode"] = extras_episode

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.get_observations(), None
