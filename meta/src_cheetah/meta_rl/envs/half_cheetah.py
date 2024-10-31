import os, sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(os.path.join(parent_dir))
sys.path.append(os.path.join(parent_dir, 'envs'))

import numpy as np
from gymnasium import utils
from gymnasium.spaces import Box
from mujoco_gym.mujoco_env import MujocoEnv
import mujoco as mj

class HalfCheetahEnv(MujocoEnv, utils.EzPickle):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}

    def __init__(self, **kwargs):
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float64)

        self.xml_file: str = "/Users/mintaekim/Desktop/HRL/Quadrotor/quadrotor_meta_fucked/meta/assets/half_cheetah.xml"
        self.frame_skip: int = 5
        self.reset_noise_scale: float = 1.0

        self.model = mj.MjModel.from_xml_path(self.xml_file)
        self.data = mj.MjData(self.model)

        utils.EzPickle.__init__(self, self.xml_file, self.frame_skip, self.reset_noise_scale, **kwargs)
        MujocoEnv.__init__(self, self.xml_file, self.frame_skip, observation_space=self.observation_space, **kwargs)

    def _reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()
    
    def _get_obs(self):
        return np.concatenate([self.data.qpos.flat[1:], self.data.qvel.flat])
    
    def step(self, action):
        xposbefore = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.data.qpos[0]

        obs = self._get_obs()
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run
        terminated = False

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return (obs, reward, terminated, False, dict(reward_run=reward_run, reward_ctrl=reward_ctrl))

    def viewer_setup(self) -> None:
        camera_id = self.model.camera_name2id("track")
        self.viewer.cam.type = 2
        self.viewer.cam.fixedcamid = camera_id
        self.viewer.cam.distance = self.model.stat.extent * 0.35
        self.viewer._hide_overlay = True
