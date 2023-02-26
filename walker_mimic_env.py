import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MuJocoPyEnv
from gymnasium.spaces import Box


class Walker2dEnv(MuJocoPyEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 125,
    }

    def __init__(self, **kwargs):
        observation_space = Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float64)
        MuJocoPyEnv.__init__(
            self, "walker2d.xml", 4, observation_space=observation_space, **kwargs
        )
        utils.EzPickle.__init__(self, **kwargs)
        self.motion = np.loadtxt('mocap_data/2dwalker_walk.txt')
        self.motion[:,1] += 1.5
        self.idx_motion = 0
        self.x_diff = 0.1
        self.num_steps = 0
        self.kp = 1
        self.kd = 1

    def _increment(self, index):
        assert index < self.motion.shape[0]
        return (index + 1) % self.motion.shape[0]

    def _get_qtarget_pose(self, index=None):
        index = self.idx_motion if index is None else index
        qtarget = self.motion[index]
        qtarget[0] = self.num_steps * self.x_diff
        return qtarget

    def _get_qtarget_vel(self, index=None):
        index = self.idx_motion if index is None else index
        q1 = self._get_qtarget_pose(index)
        q2 = self._get_qtarget_pose(self._increment(index))
        qvel = (q2 - q1) / self.dt
        return qvel

    def pd_control_step(self, pd_a):
        qtargetpose = self._get_qtarget_pose()
        qvel = (self.pd_a - self.sim.data.qpos[3:]) / self.dt
        a = self.kp * (qtargetpose[3:] - pd_a) + self.kd * (0 - qvel)
        return self.step(a)

    def _calc_imitation_reward(self, w={"pose":0.5, "vel":0.05, "end":0.15, "root":0.1, "com":0.2}):
        r = dict()
        r["pose"] = np.exp(-2 * np.sum((self.sim.data.qpos[2:] - self._get_qtarget_pose()[2:])**2))
        r["vel"] = np.exp(-0.1 * np.sum((self.sim.data.qvel[2:] - self._get_qtarget_vel()[2:])**2))
        r["end"] = 0
        r["root"] = np.exp(-40 * np.sum((self.sim.data.qpos[:2] - self._get_qtarget_pose()[:2])**2))
        r["com"] = np.exp(-10 * np.sum((self.sim.data.qvel[:2] - self._get_qtarget_vel()[:2])**2))

        return sum([r[k] * w[k] for k in w.keys()])


    def step(self, a,  w={"goal": 0.2, "imitation": 0.8}):
        self.idx_motion = self._increment(self.idx_motion)
        self.num_steps += 1
        posbefore = self.sim.data.qpos[0]
        # print(f"self.sim.data={self.sim.data}, qpos={self.sim.data.qpos}, qpos_shape={self.sim.data.qpos.shape}, idx_motion={self.idx_motion}")
        self.do_simulation(a, self.frame_skip)
        # self.set_state(*self._get_mimic_state())
        
        posafter, height, ang = self.sim.data.qpos[0:3]

        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        reward = w["goal"] * reward + w["imitation"] * self._calc_imitation_reward()
        terminated = not (height > 0.8 and height < 2.0 and ang > -1.0 and ang < 1.0)
        ob = self._get_obs()

        if self.render_mode == "human":
            self.render()

        return ob, reward, terminated, False, {}

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def _get_mimic_state(self):
        qpos = self.motion[self.idx_motion]
        qvel = (self.motion[(self.idx_motion+1)%(self.motion.shape[0])] - self.motion[self.idx_motion]) / self.dt
        return (qpos, qvel)

    def reset_model(self):
        self.idx_motion = 0
        self.set_state(
            self.init_qpos
            + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv),
        )
        self.set_state(*self._get_mimic_state())
        return self._get_obs()

    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

if __name__ == '__main__':
    import gymnasium as gym
    from gymnasium.envs.registration import register
    import imageio

    register(
        id='WalkerMimic-v4',
        entry_point='walker_mimic_env:Walker2dEnv',
        max_episode_steps=300,
    )


    env = gym.make('WalkerMimic-v4', render_mode="rgb_array")


    observation, info = env.reset()
    frames = []
    for i in range(100):
        print(i)
        frames.append(env.render())
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()
        # env.render()
    # print(frames[0])
    imageio.mimsave('walker-mimic-test.gif', frames)

    env.close()