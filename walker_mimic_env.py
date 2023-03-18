import numpy as np
import scipy

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
        "render_fps": 200,
    }
    dt = 0.005
    ep_dt = 0.005
    def __init__(self, ref=False, args=None, **kwargs):
        observation_space = Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float64)
        MuJocoPyEnv.__init__(
            self, "/home/rishia/projects/rl/xml/walker2d_low_freq.xml", 4, observation_space=observation_space, **kwargs
        )
        utils.EzPickle.__init__(self, **kwargs)
        self.motion = np.loadtxt('mocap_data/2dwalker_walk.txt')
        self.motion[:,1] += 1.35
        self.idx_motion = 0
        self.x_diff = 0.1
        self.num_steps = 0
        self.kp = args.kp if args else 1.2 #1.2 # 0.04, 0.08, 0.8
        self.kd = args.kd if args else 0.1 #33 / 1000 # 0.005, 0.01, 0.02
        self.ref = ref
        self.cycle_time = 1
        self.ref_dt = self.cycle_time / (self.motion.shape[0] - 1)
        self.interpolated_ref_motion = scipy.interpolate.interp1d(np.linspace(0, 1, num=self.motion.shape[0]), self.motion, axis=0)
        self.FLY = False
    
    def _increment(self, index):
        assert index < self.motion.shape[0]
        return (index + 1) % self.motion.shape[0]

    def _get_qref_pose(self, phase=None):
        phase = phase if phase is not None else self.frame_skip * self.num_steps * self.ep_dt
        phase = phase - int(phase)
        # print(f"phase={phase}")
        
        # query phase from reference motion
        qref = self.interpolated_ref_motion(phase)
        
        # modify x coordinate based on num_steps
        qref[0] = self.frame_skip * self.num_steps * self.x_diff * (self.ep_dt / self.ref_dt)

        if self.FLY:
            qref[[0,1,2]] = [0, 1.5, 0]
        # else:
        #     qref[0] = 0
        # print(f"phase={phase}, qref={qref}")
        return qref

    def _get_qref_vel(self, phase=None):
        phase = phase if phase is not None else self.frame_skip * self.num_steps * self.ep_dt
        phase = phase - int(phase)
        q1 = self._get_qref_pose(phase)

        # prev_phase = (phase - self.ep_dt) if (phase - self.ep_dt) > 0 else 1 - (phase - self.ep_dt)
        # q2 = self._get_qref_pose(prev_phase)
        # qvel = (q1 - q2) / self.ep_dt
        
        next_phase = (phase + self.ep_dt) - int(phase + self.ep_dt)
        q2 = self._get_qref_pose(next_phase)
        qvel = (q2 - q1) / self.ep_dt

        qvel[0] = self.x_diff / self.ref_dt
        
        if self.FLY:
            qvel[0:3] = 0
        # else:
        #     qvel[0] = 0
        # print(f"phase={phase}, qvel={qvel}")

        return qvel

    # def step(self, pd_a):
    #     qrefpose = self._get_qref_pose()
    #     qvel = (pd_a) / self.ep_dt #(pd_a - self.sim.data.qpos[3:]) / self.ep_dt
    #     a = self.kp * (qrefpose[3:] - pd_a - self.sim.data.qpos[3:]) + self.kd * (0 - qvel)
    #     return self.step_wo_pd(a)
    # {"pose":0.65, "vel":0.1, "end":0, "root":0.15, "com":0.1}
    # {"pose": 1.2, "vel": 0.006, "root": 5, "com": 0.3, "end": np.inf}
    def _calc_imitation_reward(self, w={"pose":0.3, "vel":0.2, "end":0, "root":0.3, "com":0.2}, a={"pose": 3, "vel": 0.03, "root": 6, "com": 2, "end": np.inf}):
        r = dict()
        dist = dict()
        dist["pose"] =  np.sum((self.sim.data.qpos[3:] - self._get_qref_pose()[3:])**2)
        dist["vel"] = np.sum((self.sim.data.qvel[3:] - self._get_qref_vel()[3:])**2)
        dist["end"] = 1
        dist["root"] = np.sum((self.sim.data.qpos[:3] - self._get_qref_pose()[:3])**2)
        dist["com"] = np.sum((self.sim.data.qvel[:3] - self._get_qref_vel()[:3])**2)
        # print(f"dist={dist}")

        for k in w.keys():
            r[k] = np.exp( - a[k] * dist[k])
        if not self.ref:
            print(f"num_steps={self.num_steps}, r={r}, dist={dist}, qpos={self.sim.data.qpos}, qref={self._get_qref_pose()}, qvel={self.sim.data.qvel}, qrefvel={self._get_qref_vel()}")
            # pass
        return sum([r[k] * w[k] for k in w.keys()]), sum([dist[k] * w[k] for k in w.keys()]), dist, r

    def do_simulation_with_pd(self, pd_a, frame_skip):
        qrefpose = self._get_qref_pose()
        for i in range(frame_skip):
            a =  self.kp * (pd_a + qrefpose[3:] - self.sim.data.qpos[3:]) + self.kd * (0 - self.sim.data.qvel[3:])
            a = np.clip(a, -1, 1)
            # print(f"action={a}, pd_a={pd_a}, delta_pose={self.kp * (qrefpose[3:]-self.sim.data.qpos[3:])} delta_vel={self.kd * (0 - self.sim.data.qvel[3:])}")
            q1 = np.copy(self.sim.data.qpos)
            # print(f"pd_a={pd_a}, qpos={self.sim.data.qpos[3:]}, qrefpose={qrefpose[3:]+pd_a}, qvel={self.sim.data.qvel[3:]}, qrefvel={self._get_qref_vel()}, action={a}")
            if self.FLY:
                self.sim.data.qpos[[0,1,2]] = [0,1.5,0]
                self.sim.data.qvel[0:3] = 0
            # else:
            #     self.sim.data.qpos[0] = 0
            #     self.sim.data.qvel[0] = 0
            self.do_simulation(a, 1) # 1 frame at a time
            if not self.ref:
                # print(f"nstep={self.num_steps}, q1={q1}, q2={self.sim.data.qpos}, qref={qrefpose}")
                pass
        if self.FLY:
            self.sim.data.qpos[[0,1,2]] = [0,1.5,0]
            self.sim.data.qvel[0:3] = 0
    
    def step(self, a,  w={"goal": 0., "imitation": 1.0}):
        # print(f"stepping with fly={self.FLY}")
        self.idx_motion = self._increment(self.idx_motion)
        self.num_steps += 1
        posbefore = self.sim.data.qpos[0]
        # print(f"self.sim.data={self.sim.data}, qpos={self.sim.data.qpos}, qpos_shape={self.sim.data.qpos.shape}, idx_motion={self.idx_motion}")
        self.do_simulation_with_pd(a, self.frame_skip)
        
        if self.ref:
            self.set_state(*self._get_mimic_state())
        
        posafter, height, ang = self.sim.data.qpos[0:3]

        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.ep_dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        if not self.ref:
            # print(f'pen={1e-3 * np.square(a).sum()}')
            pass
        imitation_reward, imitation_dist_sum, imitation_dist_info, imitation_r_info = self._calc_imitation_reward()
        # imitation_reward = -1 * imitation_dist_sum
        reward = w["goal"] * reward + w["imitation"] * imitation_reward
        terminated = not (height > 0.8 and height < 2.0 and ang > -1.0 and ang < 1.0)
        ob = self._get_obs()

        if self.render_mode == "human":
            self.render()
        return ob, reward, terminated, False, {"imitation_dist": imitation_dist_info}

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def _get_mimic_state(self):
        qpos = self._get_qref_pose() #self.motion[self.idx_motion]
        qvel = self._get_qref_vel() #(self.motion[(self.idx_motion+1)%(self.motion.shape[0])] - self.motion[self.idx_motion]) / self.ep_dt
        return (qpos, qvel)

    def reset_model(self):
        self.idx_motion = 0
        self.num_steps = 0
        # self.set_state(
        #     self.init_qpos
        #     + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nq),
        #     self.init_qvel
        #     + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv),
        # )
        # if self.ref:
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
    imageio.mimsave('walker-mimic-test-mocap.gif', frames)

    env.close()
