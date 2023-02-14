from matplotlib import animation
import matplotlib.pyplot as plt
# import gym 

"""
Ensure you have imagemagick installed with 
sudo apt-get install imagemagick
Open file in CLI with:
xgd-open <filelname>
"""
def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)

# #Make gym env
# env = gym.make('CartPole-v1')

# #Run the env
# observation = env.reset()
# frames = []
# for t in range(1000):
#     #Render to frames buffer
#     frames.append(env.render(mode="rgb_array"))
#     action = env.action_space.sample()
#     _, _, done, _ = env.step(action)
#     if done:
#         break
# env.close()

import gymnasium as gym
env = gym.make('Walker2d-v4')

observation, info = env.reset()
frames = []
for _ in range(1000):
    # frames.append(env.render())
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
# save_frames_as_gif(frames)


# visualization: xvfb-run -s "-screen 0 1400x900x24" jupyter notebook