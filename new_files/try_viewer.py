# dm_control_env2.py: test dm_control creation of an RL environment
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dm_control import suite

# Load one task:
env = suite.load(domain_name="cartpole", task_name="swingup")

action_spec = env.action_spec()
time_step = env.reset()

width = 480
height = 480

fig = plt.figure()
video = np.zeros((1, height, 2 * width, 3), dtype=np.uint8)
im = plt.imshow(video[0], animated=True)

max_frame = 90
frame_count = 0
started = time.time()

def one_step(args):
    action = np.random.uniform(action_spec.minimum, action_spec.maximum, size=action_spec.shape)
    time_step = env.step(action)
    
    frame = np.hstack([env.physics.render(height, width, camera_id=0),
            env.physics.render(height, width, camera_id=1)])

    im.set_array(frame) 

    # update stats
    global frame_count, started

    frame_count += 1
    if frame_count % max_frame == 0:
        elapsed = time.time() - started
        print("frame: {}, elapsed: {:.2f} secs, fps: {:.0f}".format(frame_count, elapsed, max_frame/elapsed))
        started = time.time()

    return im,

ani = animation.FuncAnimation(fig, one_step, interval=0, blit=True, repeat=False)
plt.show()