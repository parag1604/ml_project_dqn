import gym
import copy
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image as im
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

from dqn.infrastructure.dqn_utils import create_atari_q_network, wrap_deepmind


env = gym.make("PongNoFrameskip-v4")
q_net = create_atari_q_network((84, 84, 4), env.action_space.n)
q_net.load_state_dict(torch.load("dqn/data/dqn_PongDouble_PongNoFrameskip-v4_24-04-2022_12-26-32/agent_itr_7150000.pt"))
env = wrap_deepmind(env)

obs_arr = [None, None, None, env.reset()]
k = 0
zero_filter = np.zeros((5, 5, 4), dtype=int)
while True:
    if obs_arr[0] is None:
        ac_pred = 0
    else:
        obs = np.array([np.moveaxis(np.array(obs_arr).squeeze(), 0, -1)])
        out_original = q_net(torch.tensor(obs)).detach().cpu().numpy().squeeze()
        ac_pred = out_original.argmax()
        smap = np.zeros_like(obs[0, :, :, -1])
        for i in range(80):
            for j in range(80):
                img = copy.deepcopy(obs)
                img[0, i:i+5, j:j+5] = zero_filter
                t = q_net(torch.tensor(img)).detach().cpu().numpy().squeeze()
                smap[i+2, j+2] = np.power(t - out_original, 2).sum()
        smap = np.asarray(256 * (smap / smap.max()), dtype=int)
        plt.imshow(obs[0, :, :, -1])
        plt.savefig("dqn/visualizations/pong/img"+str(k)+".png")
        # plt.show()
        plt.imshow(smap)
        plt.savefig("dqn/visualizations/pong/smap"+str(k)+".png")
        # plt.show()
    obs, rew, done, _ = env.step(ac_pred)
    obs_arr = obs_arr[1:]
    obs_arr.append(obs)
    if done:
        break
    k += 1
env.close()

for i in tqdm(range(3, 410)):
    img = "dqn/visualizations/pong/img" + str(i) + ".png"
    smap = "dqn/visualizations/pong/smap" + str(i) + ".png"
    img = mpimg.imread(img)[57:425, 143:514]
    smap = mpimg.imread(smap)[57:425, 143:514]
    plt.imshow(smap)
    plt.imshow(img, alpha=0.6)
    plt.axis('off')
    plt.savefig("dqn/visualizations/pong/sup"+str(i)+".png", bbox_inches='tight')
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img)
    ax1.set_title('CNN Input')
    ax1.axis('off')
    ax2.imshow(smap)
    ax2.set_title('Saliency Map')
    ax2.axis('off')
    plt.tight_layout()
    plt.savefig("dqn/visualizations/pong/sbs"+str(i)+".png", bbox_inches='tight')
    plt.close()

sbss = []
sups = []
for i in tqdm(range(3, 410)):
    sbs = "dqn/visualizations/pong/sbs" + str(i) + ".png"
    sbs = im.open(sbs)
    sbss.append(sbs)
    sup = "dqn/visualizations/pong/sup" + str(i) + ".png"
    sup = im.open(sup)
    sups.append(sup)

sups[0].save('dqn/visualizations/pong_sup.gif',
    save_all=True, append_images=sups[1:], optimize=True,
    duration=80, loop=0)
sbss[0].save('dqn/visualizations/pong_sbs.gif',
    save_all=True, append_images=sbss[1:], optimize=True,
    duration=80, loop=0)
