# %%

import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
current_dir_name = os.path.basename(script_dir)

if current_dir_name == "scripts":
    project_root = os.path.dirname(script_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
else:
    project_root = current_dir_name

print(f'project_root: {project_root}')

# обязательно реализовать релоауды

# %%

from configs.log_config import LoggingConfig, setup_logger

log_config = LoggingConfig(
    base_log_dir=project_root + "/results",
    experiment_name="RobochairGenesisPPO_Play_Simple_v1",
    experiment_description="Запись, следовать цели в простом широком коридоре.",
    log_level="WARNING",
    overwrite_existing=True,
)

logger = setup_logger(log_config)
experiment_path = log_config.experiment_dir


import torch

from configs.genesis_config import GenesisEnvConfig

from robochair.environments import Env
 # Управление с помощью клавишь ВВЕРХ ВНИЗ ВЛЕВО ВПРАВО!!!
from robochair.environments.agent_control import AgentControl
from robochair.data_handling.recorder import EpisodeRecorder

import matplotlib.pyplot as plt

env_configuration = GenesisEnvConfig()


NUM_ENVS = 1

control = AgentControl() # Управление с помощью клавишь ВВЕРХ ВНИЗ ВЛЕВО ВПРАВО!!!

env = Env(
    env_configuration,
    num_envs=NUM_ENVS,
    show_viewer=True,
    gta_cam=True,
    # arches_using=True,  # не работает
    registered_keys=control.get_registered_keys(),
)

recorders = {}
for i in range(NUM_ENVS):
    recorders[i] = EpisodeRecorder(
        root_dir=project_root + "/data", episodes_dir="episodes"
    )

# %%

def split_img_and_obs(obs_img):
    image_size = 39 * 39
    image_shape = (1, 39, 39)
    batch_size = obs_img.shape[0]

    image = obs_img[:, -image_size:].view(batch_size, *image_shape)
    obs = obs_img[:, :-image_size]
    return image, obs


obs, _ = env.reset()

for i in range(1105):
    head_rot = env.robot.buffer.relative_angle[0]
    # head_rot = -0.5
    control.set_head(head_rot)
    actions = control.to_action_tensor(device="cuda", dtype=torch.float32)
    obs, _, rews, dones, infos = env.step(actions)
    grid, obs_ = split_img_and_obs(obs)

    # for env_idx in range(NUM_ENVS):
    #     current_recorder = recorders[env_idx]
    #     current_recorder.add_step(obs_[env_idx], grid[env_idx], rews[env_idx], actions[env_idx])
    #     if dones[env_idx] == 1:
    #         save_path = current_recorder.save_episode()

    # recorder.add_step(obs, grid, rews, actions)
    # if dones[0] == 1:
    #     recorder.save_episode()


# %%
a  = torch.tensor([])
imege_data = grid[0].squeeze(0).cpu().numpy()
plt.imshow(imege_data)
plt.show()
