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

print(f"project_root: {project_root}")

# %%
import logging

import torch
from collections import deque
import numpy as np

from robochair.data_handling.dataset import WheelchairSequenceDataset
from robochair.algorithms.diffusion import (
    modeling_diffusion,
    configuration_diffusion,
)
from robochair.algorithms.diffusion.utils import load_ckpt, save_ckpt


from tqdm import tqdm


# %%


checkpoint_path = project_root + "/results/checkpoints_diffusion"
batch_size = 256
num_epoch = 10
grad_clip_norm = 1.0

torch.backends.cudnn.benchmark = True
# torch.backends.cuda.matmul.allow_tf32 = True

device = torch.device("cuda")

dataset = WheelchairSequenceDataset(
    root_dir=project_root + "/data",
    episodes_dir="episodes_ppo",
)

cfg = configuration_diffusion.DiffusionConfig()  # пока так


policy = modeling_diffusion.DiffusionPolicy(cfg).to(device)

params = policy.get_optim_params()
# params = policy.parameters()
optimizer = torch.optim.Adam(params, lr=1e-4)

# step = 0
step = 0
recent_losses = deque(maxlen=100)  # Хранит потери за последние 100 шагов
epoch_avg_losses = []  # Список для хранения средних потерь за каждую эпоху


try:
    step = load_ckpt(policy, optimizer, checkpoint_path)
except:
    print("Чекпоинты не найдены!")
    pass

dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, num_workers=0
)

policy.train()
# %%

num_epoch = 1
for epoch in range(num_epoch):
    epoch_loss_sum = 0.0
    epoch_steps = 0

    print(f"\n--- Начало Эпохи {epoch+1}/{num_epoch} ---")
    for i, batch in enumerate(
        tqdm(dataloader, desc=f"Эпоха {epoch+1}", leave=False, total=4242)
    ):

        loss, output_dict = policy.forward(batch)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        current_loss = loss.item()
        recent_losses.append(current_loss)
        epoch_loss_sum += current_loss
        epoch_steps += 1
        step += 1

        if step % 50 == 0 and len(recent_losses) > 0:
            avg_loss_100 = np.mean(recent_losses)
            print(
                f"  Шаг {step}, Эпоха {epoch+1}, Средняя ошибка (100 шагов): {avg_loss_100:.4f}"
            )

        if step % 1000 == 0:
            save_ckpt(policy, optimizer, step, checkpoint_path)

    if epoch_steps > 0:
        avg_epoch_loss = epoch_loss_sum / epoch_steps
        epoch_avg_losses.append(avg_epoch_loss)
        save_ckpt(policy, optimizer, step, checkpoint_path)
        print(
            f"--- Эпоха {epoch+1} Завершена --- Средняя ошибка эпохи: {avg_epoch_loss:.4f} ---"
        )
# %%
