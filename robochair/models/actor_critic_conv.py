# %%

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torchvision.ops import MLP
import math
from configs.algorithms.ppo_algo_cfg import PPOPolicyConfig


class EncoderEmulator(
    nn.Module
):  # Ипользуется в самом начале, научиться следовать цели без препятствий на пути
    def __init__(self, output_dim=128, input_dim=(1, 39, 39)):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # ЗАГЛУШКА: Позже этот класс нужно будет заменить на реальный
        # энкодер (возможно MobileNet??) + слой Linear(D_large -> output_dim)

    def forward(self, x):
        # Сейчас вход 'x' (изображения) игнорируется
        # Просто возвращаем тензор нулей нужной формы (batch_size, output_dim)
        batch_size = x.shape[0]  # Получаем размер батча из любого входа
        device = (
            x.device
        )  # Используем то же устройство (cpu/gpu), что и у входа
        return torch.zeros(batch_size, self.output_dim, device=device)
        # ПОЗЖЕ ЗДЕСЬ БУДЕТ: return self.real_vision_encoder(x)


class VisEncoder(nn.Module):
    def __init__(self, output_dim=128, input_dim=(1, 39, 39)):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if not (isinstance(input_dim, (tuple, list)) and len(input_dim) == 3):
            raise ValueError(
                "input_dim must be a tuple or list of length 3 (channels, height, width)"
            )

        self.input_channels = input_dim[0]
        self.input_height = input_dim[1]
        self.input_width = input_dim[2]

        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                self.input_channels, 16, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(
                1, self.input_channels, self.input_height, self.input_width
            ).cpu()
            conv_output = self.conv_layers(dummy_input)
            self._flattened_size = conv_output.view(1, -1).shape[1]

        self.mlp_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._flattened_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.output_dim),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.mlp_head(x)
        return x


if __name__ == "__main__":
    example_input_dim = (1, 39, 39)
    dummy_input = torch.randn(4, *example_input_dim)
    encoder = VisEncoder(output_dim=128, input_dim=example_input_dim)
    print(f"Input shape: {dummy_input.shape}")
    output_vector = encoder(dummy_input)
    print(f"Output shape: {output_vector.shape}")

    if torch.cuda.is_available():
        print("\nTesting on GPU...")
        device = torch.device("cuda")
        encoder.to(device)
        dummy_input_gpu = dummy_input.to(device)
        output_vector_gpu = encoder(dummy_input_gpu)
        print(f"Output shape on GPU: {output_vector_gpu.shape}")
    else:
        print("\nGPU not available.")

# %%


class ObsEncoder(nn.Module):
    def __init__(self, input_dim, output_dim=128, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(),
        )

    def forward(self, obs_data):
        return self.mlp(obs_data)


class GenericAgentNetwork(nn.Module):
    def __init__(
        self,
        obs_encoder,
        vision_encoder,
        output_dim,
        mode="actor",
        hidden_dim1=256,
        hidden_dim2=128,
    ):

        super().__init__()
        # Проверяем корректность режима
        if mode not in ["actor", "critic"]:
            raise ValueError("mode должен быть 'actor' или 'critic'")

        self.mode = mode
        self.output_dim = output_dim

        # Сохраняем энкодеры
        self.obs_encoder = obs_encoder
        self.vision_encoder = vision_encoder

        combined_feature_dim = (
            obs_encoder.output_dim + vision_encoder.output_dim
        )

        self.mlp = nn.Sequential(
            nn.Linear(combined_feature_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim2, output_dim),
        )

    def forward(self, obs_img_data):
        batch_size = obs_img_data.shape[0]
        sensor_size = self.obs_encoder.input_dim

        sensor_data = obs_img_data[:, :sensor_size]
        image_features_flat = obs_img_data[:, sensor_size:]

        image_data = image_features_flat.view(
            batch_size, *self.vision_encoder.input_dim
        )

        encoded_obs = self.obs_encoder(sensor_data)
        encoded_vision = self.vision_encoder(image_data)

        combined_features = torch.cat((encoded_obs, encoded_vision), dim=1)
        output = self.mlp(combined_features)

        if self.mode == "actor":
            return torch.tanh(output)
        else:
            return output


# %%


class ActorCriticConv(nn.Module):

    def __init__(
        self, num_actor_obs, num_actions, cfg: PPOPolicyConfig, image_shape=(1, 39, 39)
    ):

        super(ActorCriticConv, self).__init__()

        self.image_shape = image_shape
        self.image_dim = np.prod(image_shape)
        self.num_actor_obs = num_actor_obs

        num_vector_features = num_actor_obs - self.image_dim

        sensor_enc_shared = ObsEncoder(
            input_dim=num_vector_features, output_dim=128
        )
        vision_enc_shared = VisEncoder(output_dim=128, input_dim=image_shape)
        # vision_enc_shared = EncoderEmulator(output_dim=128, input_dim=image_shape)

        self.actor = GenericAgentNetwork(
            obs_encoder=sensor_enc_shared,
            vision_encoder=vision_enc_shared,
            output_dim=num_actions,
            mode="actor",
        )

        # Создаем критика (с output_dim=1)
        self.critic = GenericAgentNetwork(
            obs_encoder=sensor_enc_shared,
            vision_encoder=vision_enc_shared,
            output_dim=1,  # У критика выход всегда 1
            mode="critic",
        )

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        init_noise_std = 1.0
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        self.s_flag = True

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(
                mod for mod in sequential if isinstance(mod, nn.Linear)
            )
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value


# %%
# --- Небольшой тест ---
if __name__ == "__main__":

    # 1. Определяем параметры для теста
    num_vector_features_test = 20
    image_shape_test = (1, 39, 39)  # Каналы, Высота, Ширина
    num_actions_test = 4  # Примерное количество действий
    batch_size_test = 8  # Небольшой размер батча для теста

    image_dim_test = np.prod(image_shape_test)  # 39 * 39 = 1521
    num_actor_obs_test = num_vector_features_test + image_dim_test
    num_critic_obs_test = num_actor_obs_test

    print(f"--- Параметры теста ---")
    print(f"Размер векторных фич: {num_vector_features_test}")
    print(f"Размер картинки (CxHxW): {image_shape_test}")
    print(f"Размер плоской картинки: {image_dim_test}")
    print(f"Полный размер Actor Obs: {num_actor_obs_test}")
    print(f"Полный размер Critic Obs: {num_critic_obs_test}")
    print(f"Количество действий: {num_actions_test}")
    print(f"Размер батча: {batch_size_test}")
    print("-" * 25)

    # 3. Создаем экземпляр сети ActorCriticConv
    # Используем 'cpu' для простоты теста, если нет GPU или не настроен CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")

    model = ActorCriticConv(
        num_actor_obs=num_actor_obs_test,
        # num_critic_obs=num_critic_obs_test,
        num_actions=num_actions_test,
        image_shape=image_shape_test,
    ).to(device)

    # # %%
    # OLD_CHECKPOINT_PATH = "/home/o/Documents/smart-car/logs/model_pre_vis_emul.pt"
    # NEW_CHECKPOINT_PATH = "/home/o/Documents/smart-car/logs/model_pre_with_clear_visencoder.pt"
    # loaded_dict_old = torch.load(OLD_CHECKPOINT_PATH, map_location='cpu')
    # saved_state_dict = loaded_dict_old['model_state_dict']
    # saved_iter = loaded_dict_old.get('iter', 0)
    # saved_infos = loaded_dict_old.get('infos', None)
    # print(f"Loaded old state dict with {len(saved_state_dict)} parameter tensors.")

    # # model.load_state_dict(loaded_dict['model_state_dict'])

    # # %%
    # initial_new_state_dict = model.state_dict()
    # print(f"Instantiated new model with {len(initial_new_state_dict)} parameter tensors.")
    # # %%
    # import copy

    # merged_state_dict = copy.deepcopy(initial_new_state_dict)
    # updated_keys_count = 0
    # mismatched_keys_count = 0

    # for key, saved_param in saved_state_dict.items():
    #     if key in merged_state_dict:
    #         # Ключ есть в обеих моделях, проверяем совпадение размеров
    #         if merged_state_dict[key].shape == saved_param.shape:
    #             # Размеры совпадают - копируем веса из старого чекпоинта
    #             merged_state_dict[key].copy_(saved_param)
    #             updated_keys_count += 1
    #         else:
    #             # Размеры не совпадают - оставляем веса новой модели (случайные)
    #             print(f"  [!] Shape mismatch for key '{key}'. Keeping initial weight. "
    #                 f"(Saved: {saved_param.shape}, Current: {merged_state_dict[key].shape})")
    #             mismatched_keys_count += 1
    #     # else: Ключа из старого словаря нет в новом - просто игнорируем (это нормально)

    # print(f"Merge complete:")
    # print(f"  Updated {updated_keys_count} parameter tensors with weights from the old checkpoint.")
    # if mismatched_keys_count > 0:
    #     print(f"  Skipped {mismatched_keys_count} parameter tensors due to shape mismatch (kept initial).")
    # initial_keys_kept = len(initial_new_state_dict) - updated_keys_count
    # print(f"  Kept initial weights for {initial_keys_kept} parameter tensors (expected for VisEncoder and potentially mismatched keys).")

    # print(f"Saving NEW checkpoint to: {NEW_CHECKPOINT_PATH}")
    # # Создаем финальный словарь для сохранения
    # # ВАЖНО: Не сохраняем старый несовместимый optimizer_state_dict
    # new_save_dict = {
    #     'model_state_dict': merged_state_dict, # Содержит старые веса + новые для VisEncoder
    #     'optimizer_state_dict': None, # Или пустой словарь {}
    #     'iter': saved_iter,
    #     'infos': saved_infos,
    # }
    # torch.save(new_save_dict, NEW_CHECKPOINT_PATH)

    # %%
    # Устанавливаем модель в режим оценки (отключает dropout и т.п., если есть)
    model.eval()

    # 4. Генерируем два батча случайных данных (на нужном устройстве)
    # Данные имитируют конкатенированный вектор + плоское изображение
    dummy_actor_observations = torch.randn(
        batch_size_test, num_actor_obs_test
    ).to(device)
    dummy_critic_observations = torch.randn(
        batch_size_test, num_critic_obs_test
    ).to(device)

    print(f"\n--- Запуск теста ---")
    print(f"Форма входных данных Actor: {dummy_actor_observations.shape}")
    print(f"Форма входных данных Critic: {dummy_critic_observations.shape}")

    # 5. Выполняем прямой проход для получения действий и оценки
    try:
        with torch.no_grad():  # Отключаем расчет градиентов для теста/инференса
            # Получаем действия (используем act_inference, который не требует distribution)
            actions = model.act_inference(dummy_actor_observations)

            # Получаем оценку состояния
            value = model.evaluate(dummy_critic_observations)

        print("\n--- Результаты ---")
        print(f"Форма выходных действий (Actions): {actions.shape}")
        print(f"Форма выходной оценки (Value): {value.shape}")

        # 6. Проверяем ожидаемые размерности
        expected_actions_shape = torch.Size([batch_size_test, num_actions_test])
        expected_value_shape = torch.Size([batch_size_test, 1])

        assert (
            actions.shape == expected_actions_shape
        ), f"Ожидалось {expected_actions_shape}, получено {actions.shape}"
        assert (
            value.shape == expected_value_shape
        ), f"Ожидалось {expected_value_shape}, получено {value.shape}"

        print(
            "\nТест пройден успешно! Размерности выходов соответствуют ожидаемым."
        )

    except Exception as e:
        print(f"\nОШИБКА во время выполнения теста: {e}")
        import traceback

        traceback.print_exc()
