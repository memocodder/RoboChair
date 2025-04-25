import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import time
import logging
from collections import deque
import numpy as np
from tqdm import tqdm
import sys

from .utils import save_ckpt, load_ckpt
from robochair.data_handling.dataset import WheelchairSequenceDataset
from robochair.models.diffusion import DiffusionPolicy
from configs.diffusion_config import (
    DiffusionAlgoConfig,
    DiffusionRunnerConfig,
    DiffusionPolicyConfig,
    TrainingConfig,
)


class DiffusionRunner:
    """
    Runner для оффлайн обучения Diffusion Policy.
    """

    def __init__(
        self, config: DiffusionAlgoConfig, logger_instance: logging.Logger
    ):
        self.config: DiffusionAlgoConfig = config
        self.runner_cfg: DiffusionRunnerConfig = config.runner
        self.policy_cfg: DiffusionPolicyConfig = config.policy
        self.training_cfg: TrainingConfig = config.policy.training
        self.logger = logger_instance
        self.device: str = config.device

        self.checkpoint_dir: str = self.runner_cfg.checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.logger.info(f"Устройство: {self.device}")
        self.logger.info(f"Директория чекпоинтов: {self.checkpoint_dir}")

        # Модель и Оптимизатор
        self.policy: torch.nn.Module = DiffusionPolicy(self.policy_cfg).to(
            self.device
        )
        self.optimizer: torch.optim.Optimizer = self._create_optimizer()

        # Состояние обучения
        self.writer: SummaryWriter = SummaryWriter(log_dir=self.checkpoint_dir)
        self.global_step: int = 0

        # Загрузка чекпоинта при инициализации
        self.load_checkpoint()

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Создает оптимизатор."""

        optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.training_cfg.learning_rate,
            betas=self.training_cfg.betas,
            eps=self.training_cfg.eps,
            weight_decay=self.training_cfg.weight_decay,
        )
        self.logger.info(
            f"Оптимизатор AdamW создан с lr={self.training_cfg.learning_rate}."
        )
        return optimizer

    def load_checkpoint(self):
        """Загружает последний чекпоинт (модель и оптимизатор)."""
        self.logger.info(f"Попытка загрузки чекпоинта из {self.checkpoint_dir}")
        # load_ckpt возвращает шаг СЛЕДУЮЩИЙ за сохраненным
        start_step = load_ckpt(
            model=self.policy,
            optimizer=self.optimizer,  # Передаем оптимизатор для загрузки его состояния
            ckpt_dir=self.checkpoint_dir,
            device=torch.device(self.device),
        )
        # load_ckpt возвращает 0 если ничего не загружено или ошибка
        if start_step > 0:
            self.global_step = (
                start_step - 1
            )  # Устанавливаем текущий шаг как последний сохраненный
            self.logger.info(
                f"Чекпоинт загружен. Возобновление с шага {self.global_step + 1}."
            )
        else:
            self.global_step = 0
            self.logger.info(
                "Чекпоинты не найдены или ошибка загрузки. Обучение начнется с шага 0."
            )

    def save_checkpoint(self, checkpoint_dir=None):
        """Сохраняет чекпоинт (модель и оптимизатор)."""
        self.logger.debug(f"Сохранение чекпоинта на шаге {self.global_step}")
        try:
            # Передаем модель, оптимизатор, шаг и директорию
            save_ckpt(
                model=self.policy,
                optimizer=self.optimizer,
                step=self.global_step,
                ckpt_dir=checkpoint_dir or self.checkpoint_dir,
            )
            self.logger.info(f"Чекпоинт сохранен для шага {self.global_step}.")
        except Exception as e:
            self.logger.error(
                f"Ошибка сохранения чекпоинта на шаге {self.global_step}: {e}"
            )

    def learn(self, dataset_root_dir: str, episodes_subdir: str):
        """
        Основной цикл оффлайн обучения. БЕЗ ПЛАНИРОВЩИКА.
        """
        num_epochs = self.runner_cfg.total_train_epochs or 1
        batch_size = self.runner_cfg.batch_size
        log_interval = self.runner_cfg.log_interval_steps
        save_interval = self.runner_cfg.save_interval_steps

        try:
            dataset = WheelchairSequenceDataset(
                root_dir=dataset_root_dir, episodes_dir=episodes_subdir
            )
            # Уменьшаем num_workers до 0 для простоты и избежания проблем с multiprocessing
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=0,
            )
            self.logger.info(
                f"Датасет загружен ({len(dataset)} записей). Размер батча: {batch_size}. num_workers=0."
            )
        except Exception as e:
            self.logger.error(
                f"Ошибка создания датасета/загрузчика из {dataset_root_dir}/{episodes_subdir}: {e}"
            )
            return

        recent_losses = deque(maxlen=max(1, log_interval))
        start_time = time.time()
        self.logger.info(f"Начало обучения на {num_epochs} эпох...")

        for epoch in range(num_epochs):
            self.policy.train()
            pbar = tqdm(
                dataloader, desc=f"Эпоха {epoch + 1}/{num_epochs}", leave=False
            )

            for batch in pbar:
                # Остановка по шагам
                if (
                    self.runner_cfg.total_train_steps
                    and self.global_step >= self.runner_cfg.total_train_steps
                ):
                    self.logger.info(
                        f"Достигнуто {self.runner_cfg.total_train_steps} шагов. Остановка."
                    )
                    break

                # --- Шаг Обучения ---
                try:

                    loss, _ = self.policy(batch)
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    current_loss = loss.item()
                    recent_losses.append(current_loss)

                except Exception as e:
                    self.logger.error(
                        f"Ошибка на шаге обучения {self.global_step}: {e}"
                    )
                    continue  # Пропускаем шаг

                self.global_step += 1

                # --- Логирование ---
                if self.global_step % log_interval == 0:
                    avg_loss = np.mean(recent_losses) if recent_losses else 0
                    self.writer.add_scalar(
                        "Loss/train_step", current_loss, self.global_step
                    )
                    self.writer.add_scalar(
                        "Loss/train_avg_recent", avg_loss, self.global_step
                    )
                    self.writer.add_scalar(
                        "LearningRate",
                        self.training_cfg.learning_rate,
                        self.global_step,
                    )  # Логируем константный LR
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

                # --- Сохранение ---
                if self.global_step % save_interval == 0:
                    self.save_checkpoint()

            # Проверка остановки по шагам после эпохи
            if (
                self.runner_cfg.total_train_steps
                and self.global_step >= self.runner_cfg.total_train_steps
            ):
                break

        total_time = time.time() - start_time
        self.logger.info(
            f"Обучение завершено. Шагов: {self.global_step}. Время: {total_time:.2f} сек."
        )
        self.writer.close()
        self.save_checkpoint()  # Финальное сохранение

    def get_inference_policy(self) -> torch.nn.Module:
        """Возвращает модель в режиме инференса."""
        self.policy.eval()
        self.logger.info("Политика переведена в режим eval() для инференса.")
        return self.policy
