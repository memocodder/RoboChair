import torch
from torch.utils.data import IterableDataset, DataLoader
from pathlib import Path
import re
import random
from typing import List, Dict, Tuple, Iterator, Union, Optional


class WheelchairSequenceDataset(IterableDataset):
    """
    IterableDataset, который загружает данные эпизодов инвалидной коляски
    и возвращает СЛУЧАЙНЫЕ ПОСЛЕДОВАТЕЛЬНОСТИ наблюдений и действий,
    подходящие для обучения DiffusionPolicy.
    """

    DIR_NAME_PATTERN = re.compile(r"^(\d+)_(\d+)$")
    # Оставляем старые имена для загрузки файлов, но будем использовать
    # стандартные ключи lerobot в выходном словаре.
    FEATURE_FILENAMES = {
        "grids": "grids.pt",
        "state": "obs.pt",  # Назовем внутренне 'state' для ясности
        "rewards": "rewards.pt",  # Пока не используется в выходе
        "actions": "actions.pt",
    }
    # Ключи для загрузки из файлов (порядок важен для _load_episode)
    LOAD_FEATURES_ORDER = ["grids", "state", "rewards", "actions"]

    def __init__(
        self,
        root_dir: Union[str, Path] = "data",
        episodes_dir: str = "episodes",
        n_obs_steps: int = 2,  # Количество шагов наблюдений в последовательности
        diffusion_horizon: int = 16,  # Количество шагов действий в последовательности (== config.horizon)
        episode_chunk_size: int = 10,  # Сколько эпизодов загружать в память за раз
        # Ключ(и) для данных сетки, как они будут в выходном словаре
        # Должны совпадать с config.image_features
        grid_keys: Union[str, List[str]] = "observation.images",
        device: Union[str, torch.device] = "cuda",
    ):
        super().__init__()
        self.root = Path(root_dir).resolve()
        self.episodes_dir = self.root / episodes_dir
        if not self.episodes_dir.is_dir():
            raise FileNotFoundError(
                f"Директория с эпизодами не найдена: {self.episodes_dir}"
            )

        if n_obs_steps < 1:
            raise ValueError("n_obs_steps должен быть >= 1")
        if diffusion_horizon < 1:
            raise ValueError("diffusion_horizon должен быть >= 1")

        self.n_obs_steps = n_obs_steps
        self.diffusion_horizon = diffusion_horizon
        # Минимальная длина эпизода для генерации хотя бы одной последовательности
        self.min_episode_len = max(n_obs_steps, diffusion_horizon)
        self.episode_chunk_size = episode_chunk_size
        self.device = torch.device(device)

        # Обрабатываем ключ(и) для сетки
        if isinstance(grid_keys, str):
            self.grid_keys = [grid_keys]
        elif isinstance(grid_keys, list) and len(grid_keys) == 1:
            self.grid_keys = grid_keys  # Пока поддерживаем только одну сетку
        else:
            # TODO: Расширить для поддержки нескольких сеток, если нужно
            raise NotImplementedError(
                "Поддерживается только один ключ сетки (одна камера)"
            )
        self.output_grid_key = self.grid_keys[0]  # Ключ для выходного словаря

        self.lengths = []

        self.episode_paths = self._find_and_sort_episodes()
        if not self.episode_paths:
            raise ValueError(
                f"В директории {self.episodes_dir} не найдено валидных эпизодов."
            )
        
        self._calculated_len: Optional[int] = None

        print(f"Датасет инициализирован для директории: {self.episodes_dir}")
        print(f"Найдено {len(self.episode_paths)} эпизодов.")
        print(f"Длина посл-ти наблюдений (n_obs_steps): {self.n_obs_steps}")
        print(
            f"Длина посл-ти действий (diffusion_horizon): {self.diffusion_horizon}"
        )
        print(f"Минимальная длина эпизода: {self.min_episode_len}")
        print(f"Размер чанка для загрузки: {self.episode_chunk_size} эпизодов")
        print(f"Целевое устройство для тензоров: {self.device}")

    def _find_and_sort_episodes(self) -> List[Path]:
        # (без изменений)
        valid_episodes = []
        for item in self.episodes_dir.iterdir():
            if item.is_dir():
                match = self.DIR_NAME_PATTERN.match(item.name)
                if match:
                    try:
                        index = int(match.group(1))
                        self.lengths.append(int(match.group(2)))
                        # Проверяем наличие всех необходимых файлов
                        if all(
                            (item / self.FEATURE_FILENAMES[key]).exists()
                            for key in self.LOAD_FEATURES_ORDER
                        ):
                            valid_episodes.append((index, item))
                        else:
                            print(
                                f"Предупреждение: В эпизоде {item.name} отсутствуют некоторые файлы. Пропускаем."
                            )
                    except ValueError:
                        continue
        valid_episodes.sort(key=lambda x: x[0])
        return [path for index, path in valid_episodes]
    

    def __len__(self) -> int:
        if self._calculated_len is not None:
            return self._calculated_len

        print("Вычисление общей длины датасета по списку длин...")
        total_sequences = 0
        processed_episodes = 0
        skipped_episodes_len = 0

        for ep_len in self.lengths:
            if ep_len >= self.min_episode_len:
                num_valid_starts = ep_len - max(self.n_obs_steps, self.diffusion_horizon) + 1
                if num_valid_starts > 0:
                    total_sequences += num_valid_starts
                    processed_episodes += 1
                else:
                    skipped_episodes_len += 1
            else:
                skipped_episodes_len += 1

        print(f"Вычисление длины завершено.")
        print(f"  Всего эпизодов найдено (по списку длин): {len(self.lengths)}")
        print(f"  Эпизодов, учтенных в длине (>= {self.min_episode_len} шагов): {processed_episodes}")
        print(f"  Эпизодов, пропущено из-за длины: {skipped_episodes_len}")
        print(f"  Общее количество последовательностей: {total_sequences}")

        self._calculated_len = total_sequences
        return self._calculated_len

    def _load_episode(
        self, episode_path: Path
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Загружает все данные для одного эпизода на self.device."""
        try:
            loaded_tensors = {}
            for feature in self.LOAD_FEATURES_ORDER:
                filename = self.FEATURE_FILENAMES[feature]
                loaded_tensors[feature] = torch.load(
                    episode_path / filename, map_location=self.device
                )

            # Проверка консистентности длины
            n_frames = loaded_tensors[self.LOAD_FEATURES_ORDER[0]].shape[0]
            if not all(t.shape[0] == n_frames for t in loaded_tensors.values()):
                print(
                    f"Предупреждение: Несоответствие кол-ва шагов в эпизоде {episode_path.name}. Пропускаем."
                )
                return None

            # Добавляем проверку минимальной длины
            if n_frames < self.min_episode_len:
                print(
                    f"Предупреждение: Эпизод {episode_path.name} слишком короткий ({n_frames} < {self.min_episode_len}). Пропускаем."
                )
                return None

            return loaded_tensors

        except FileNotFoundError as e:
            print(
                f"Предупреждение: Ошибка FileNotFoundError при загрузке {episode_path.name}. Пропускаем. {e}"
            )
            return None
        except Exception as e:
            print(
                f"Предупреждение: Ошибка при загрузке эпизода {episode_path.name}. Пропускаем. {e}"
            )
            return None

    def _generate_sequences_from_chunk(
        self, episode_paths_chunk: List[Path]
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Загружает чанк эпизодов и генерирует из них все возможные
        валидные последовательности.
        """
        sequences_buffer = []
        for episode_path in episode_paths_chunk:
            episode_data = self._load_episode(episode_path)
            if episode_data is None:
                continue  # Пропускаем, если эпизод не загрузился или слишком короткий

            ep_len = episode_data[self.LOAD_FEATURES_ORDER[0]].shape[0]
            ep_grids = episode_data["grids"]  # (L, C, H, W)
            ep_state = episode_data["state"]  # (L, state_dim)
            ep_actions = episode_data["actions"]  # (L, action_dim)
            # ep_rewards = episode_data["rewards"] # (L,) - пока не используем

            ep_actions = torch.clip(ep_actions, -1.0, 1.0)
            ep_actions = torch.nan_to_num(ep_actions, nan=0.0)
            ep_state = torch.clip(ep_state, -4.0, 4.0)
            ep_state = torch.nan_to_num(ep_state, nan=0.0)

            # Определяем, сколько валидных стартовых позиций есть в эпизоде
            # Последний старт i: i + max(n_obs, n_act) - 1 < L => i < L - max + 1
            num_valid_starts = (
                ep_len - max(self.n_obs_steps, self.diffusion_horizon) + 1
            )

            for i in range(num_valid_starts):
                # Индексы для срезов
                obs_start, obs_end = i, i + self.n_obs_steps
                act_start, act_end = (
                    i,
                    i + self.diffusion_horizon,
                )  # Действия начинаются с того же шага i

                # Извлекаем последовательности
                obs_seq = ep_state[
                    obs_start:obs_end
                ]  # (n_obs_steps, state_dim)
                grid_seq = ep_grids[obs_start:obs_end]  # (n_obs_steps, C, H, W)
                action_seq = ep_actions[
                    act_start:act_end
                ]  # (diffusion_horizon, action_dim)
                # Создаем маску для действий (пока предполагаем, что нет паддинга в исходных данных)
                # Форма (diffusion_horizon,)
                pad_seq = torch.zeros(
                    self.diffusion_horizon, dtype=torch.bool, device=self.device
                )

                # Формируем выходной словарь с ключами lerobot
                sequence_dict = {
                    "observation.state": obs_seq,  # Используем константу/строку "observation.state"
                    self.output_grid_key: grid_seq,  # Используем ключ из init, н-р "observation.grid"
                    "action": action_seq,
                    "action_is_pad": pad_seq,
                    # Можно добавить другие данные при необходимости, например, episode_index
                    # "episode_index": torch.tensor(episode_path.name.split('_')[0], dtype=torch.long)
                }
                sequences_buffer.append(sequence_dict)

        return sequences_buffer

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Итератор, который загружает чанки эпизодов, генерирует из них
        последовательности, перемешивает их и выдает по одной.
        """
        num_episodes = len(self.episode_paths)
        # Перемешиваем порядок загрузки эпизодов для каждой эпохи
        shuffled_episode_paths = random.sample(self.episode_paths, num_episodes)

        for i in range(0, num_episodes, self.episode_chunk_size):
            episode_chunk_to_load = shuffled_episode_paths[
                i : min(i + self.episode_chunk_size, num_episodes)
            ]
            if not episode_chunk_to_load:
                continue  # Пропускаем, если чанк пустой

            # Генерируем все последовательности из этого чанка
            sequences_in_chunk = self._generate_sequences_from_chunk(
                episode_chunk_to_load
            )
            if not sequences_in_chunk:
                # print(f"Предупреждение: В чанке эпизодов {i//self.episode_chunk_size} не найдено валидных последовательностей.")
                continue  # Пропускаем, если не удалось сгенерировать ни одной посл-ти

            # Перемешиваем последовательности ВНУТРИ чанка
            random.shuffle(sequences_in_chunk)

            # Выдаем последовательности из перемешанного буфера
            for sequence in sequences_in_chunk:
                yield sequence


# --- Пример использования ---
if __name__ == "__main__":
    import time

    # Замените на ваши реальные константы/значения из конфига
    DATA_ROOT_DIR = "data"
    N_OBS_STEPS = 2
    DIFFUSION_HORIZON = 16  # Должно совпадать с config.horizon
    EPISODE_CHUNK_SIZE = 10
    DATALOADER_BATCH_SIZE = (
        256  # Уменьшим для примера, т.к. последовательности больше весят
    )
    GRID_KEY = "observation.grid"  # Как в конфиге diffusion

    try:
        dataset = WheelchairSequenceDataset(
            root_dir=DATA_ROOT_DIR,
            n_obs_steps=N_OBS_STEPS,
            diffusion_horizon=DIFFUSION_HORIZON,
            episode_chunk_size=EPISODE_CHUNK_SIZE,
            grid_keys=GRID_KEY,
            device=(
                "cuda" if torch.cuda.is_available() else "cpu"
            ),  # Автовыбор устройства
        )

        dataloader = DataLoader(
            dataset,
            batch_size=DATALOADER_BATCH_SIZE,
            num_workers=0,  # IterableDataset плохо работает с num_workers > 0 в простых случаях
        )

        print("\nНачало итерации через DataLoader (пример 5 батчей)...")
        start_time = time.time()
        batches_processed = 0
        total_sequences = 0

        # Используем cycle для удобства, если нужно много батчей
        # dl_iter = cycle(dataloader)
        # for i in range(5):
        #     batch = next(dl_iter)

        # Переменные для хранения общих min/max и флага NaN
        overall_min_state, overall_max_state = float("inf"), float("-inf")
        overall_min_action, overall_max_action = float("inf"), float("-inf")
        overall_min_grid, overall_max_grid = float("inf"), float("-inf")
        found_nan_state, found_nan_action, found_nan_grid = False, False, False

        for i, batch in enumerate(dataloader):
            # if i >= 10: # Ограничим вывод 5 батчами для примера
            #     break

            batches_processed += 1
            current_batch_size = batch[OBS_ROBOT].shape[0]
            total_sequences += current_batch_size

            # print(f"\n--- Батч DataLoader {i} ---")
            # print(f"  Кол-во последовательностей в батче: {current_batch_size}")
            # print(f"  Ключи в батче: {list(batch.keys())}")
            # print(f"  Устройство: {batch[OBS_ROBOT].device}")
            # print(
            #     f"  Форма {OBS_ROBOT}: {batch[OBS_ROBOT].shape}"
            # )  # (B, n_obs_steps, state_dim)
            # print(
            #     f"  Форма {GRID_KEY}: {batch[GRID_KEY].shape}"
            # )  # (B, n_obs_steps, C, H, W)
            # print(
            #     f"  Форма action: {batch['action'].shape}"
            # )  # (B, diffusion_horizon, action_dim)

            # print(
            #     f"  Пример action[0]: {batch['action'][0].tolist()}"
            # )

            # print(
            #     f"  Форма action_is_pad: {batch['action_is_pad'].shape}"
            # )  # (B, diffusion_horizon)
            # print(
            #     f"  Пример action_is_pad[0]: {batch['action_is_pad'][0].tolist()}"
            # )  # Ожидаем все False

            # --- Проверки для текущего батча ---
            state_tensor = batch[OBS_ROBOT]
            action_tensor = batch["action"]
            grid_tensor = batch[GRID_KEY]

            # Проверка state
            min_state_batch = torch.min(state_tensor).item()
            max_state_batch = torch.max(state_tensor).item()
            has_nan_state_batch = torch.isnan(state_tensor).any().item()
            print(
                f"  {OBS_ROBOT}: Min={min_state_batch:.4f}, Max={max_state_batch:.4f}, NaN присутствует={has_nan_state_batch}"
            )
            overall_min_state = min(overall_min_state, min_state_batch)
            overall_max_state = max(overall_max_state, max_state_batch)
            if has_nan_state_batch:
                found_nan_state = True

            # Проверка action
            min_action_batch = torch.min(action_tensor).item()
            max_action_batch = torch.max(action_tensor).item()
            has_nan_action_batch = torch.isnan(action_tensor).any().item()
            print(
                f"  action: Min={min_action_batch:.4f}, Max={max_action_batch:.4f}, NaN присутствует={has_nan_action_batch}"
            )
            overall_min_action = min(overall_min_action, min_action_batch)
            overall_max_action = max(overall_max_action, max_action_batch)
            if has_nan_action_batch:
                found_nan_action = True

            # Проверка grid
            min_grid_batch = torch.min(grid_tensor).item()
            max_grid_batch = torch.max(grid_tensor).item()
            has_nan_grid_batch = torch.isnan(grid_tensor).any().item()
            print(
                f"  {GRID_KEY}: Min={min_grid_batch:.4f}, Max={max_grid_batch:.4f}, NaN присутствует={has_nan_grid_batch}"
            )
            overall_min_grid = min(overall_min_grid, min_grid_batch)
            overall_max_grid = max(overall_max_grid, max_grid_batch)
            if has_nan_grid_batch:
                found_nan_grid = True

            # print(f"  Форма {OBS_ROBOT}: {state_tensor.shape}")
            # print(f"  Форма {GRID_KEY}: {grid_tensor.shape}")
            # print(f"  Форма action: {action_tensor.shape}")

        end_time = time.time()
        print("\n--- Итоги Примера ---")
        print(f"Обработано батчей: {batches_processed}")
        print(f"Всего последовательностей: {total_sequences}")
        if batches_processed > 0:
            print(
                f"Среднее время на батч: {(end_time - start_time) / batches_processed:.4f} сек"
            )

        print("\n--- Общая статистика ---")
        print(f"  {OBS_ROBOT}:")
        print(f"    Общий Min: {overall_min_state:.4f}")
        print(f"    Общий Max: {overall_max_state:.4f}")
        print(f"    Найдены NaN: {'Да' if found_nan_state else 'Нет'}")
        print(f"  action:")
        print(f"    Общий Min: {overall_min_action:.4f}")
        print(f"    Общий Max: {overall_max_action:.4f}")
        print(f"    Найдены NaN: {'Да' if found_nan_action else 'Нет'}")
        print(f"  {GRID_KEY}:")
        print(f"    Общий Min: {overall_min_grid:.4f}")
        print(f"    Общий Max: {overall_max_grid:.4f}")
        print(f"    Найдены NaN: {'Да' if found_nan_grid else 'Нет'}")

    except FileNotFoundError as e:
        print(f"\nОШИБКА: Не удалось найти директорию с данными!")
        print(f"Детали: {e}")
    except ValueError as e:
        print(f"\nОШИБКА: Проблема с данными или параметрами!")
        print(f"Детали: {e}")
    except Exception as e:
        print(f"\nНеожиданная ОШИБКА: {e}")
        import traceback

        traceback.print_exc()  # Печатаем traceback для неожиданных ошибок
