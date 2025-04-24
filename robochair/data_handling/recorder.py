import torch
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Set
import re
import shutil


class EpisodeRecorder:

    DIR_NAME_PATTERN = re.compile(r"^(\d+)_(\d+)$")
    FEATURE_FILENAMES = {
        "grids": "grids.pt",
        "obs": "obs.pt",
        "rewards": "rewards.pt",
        "actions": "actions.pt",
    }
    FEATURES = list(FEATURE_FILENAMES.keys())

    def __init__(
        self,
        root_dir: Union[str, Path] = "data",
        episodes_dir: str = "episodes",
    ):

        self.root = Path(root_dir)
        self.episodes_dir = self.root / episodes_dir
        self.episodes_dir.mkdir(parents=True, exist_ok=True)

        # Инициализация буфера для текущего эпизода
        self.buffer: Dict[str, List[torch.Tensor]] = {}
        self.current_episode_steps: int = 0
        self.start_new_episode()  # Сразу готовим к записи первого эпизода

        print(
            f"Рекордер инициализирован. Директория для эпизодов: {self.episodes_dir.resolve()}"
        )

    def start_new_episode(self):
        self.buffer = {feature: [] for feature in self.FEATURES}
        self.current_episode_steps = 0

    def add_step(self, obs, grids, rewards, actions):

        self.buffer["obs"].append(obs)
        self.buffer["grids"].append(grids)
        self.buffer["rewards"].append(rewards)
        self.buffer["actions"].append(actions)
        self.current_episode_steps += 1

    def _get_last_episode_index(self) -> int:
        """Находит наибольший индекс эпизода, сканируя существующие директории."""
        max_index = -1
        for item in self.episodes_dir.iterdir():
            if item.is_dir():
                match = self.DIR_NAME_PATTERN.match(item.name)
                if match:
                    try:
                        index = int(match.group(1))
                        max_index = max(max_index, index)
                    except ValueError:
                        continue  # Игнорируем папки с нечисловым индексом
        return max_index

    def save_episode(self) -> Optional[Path]:

        if self.current_episode_steps == 0:
            print("Предупреждение: Нет данных для сохранения текущего эпизода.")
            return None

        episode_length = self.current_episode_steps
        episode_index = self._get_last_episode_index() + 1
        dir_name = f"{episode_index}_{episode_length}"
        episode_path = self.episodes_dir / dir_name

        print(
            f"Сохранение эпизода {episode_index} (длина {episode_length}) в {episode_path}..."
        )

        episode_path.mkdir(
            exist_ok=False
        )  # exist_ok=False чтобы не перезаписать случайно

        for feature, tensor_list in self.buffer.items():
            if not tensor_list:
                raise ValueError(
                    f"Отсутсвуют данные для {feature} в эписоде {episode_index}!"
                )

            try:
                stacked_tensor = torch.stack(tensor_list, dim=0)
            except Exception as e:
                print(
                    f"Ошибка: Не удалось объединить тензоры для признака '{feature}'. Проверьте консистентность данных шагов."
                )
                shutil.rmtree(episode_path)
                raise RuntimeError(
                    f"Ошибка объединения тензоров для '{feature}'"
                ) from e

            file_path = episode_path / self.FEATURE_FILENAMES[feature]
            torch.save(stacked_tensor, file_path)

        print(f"Эпизод {episode_index} успешно сохранен.")

        self.start_new_episode()
        return episode_path
