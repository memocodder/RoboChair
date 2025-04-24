import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import re
from typing import Optional

def save_ckpt(model: nn.Module, optimizer: optim.Optimizer, step: int, ckpt_dir: str):
    """Сохраняет модель и оптимизатор."""
    p = Path(ckpt_dir)
    p.mkdir(parents=True, exist_ok=True)
    ckpt_path = p / f"ckpt_{step:09d}.pth"
    torch.save({
        'step': step,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, ckpt_path)

def load_ckpt(model: nn.Module, optimizer: optim.Optimizer, ckpt_dir: str, load_step: Optional[int] = None, device: Optional[torch.device] = None) -> int:
    """Загружает модель и оптимизатор, возвращает шаг для возобновления."""
    p = Path(ckpt_dir)
    if not p.is_dir(): return 0

    if load_step is not None:
        ckpt_path = p / f"ckpt_{load_step:09d}.pth"
        if not ckpt_path.is_file(): return 0
    else:
        files = list(p.glob("ckpt_*.pth"))
        if not files: return 0
        latest_step = -1
        ckpt_path = None
        for f in files:
            match = re.search(r"ckpt_(\d+)\.pth", f.name)
            if match:
                step = int(match.group(1))
                if step > latest_step:
                    latest_step = step
                    ckpt_path = f
        if ckpt_path is None: return 0

    try:
        map_loc = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt = torch.load(ckpt_path, map_location=map_loc)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        return ckpt['step'] + 1
    except Exception:
        return 0 # Возврат 0 при любой ошибке загрузки

# --- Пример ---
if __name__ == "__main__":
    # Создать
    m = nn.Linear(10, 2)
    o = optim.Adam(m.parameters())
    d = "policies/checkpoints"

    # Сохранить
    save_ckpt(m, o, 100, d)
    save_ckpt(m, o, 200, d)

    # Загрузить (последний)
    m2 = nn.Linear(10, 2)
    o2 = optim.Adam(m2.parameters())
    start_step = load_ckpt(m2, o2, d)
    print(f"Resume from step: {start_step}") # Ожидаем 201

    # Загрузить (конкретный)
    m3 = nn.Linear(10, 2)
    o3 = optim.Adam(m3.parameters())
    start_step_100 = load_ckpt(m3, o3, d, load_step=100)
    print(f"Resume from step: {start_step_100}") # Ожидаем 101

    # Загрузить (несуществующий)
    m4 = nn.Linear(10, 2)
    o4 = optim.Adam(m4.parameters())
    start_step_none = load_ckpt(m4, o4, d, load_step=999)
    print(f"Resume from step: {start_step_none}") # Ожидаем 0