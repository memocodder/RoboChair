import torch
from pyglet.window import key


class AgentControl:
    def __init__(self):
        self.move_forward_backward = 0.0
        self.turn_left_right = 0.0
        self.head = 0.0
        self.eyes = 0.0

    def set_movement(self, value):
        self.move_forward_backward = float(value)

    def set_turning(self, value):
        self.turn_left_right = float(value)

    def set_head(self, value):
        self.head = float(value)

    def to_action_tensor(self, device="cuda", dtype=torch.float32):
        action_list = [
            self.move_forward_backward,
            self.turn_left_right,
            self.eyes,
            self.head,
        ]

        action_tensor = torch.tensor(action_list, device=device, dtype=dtype)
        return action_tensor.unsqueeze(0)

    def get_registered_keys(self):
        return {
            key.LEFT: {
                "press": lambda s: self.set_turning(
                    1.0
                ),  # 's' - это self объекта-обработчика
                "release": lambda s: self.set_turning(0.0),
            },
            key.RIGHT: {
                "press": lambda s: self.set_turning(-1.0),
                "release": lambda s: self.set_turning(0.0),
            },
            key.UP: {
                "press": lambda s: self.set_movement(1.0),
                "release": lambda s: self.set_movement(0.0),
            },
            key.DOWN: {
                "press": lambda s: self.set_movement(-1.0),
                "release": lambda s: self.set_movement(0.0),
            },
            # key.SPACE: { ... }
        }
