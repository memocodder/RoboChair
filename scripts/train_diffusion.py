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


from configs.log_config import LoggingConfig, setup_logger

log_config = LoggingConfig(
    base_log_dir=project_root + "/results",
    experiment_name="RobochairGenesisDiffusionTrain_Simple_v1",
    experiment_description="Обучение диффузионной политики на экспертных записях PPO.",
    log_level="INFO",
    overwrite_existing=True,
)

logger = setup_logger(log_config)
experiment_path = log_config.experiment_dir


from robochair.algorithms.diffusion import DiffusionRunner

from configs.diffusion_config import DiffusionAlgoConfig, DiffusionRunnerConfig

# %%

algo_configuration = DiffusionAlgoConfig(
    runner=DiffusionRunnerConfig(checkpoint_dir=experiment_path)
)

runner = DiffusionRunner(algo_configuration, logger)

runner.load_checkpoint()
runner.learn(project_root + "/data", "episodes_ppo")

# %%

 # веса должны сохраниться в experiment_path, что б не дублировать 
 # веса можно указать путь к чекпоинтам експеримента в scripts/eval_diffusion.py
runner.save_checkpoint(checkpoint_dir=project_root + "trained_models")
