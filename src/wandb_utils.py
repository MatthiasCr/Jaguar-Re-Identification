import os

import wandb

WANDB_TEAM_NAME = "juggling-jaguars"
WANDB_PROJECT_NAME = "jaguar-reid-jugglingjaguars"

def init_wandb(run_config, run_name, param_count):
    wandb_config = {
        key: (str(value) if hasattr(value, "__fspath__") else value)
        for key, value in run_config.items()
    }
    if param_count is not None:
        wandb_config["model_param_count"] = param_count

    api_key = os.getenv("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)
    else:
        wandb.login()

    wandb.init(
        entity=WANDB_TEAM_NAME,
        project=WANDB_PROJECT_NAME,
        config=wandb_config,
        name=run_name,
    )
    return wandb


def log_image(name, fig):
    wandb.log({name: wandb.Image(fig)})
