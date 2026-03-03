import os
from pathlib import Path

import wandb

WANDB_TEAM_NAME = "juggling-jaguars"
WANDB_PROJECT_NAME = "jaguar-reid-jugglingjaguars"

def init_wandb(run_config, run_name, param_count, param_count_backbone=None):
    wandb_config = {
        key: (str(value) if hasattr(value, "__fspath__") else value)
        for key, value in run_config.items()
    }
    wandb_config["model_param_count"] = param_count
    if param_count_backbone is not None:
        wandb_config["backbone_param_count"] = param_count_backbone

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


def log_checkpoint_artifact(wandb_run, checkpoint_path, artifact_name, description="Model checkpoint"):
    checkpoint_path = Path(checkpoint_path)
    artifact = wandb.Artifact(
        name=artifact_name,
        type="model",
        description=description,
    )
    artifact.add_file(str(checkpoint_path))
    wandb_run.log_artifact(artifact)


def log_submission_artifact(wandb_run, submission_path, artifact_name, description="Competition submission file"):
    submission_path = Path(submission_path)
    artifact = wandb.Artifact(
        name=artifact_name,
        type="submission",
        description=description,
    )
    artifact.add_file(str(submission_path))
    wandb_run.log_artifact(artifact)
