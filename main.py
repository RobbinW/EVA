"""
Main entry for EVA.

The public phase-1 release is inference-first, but we keep the same
Hydra/experiment/algorithm/dataset structure that will later host
supervised training and RL post-training.
"""

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from omegaconf.omegaconf import open_dict

from utils.print_utils import cyan
from utils.distributed_utils import is_rank_zero
from utils.ckpt_utils import is_run_id


def run_local(cfg: DictConfig):
    # delay some imports in case they are not needed in non-local envs for submission
    from experiments import build_experiment

    # Get yaml names
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    cfg_choice = OmegaConf.to_container(hydra_cfg.runtime.choices)

    with open_dict(cfg):
        if cfg_choice["experiment"] is not None:
            cfg.experiment._name = cfg_choice["experiment"]
        if cfg_choice["dataset"] is not None:
            cfg.dataset._name = cfg_choice["dataset"]
        if cfg_choice["algorithm"] is not None:
            cfg.algorithm._name = cfg_choice["algorithm"]

    # Set up the output directory.
    output_dir = Path(hydra_cfg.runtime.output_dir)
    if is_rank_zero:
        print(cyan(f"Outputs will be saved to:"), output_dir)
        (output_dir.parents[1] / "latest-run").unlink(missing_ok=True)
        (output_dir.parents[1] / "latest-run").symlink_to(
            output_dir, target_is_directory=True
        )

    # Resolve ckpt path
    resume = cfg.get("resume", None)
    load = cfg.get("load", None)
    checkpoint_path = None
    load_id = None
    if load and not is_run_id(load):
        checkpoint_path = load
    if resume:
        load_id = resume
    elif load and is_run_id(load):
        load_id = load
    else:
        load_id = None

    if load_id:
        run_path = f"{cfg.wandb.entity}/{cfg.wandb.project}/{load_id}"
        checkpoint_path = Path("outputs/downloaded") / run_path / "model.ckpt"

    # launch experiment
    experiment = build_experiment(cfg, output_dir, checkpoint_path)

    # for those who are searching, this is where we call tasks like 'training, validation, main'
    for task in cfg.experiment.tasks:
        experiment.exec_task(task)


@hydra.main(
    version_base=None,
    config_path="configurations",
    config_name="config",
)
def run(cfg: DictConfig):
    if "name" not in cfg:
        raise ValueError(
            "must specify a name for the run with command line argument '+name=[name]'"
        )

    if cfg.wandb.mode == "online" and not cfg.wandb.get("entity", None):
        raise ValueError(
            "wandb.entity is required when wandb.mode=online"
        )

    if cfg.wandb.project is None:
        cfg.wandb.project = str(Path(__file__).parent.name)

    # If resuming or loading a wandb ckpt and not on a compute node, download the checkpoint.
    resume = cfg.get("resume", None)
    load = cfg.get("load", None)

    if resume and load:
        raise ValueError(
            "When resuming a wandb run with `resume=[wandb id]`, checkpoint will be loaded from the cloud"
            "and `load` should not be specified."
        )

    if resume:
        load_id = resume
    elif load and is_run_id(load):
        load_id = load
    else:
        load_id = None

    if load_id:
        raise NotImplementedError(
            "wandb checkpoint download is not included in the public phase-1 EVA release. "
            "Please pass a local checkpoint path with load=/path/to/model.ckpt."
        )

    if cfg.get("cluster") is not None:
        raise NotImplementedError(
            "Cluster submission is not included in the public phase-1 EVA release."
        )

    run_local(cfg)


if __name__ == "__main__":
    run()
