import torch
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

from algorithms.wan import WanImageToVideo, WanTextToVideo
from datasets.video_base import SingleFrameVideoDataset
from .exp_base import BaseLightningExperiment


class VideoPredictionExperiment(BaseLightningExperiment):
    """
    A video prediction experiment
    """

    compatible_algorithms = dict(
        wan_i2v=WanImageToVideo,
        wan_t2v=WanTextToVideo,
        wan_toy=WanImageToVideo,
    )

    compatible_datasets = dict(
        image_csv=SingleFrameVideoDataset,
        ours_test=SingleFrameVideoDataset,
    )

    def _build_strategy(self):
        from lightning.pytorch.strategies.fsdp import FSDPStrategy

        if self.cfg.strategy == "ddp":
            return super()._build_strategy()
        elif self.cfg.strategy == "fsdp":
            # Fix: Only use device mesh for multi-node setups where HSDP is intended.
            # For single node (or when calculation doesn't match world size), use None.
            if self.cfg.num_nodes > 1:
                # Assuming 8 GPUs per node for calculation consistency with your hardware
                # Adjust logic if nodes have different GPU counts
                if self.cfg.num_nodes >= 8:
                    device_mesh = (self.cfg.num_nodes // 8, 32)
                else: 
                     # Original logic was (1, num_nodes * 4), presumably for 4-GPU nodes?
                     # Let's set it to None to be safe unless we are strictly doing HSDP
                     device_mesh = None
            else:
                device_mesh = None
                
            return FSDPStrategy(
                mixed_precision=MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.bfloat16,
                    buffer_dtype=torch.bfloat16,
                ),
                cpu_offload=True, # Enable CPU Offload to save GPU memory for VAE activations
                auto_wrap_policy=ModuleWrapPolicy(self.algo.classes_to_shard()),
                sharding_strategy="FULL_SHARD",
                # sharding_strategy="HYBRID_SHARD",
                device_mesh=device_mesh,
            )

        else:
            return self.cfg.strategy

    def download_dataset(self):
        dataset = self._build_dataset("training")
