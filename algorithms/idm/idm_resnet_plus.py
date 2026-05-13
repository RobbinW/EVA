from algorithms.common.base_algo import BaseAlgo
from idm import IDM


class IDMResNetPlus(BaseAlgo):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model_name = "resnet_plus"

    def build_model(self):
        return IDM(
            model_name=self.model_name,
            output_dim=self.cfg.get("output_dim", 14),
            num_frames=self.cfg.get("num_frames", 3),
            backbone_type=self.cfg.get("backbone_type", "resnet50"),
        )

    def run(self, *args, **kwargs):
        raise NotImplementedError("IDMResNetPlus is used through experiment=exp_idm.")
