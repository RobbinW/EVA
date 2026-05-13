import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

from idm import IDM


def plot_actions(gt_actions, pred_actions, save_dir, sample_idx):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    num_steps = gt_actions.shape[0]
    num_dims = gt_actions.shape[1]
    cols = 4
    rows = math.ceil(num_dims / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.flatten()
    time_steps = np.arange(num_steps)

    for i in range(num_dims):
        ax = axes[i]
        if not np.all(gt_actions[:, i] == 0):
            ax.plot(
                time_steps,
                gt_actions[:, i],
                color="blue",
                label="Ground Truth",
                linewidth=1,
            )
        if i < pred_actions.shape[1]:
            ax.plot(
                time_steps,
                pred_actions[:, i],
                color="red",
                label="Predicted",
                linestyle="--",
                linewidth=1,
            )
        ax.set_title(f"Dimension {i}")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Value")

        vals = [gt_actions[:, i]]
        if i < pred_actions.shape[1]:
            vals.append(pred_actions[:, i])
        vals = np.concatenate(vals)
        if np.all(vals >= -1.2) and np.all(vals <= 1.2):
            ax.set_ylim(-1.2, 1.2)

        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.7)

    for i in range(num_dims, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(save_dir / f"sample_{sample_idx}_comparison.png")
    plt.close()


class IDMSmoothnessReward:
    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        model_name: str = "resnet_plus",
        num_frames: int = 3,
        save_dir: str | None = None,
    ):
        self.device = device
        self.num_frames = num_frames
        self.save_dir = save_dir
        self.net = None

        self.net = IDM(model_name=model_name, output_dim=14, num_frames=num_frames)

        loaded_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = loaded_dict["model_state_dict"] if "model_state_dict" in loaded_dict else loaded_dict
        self.net.load_state_dict(state_dict, strict=False)
        self.net.to(device)
        self.net.eval()

        self.resize = transforms.Resize((512, 512))
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def preprocess_video(self, video_tensor):
        video_tensor = self.resize(video_tensor)
        return self.normalize(video_tensor)

    def predict_actions(self, video_tensor):
        if self.net is None:
            return torch.zeros((video_tensor.shape[0], 14), device=self.device)

        processed_frames = self.preprocess_video(video_tensor)
        num_frames = processed_frames.shape[0]
        actions = []

        with torch.no_grad():
            for i in range(num_frames):
                if self.num_frames == 1:
                    indices = [i]
                elif self.num_frames == 2:
                    indices = [i - 1, i]
                elif self.num_frames == 3:
                    indices = [i - 1, i, i + 1]
                else:
                    indices = [i] * self.num_frames

                indices = [max(0, min(idx, num_frames - 1)) for idx in indices]
                stack = [processed_frames[idx] for idx in indices]
                img_input = torch.stack(stack).to(self.device).unsqueeze(0)
                output = self.net(img_input, None)
                actions.append(output)

        return torch.cat(actions, dim=0)

    def compute_smoothness(self, actions):
        if actions.shape[0] < 4:
            return 0.0

        dt = 1.0 / 16.0
        _, dims = actions.shape
        device, dtype = actions.device, actions.dtype

        v_arm = torch.tensor(
            [3.1416, 3.4034, 3.1416, 3.9270, 3.9270, 3.9270],
            device=device,
            dtype=dtype,
        ) * 0.85

        v_max = torch.zeros(dims, device=device, dtype=dtype)
        v_max[0:6] = v_arm
        v_max[7:13] = v_arm
        v_max[6] = 2.0
        v_max[13] = 2.0

        a_max = torch.full((dims,), 5.0, device=device, dtype=dtype)
        a_max[6] = 10.0
        a_max[13] = 10.0

        v = (actions[1:] - actions[:-1]) / dt
        a = (v[1:] - v[:-1]) / dt
        j = (a[1:] - a[:-1]) / dt

        def huber(x, delta):
            ax = x.abs()
            return torch.where(ax <= delta, 0.5 * (x**2), delta * (ax - 0.5 * delta))

        weights = torch.ones(dims, device=device, dtype=dtype)
        weights[6] = 0.2
        weights[13] = 0.2
        weights = weights / (weights.mean() + 1e-8)

        delta_a = a_max * 0.5
        delta_j = (a_max / dt) * 0.5

        acc_pen = (huber(a, delta_a) * weights).mean()
        jerk_pen = (huber(j, delta_j) * weights).mean()

        v_violate = torch.relu(v.abs() - v_max)
        a_violate = torch.relu(a.abs() - a_max)
        vlim_pen = ((v_violate**2) * weights).mean()
        alim_pen = ((a_violate**2) * weights).mean()

        penalty_raw = (
            0.5 * jerk_pen
            + 0.25 * acc_pen
            + 2.0 * vlim_pen
            + 1.0 * alim_pen
        )


        P0 = 2000.0      # try 2000 ~ 5000

        gamma = 0.5      # 0.5 is very safe for GRPO
        MaxReward = 10.0

        score = MaxReward * torch.pow(1.0 + penalty_raw / P0, -gamma)

        return score.item()

    def __call__(self, images, prompts, metadata):
        rewards = []

        images = (images.float() + 1.0) / 2.0
        images = torch.clamp(images, 0.0, 1.0)

        batch_size, _, _, _, _ = images.shape
        should_plot = self.save_dir is not None

        for i in range(batch_size):
            video = images[i].permute(1, 0, 2, 3)
            actions = self.predict_actions(video)
            score = self.compute_smoothness(actions)
            rewards.append(score)

            if should_plot and i == 0:
                prompt_snippet = prompts[i][:30].replace(" ", "_") if i < len(prompts) else "unknown"
                phase = "unknown"
                step_info = ""
                rank_str = ""

                if metadata and isinstance(metadata, list) and len(metadata) > i:
                    meta_item = metadata[i]
                    if isinstance(meta_item, dict):
                        phase = meta_item.get("phase", "unknown")
                        step = meta_item.get("step", "")
                        epoch = meta_item.get("epoch", "")
                        rank = meta_item.get("rank", None)
                        sample_idx = meta_item.get("sample_idx", None)
                        if step != "":
                            step_info = f"_step{step}"
                        if epoch != "":
                            step_info += f"_epoch{epoch}"
                        if sample_idx is not None:
                            step_info += f"_s{sample_idx}"
                        if rank is not None:
                            rank_str = f"_rank{rank}"

                sample_name = f"{phase}_video_{prompt_snippet}{step_info}{rank_str}"
                gt = np.zeros_like(actions.cpu().numpy())
                plot_actions(gt, actions.cpu().numpy(), self.save_dir, sample_name)

        return {"idm_smoothness": np.array(rewards)}, metadata


def idm_smoothness_score(
    device,
    checkpoint_path,
    num_frames: int = 3,
    save_dir: str | None = None,
):
    scorer = IDMSmoothnessReward(
        checkpoint_path=checkpoint_path,
        device=device,
        num_frames=num_frames,
        save_dir=save_dir,
    )

    def _fn(images, prompts, metadata):
        return scorer(images, prompts, metadata)

    return _fn
