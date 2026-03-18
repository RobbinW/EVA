# EVA: Aligning Video World Models with Executable Robot Actions via Inverse Dynamics Rewards

<p align="center">
  <a href="https://eva-project-page.github.io/">Project Page</a>
</p>

## Overview

EVA is a post-training framework for aligning video world models with physically executable robot actions.

Recent work explores video generative models as visual planners for robotic manipulation. However, these models often produce rollouts that violate rigid-body and kinematic consistency, producing unstable or infeasible control commands when decoded by an inverse dynamics model. We refer to this mismatch between visual generation and physically executable control as the executability gap.

We introduce **Executable Video Alignment (EVA)**, a reinforcement-learning post-training framework for aligning video world models. EVA trains an inverse dynamics model on real robot trajectories and repurposes it as a reward model that evaluates generated videos through the action sequences they induce, encouraging smooth motions measured by velocity, acceleration, and jerk while penalizing actions that violate embodiment constraints.

## Main Figure

<p align="center">
  <img src="assets/main_figure.png" width="90%">
</p>

## Release Progress

- [x] inference code
- [ ] supervised fine-tuning code
- [ ] reinforcement learning post-training code
- [ ] evaluation on RoboTwin

## Downloading the Checkpoints

### Downloading Our Fine-tuned Checkpoints


You can download EVA model fine-tuned on RoboTwin data:

```bash
huggingface-cli download RobbinWang123/EVA \
  --include "eva_i2v_14B.ckpt" \
  --local-dir ./data/ckpts
```

After downloading, the checkpoint is expected at:

```bash
./data/ckpts/eva_i2v_14B.ckpt
```

### Downloading Wan 2.1 Pre-trained Checkpoints

This codebase uses the **Wan 2.1 Image-to-Video 14B** model as the base model.

Please refer to the official Wan release for the most up-to-date download instructions:

- https://github.com/Wan-Video/Wan2.1

Example download command:

```bash
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P \
  --local-dir ./data/ckpts/Wan2.1-I2V-14B-480P
```

The downloaded directory should contain the Wan diffusion model, VAE, text encoder, and CLIP encoder.

## Environment Setup

### Create Conda Environment

```bash
conda create python=3.10 -n eva
conda activate eva
```

### Install Dependencies

```bash
# Install dependencies
pip install -r requirements.txt

# Install Flash Attention
# This may take several minutes to compile
pip install flash-attn --no-build-isolation
```

## Inference


### Example Inference Command

```bash
CUDA_VISIBLE_DEVICES=0 python -m main \
  +name=demo_infer \
  experiment=exp_inference \
  algorithm=wan_i2v \
  dataset=image_csv \
  dataset.data_root=data/test_images \
  dataset.metadata_path=metadata.csv \
  algorithm.model.tuned_ckpt_path=/path/to/EVA/data/ckpts/eva_i2v_14B.ckpt \
  algorithm.hist_guidance=1.5 \
  algorithm.lang_guidance=2.5 \
  algorithm.logging.video_type=single
```

Generated videos will be written to:

```bash
outputs/<date>/<time>/videos/
```

### Notes

- `algorithm.model.tuned_ckpt_path` should point to the EVA fine-tuned checkpoint.
- The Wan base checkpoint paths can be set in `configurations/algorithm/wan_i2v.yaml`.
- `algorithm.hist_guidance` and `algorithm.lang_guidance` control image and language guidance strength during inference.

## Acknowledgment

We thank the authors of the following repository for their open-source release and code structure:

- large-video-planner: https://github.com/buoyancy99/large-video-planner/tree/main
- Wan2.1: https://github.com/Wan-Video/Wan2.1
