import math
from typing import Optional, Union

import torch
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils.torch_utils import randn_tensor


def sde_step_with_logprob(
    self: UniPCMultistepScheduler,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    prev_sample: Optional[torch.FloatTensor] = None,
    generator: Optional[torch.Generator] = None,
    determistic: bool = False,
    return_dt_and_std_dev_t: bool = False,
):
    model_output = model_output.float()
    sample = sample.float()
    if prev_sample is not None:
        prev_sample = prev_sample.float()

    step_index = [self.index_for_timestep(t) for t in timestep]
    prev_step_index = [step + 1 for step in step_index]

    self.sigmas = self.sigmas.to(sample.device)
    sigma = self.sigmas[step_index].view(-1, 1, 1, 1, 1)
    sigma_prev = self.sigmas[prev_step_index].view(-1, 1, 1, 1, 1)
    sigma_max = self.sigmas[1].item()
    sigma_min = self.sigmas[-1].item()
    dt = sigma_prev - sigma

    std_dev_t = sigma_min + (sigma_max - sigma_min) * sigma
    prev_sample_mean = (
        sample * (1 + std_dev_t**2 / (2 * sigma) * dt)
        + model_output * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
    )

    if prev_sample is not None and generator is not None:
        raise ValueError(
            "Cannot pass both generator and prev_sample. Please make sure only one is set."
        )

    if prev_sample is None:
        variance_noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )
        prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1 * dt) * variance_noise

    if determistic:
        prev_sample = sample + dt * model_output

    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2)
        / (2 * ((std_dev_t * torch.sqrt(-1 * dt)) ** 2))
        - torch.log(std_dev_t * torch.sqrt(-1 * dt))
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    if return_dt_and_std_dev_t:
        return prev_sample, log_prob, prev_sample_mean, std_dev_t, torch.sqrt(-1 * dt)
    return prev_sample, log_prob, prev_sample_mean, std_dev_t * torch.sqrt(-1 * dt)
