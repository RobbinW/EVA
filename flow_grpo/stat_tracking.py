import numpy as np
import torch


class PerPromptStatTracker:
    def __init__(self, global_std: bool = False):
        self.global_std = global_std
        self.stats = {}
        self.history_prompts = set()

    def update(self, prompts, rewards, stat_type: str = "grpo"):
        prompts = np.array(prompts)
        rewards = np.array(rewards, dtype=np.float64)
        unique = np.unique(prompts)
        advantages = np.empty_like(rewards) * 0.0

        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            if prompt not in self.stats:
                self.stats[prompt] = []
            self.stats[prompt].extend(prompt_rewards)
            self.history_prompts.add(hash(prompt))

        for prompt in unique:
            self.stats[prompt] = np.stack(self.stats[prompt])
            prompt_rewards = rewards[prompts == prompt]
            mean = np.mean(self.stats[prompt], axis=0, keepdims=True)
            if self.global_std:
                std = np.std(rewards, axis=0, keepdims=True) + 1e-4
            else:
                std = np.std(self.stats[prompt], axis=0, keepdims=True) + 1e-4

            if stat_type == "grpo":
                advantages[prompts == prompt] = (prompt_rewards - mean) / std
            elif stat_type == "rwr":
                advantages[prompts == prompt] = prompt_rewards
            elif stat_type == "sft":
                advantages[prompts == prompt] = (
                    torch.tensor(prompt_rewards) == torch.max(torch.tensor(prompt_rewards))
                ).float().numpy()
            elif stat_type == "dpo":
                prompt_advantages = torch.tensor(prompt_rewards)
                max_idx = torch.argmax(prompt_advantages)
                min_idx = torch.argmin(prompt_advantages)
                if max_idx == min_idx:
                    min_idx = 0
                    max_idx = 1
                result = torch.zeros_like(prompt_advantages).float()
                result[max_idx] = 1.0
                result[min_idx] = -1.0
                advantages[prompts == prompt] = result.numpy()

        return advantages

    def get_stats(self):
        avg_group_size = (
            sum(len(v) for v in self.stats.values()) / len(self.stats)
            if self.stats
            else 0
        )
        return avg_group_size, len(self.history_prompts)

    def clear(self):
        self.stats = {}
